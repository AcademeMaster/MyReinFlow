"""
Noisy Flow MLP for PPO Flow algorithm - adds learnable exploration noise to flow models.
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Tuple, Optional
import numpy as np
from reflow.networks.flow_mlp import FlowMLP
from reflow.networks.embeddings import SinusoidalPosEmb
from reflow.networks.mlp import MLP


class NoisyFlowMLP(nn.Module):
    """
    Noisy Flow MLP that adds learnable exploration noise to a base flow policy.
    Used in PPO Flow for fine-tuning with adaptive exploration.
    """

    def __init__(
        self,
        policy: FlowMLP,
        denoising_steps: int,
        learn_explore_noise_from: int,
        inital_noise_scheduler_type: str,
        min_logprob_denoising_std: float,
        max_logprob_denoising_std: float,
        learn_explore_time_embedding: bool = True,
        time_dim_explore: int = 16,
        use_time_independent_noise: bool = False,
        device: torch.device = torch.device("cpu"),
        noise_hidden_dims: Optional[List[int]] = None,
        activation_type: str = "Mish",
    ):
        super().__init__()

        self.policy = policy
        self.denoising_steps = int(denoising_steps)
        self.learn_explore_noise_from = int(learn_explore_noise_from)
        self.inital_noise_scheduler_type = inital_noise_scheduler_type
        self.min_logprob_denoising_std = float(min_logprob_denoising_std)
        self.max_logprob_denoising_std = float(max_logprob_denoising_std)
        self.learn_explore_time_embedding = bool(learn_explore_time_embedding)
        self.time_dim_explore = int(time_dim_explore)
        self.use_time_independent_noise = bool(use_time_independent_noise)
        self.device = device

        # Get dimensions from policy
        self.horizon_steps = policy.horizon_steps
        self.action_dim = policy.action_dim
        self.act_dim_total = policy.act_dim_total
        self.cond_dim = policy.cond_dim

        # Initialize noise scheduling parameters
        self._init_noise_scheduler()

        # Create learnable noise network
        if noise_hidden_dims is None:
            noise_hidden_dims = [128, 128]
        self._create_noise_network(noise_hidden_dims, activation_type)

    def _init_noise_scheduler(self) -> None:
        if self.inital_noise_scheduler_type == "linear":
            noise_schedule = torch.linspace(
                self.max_logprob_denoising_std,
                self.min_logprob_denoising_std,
                self.denoising_steps,
                dtype=torch.float32,
            )
        elif self.inital_noise_scheduler_type == "cosine":
            steps = torch.arange(self.denoising_steps, dtype=torch.float32)
            noise_schedule = self.min_logprob_denoising_std + 0.5 * (
                self.max_logprob_denoising_std - self.min_logprob_denoising_std
            ) * (1.0 + torch.cos(np.pi * steps / max(1.0, float(self.denoising_steps - 1))))
        else:
            base = (self.max_logprob_denoising_std + self.min_logprob_denoising_std) / 2.0
            noise_schedule = torch.full((self.denoising_steps,), base, dtype=torch.float32)

        # Register buffers and cached Python values
        self.register_buffer("base_noise_schedule", noise_schedule)
        # Keep a Python list for scheduling logic without Tensor ops
        self.base_noise_schedule_list = [float(x) for x in noise_schedule.tolist()]

        # Scalar bounds as buffers
        logvar_min_val = float(np.log(self.min_logprob_denoising_std**2 + 1e-12))
        logvar_max_val = float(np.log(self.max_logprob_denoising_std**2 + 1e-12))
        self.register_buffer("logvar_min", torch.tensor(logvar_min_val, dtype=torch.float32))
        self.register_buffer("logvar_max", torch.tensor(logvar_max_val, dtype=torch.float32))
        # Python float copies for clamp
        self.logvar_min_value = float(logvar_min_val)
        self.logvar_max_value = float(logvar_max_val)

    def _create_noise_network(self, noise_hidden_dims: List[int], activation_type: str) -> None:
        if self.learn_explore_time_embedding:
            self.time_embedding_explore = nn.Sequential(
                SinusoidalPosEmb(self.time_dim_explore),
                nn.Linear(self.time_dim_explore, self.time_dim_explore * 2),
                nn.Mish(),
                nn.Linear(self.time_dim_explore * 2, self.time_dim_explore),
            )
            time_input_dim = self.time_dim_explore
        else:
            time_input_dim = 1

        if self.use_time_independent_noise:
            input_dim = self.act_dim_total + self.cond_dim
        else:
            input_dim = time_input_dim + self.act_dim_total + self.cond_dim

        dims = [input_dim] + noise_hidden_dims + [self.act_dim_total]
        self.mlp_logvar = MLP(
            dims,
            activation_type=activation_type,
            out_activation_type="Identity",
            use_layernorm=False,
        )

    def forward(
        self,
        action: Tensor,
        time: Tensor,
        cond: Dict[str, Tensor],
        learn_exploration_noise: bool = True,
        step: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        B, _, _ = action.shape
        base_velocity = self.policy.predict_velocity(action, time, cond)

        if learn_exploration_noise and step >= self.learn_explore_noise_from:
            noise_std = self.predict_noise_std(action, time, cond)
        else:
            if isinstance(step, int) and step < len(self.base_noise_schedule_list):
                base_std = float(self.base_noise_schedule_list[step])
            else:
                base_std = float(self.min_logprob_denoising_std)
            noise_std = torch.full((B, self.act_dim_total), base_std, device=action.device, dtype=action.dtype)
        return base_velocity, noise_std

    def predict_noise_std(self, action: Tensor, time: Tensor, cond: Dict[str, Tensor]) -> Tensor:
        B, _, _ = action.shape
        action_flat = action.view(B, -1)
        state_flat = cond["state"].view(B, -1)

        if self.use_time_independent_noise:
            feat = torch.cat([action_flat, state_flat], dim=-1)
        else:
            if self.learn_explore_time_embedding:
                time_input = time.view(B, 1) if time.dim() == 1 else time
                time_emb = self.time_embedding_explore(time_input).view(B, self.time_dim_explore)
            else:
                time_emb = time.view(B, 1)
            feat = torch.cat([action_flat, time_emb, state_flat], dim=-1)

        logvar = self.mlp_logvar(feat)
        logvar = torch.clamp(logvar, min=self.logvar_min_value, max=self.logvar_max_value)
        std = torch.exp(0.5 * logvar)
        return std

    @torch.no_grad()
    def sample_action(
        self,
        cond: Dict[str, Tensor],
        inference_steps: int,
        clip_intermediate_actions: bool = True,
        act_range: List[float] = [-1.0, 1.0],
        z: Optional[Tensor] = None,
        save_chains: bool = False,
        eval_mode: bool = False,
    ):
        B = cond["state"].shape[0]
        device = cond["state"].device
        x_hat = z if z is not None else torch.randn(B, self.horizon_steps, self.action_dim, device=device)
        x_chain: Optional[Tensor] = None
        if save_chains:
            x_chain = torch.zeros((B, inference_steps + 1, self.horizon_steps, self.action_dim), device=device)
            x_chain[:, 0] = x_hat

        dt = 1.0 / float(inference_steps)
        steps = torch.linspace(0.0, 1.0, inference_steps, device=device).repeat(B, 1)
        for i in range(inference_steps):
            t = steps[:, i]
            vt, noise_std = self.forward(x_hat, t, cond, learn_exploration_noise=True, step=i)
            x_hat = x_hat + vt * dt
            if clip_intermediate_actions:
                x_hat = x_hat.clamp(act_range[0], act_range[1])
            if not eval_mode:
                noise_std_reshaped = noise_std.view(B, self.horizon_steps, self.action_dim)
                x_hat = x_hat + torch.randn_like(x_hat) * noise_std_reshaped
            if i == inference_steps - 1:
                x_hat = x_hat.clamp(act_range[0], act_range[1])
            if save_chains and x_chain is not None:
                x_chain[:, i + 1] = x_hat

        if save_chains and x_chain is not None:
            return x_hat, x_chain
        return x_hat
