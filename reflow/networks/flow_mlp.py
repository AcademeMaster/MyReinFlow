"""
Minimal FlowMLP for ReFlow velocity prediction used by FQL baseline.
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple
from reflow.networks.mlp import MLP, ResidualMLP
from reflow.networks.embeddings import SinusoidalPosEmb


class FlowMLP(nn.Module):
    def __init__(
        self,
        horizon_steps: int,
        action_dim: int,
        cond_dim: int,
        time_dim: int = 16,
        mlp_dims: List[int] | None = None,
        cond_mlp_dims: List[int] | None = None,
        activation_type: str = "Mish",
        out_activation_type: str = "Identity",
        use_layernorm: bool = False,
        residual_style: bool = False,
    ) -> None:
        super().__init__()
        mlp_dims = mlp_dims or [256, 256]
        self.time_dim = time_dim
        self.horizon_steps = int(horizon_steps)
        self.action_dim = int(action_dim)
        self.act_dim_total = self.horizon_steps * self.action_dim
        self.cond_dim = int(cond_dim)

        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )

        model = ResidualMLP if residual_style else MLP

        if cond_mlp_dims:
            self.cond_mlp = MLP(
                [self.cond_dim] + cond_mlp_dims,
                activation_type=activation_type,
                out_activation_type="Identity",
            )
            cond_enc_dim = cond_mlp_dims[-1]
        else:
            cond_enc_dim = self.cond_dim

        input_dim = time_dim + self.act_dim_total + cond_enc_dim

        self.mlp_mean = model(
            [input_dim] + mlp_dims + [self.act_dim_total],
            activation_type=activation_type,
            out_activation_type=out_activation_type,
            use_layernorm=use_layernorm,
        )

    def _forward_core(self, action: Tensor, time: Tensor | int | float, cond: dict) -> Tuple[Tensor, Tensor, Tensor]:
        B, Ta, Da = action.shape
        action_flat = action.view(B, -1)
        state_flat = cond["state"].view(B, -1)
        state_enc = self.cond_mlp(state_flat) if hasattr(self, "cond_mlp") else state_flat

        if isinstance(time, (int, float)):
            t_in = torch.ones((B, 1), device=action.device, dtype=action.dtype) * float(time)
        else:
            t_in = time.view(B, 1)
        time_emb = self.time_embedding(t_in).view(B, self.time_dim)

        feat = torch.cat([action_flat, time_emb, state_enc], dim=-1)
        vel = self.mlp_mean(feat).view(B, Ta, Da)
        return vel, time_emb, state_enc

    def forward(self, action: Tensor, time: Tensor | int | float, cond: dict, output_embedding: bool = False):
        vel, time_emb, state_enc = self._forward_core(action, time, cond)
        if output_embedding:
            return vel, time_emb, state_enc
        return vel

    def predict_velocity(self, action: Tensor, time: Tensor | int | float, cond: dict) -> Tensor:
        vel, _, _ = self._forward_core(action, time, cond)
        return vel

    @torch.no_grad()
    def sample_action(
        self,
        cond: dict,
        inference_steps: int,
        clip_intermediate_actions: bool,
        act_range: List[float],
        z: Tensor | None = None,
        save_chains: bool = False,
    ):
        B = cond["state"].shape[0]
        device = cond["state"].device
        x_hat: Tensor = z if z is not None else torch.randn(B, self.horizon_steps, self.action_dim, device=device)
        x_chain: Tensor | None = None
        if save_chains:
            x_chain = torch.zeros((B, inference_steps + 1, self.horizon_steps, self.action_dim), device=device)
            x_chain[:, 0] = x_hat
        dt = (1.0 / inference_steps) * torch.ones_like(x_hat, device=device)
        steps = torch.linspace(0, 1, inference_steps, device=device).repeat(B, 1)
        for i in range(inference_steps):
            t = steps[:, i]
            vt = self.predict_velocity(x_hat, t, cond)
            x_hat = x_hat + vt * dt
            if clip_intermediate_actions or i == inference_steps - 1:
                x_hat = x_hat.clamp(act_range[0], act_range[1])
            if save_chains and x_chain is not None:
                x_chain[:, i + 1] = x_hat
        if save_chains and x_chain is not None:
            return x_hat, x_chain
        return x_hat
    

