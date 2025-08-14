import torch 
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import copy
from reflow.flows.reflow import ReFlow
from reflow.models.critic import CriticObsAct

class OneStepActor(nn.Module):
    """Distill a multistep flow model to single step stochastic policy"""
    def __init__(self, 
                 obs_dim, 
                 cond_steps,
                 action_dim, 
                 horizon_steps, 
                 hidden_dim=512):
        """Initialize the OneStepActor with a neural network to map observations and noise to actions.

        Args:
            obs_dim (int): Dimension of the observation space.
            cond_steps (int): Number of observation steps in the horizon.
            action_dim (int): Dimension of the action space.
            horizon_steps (int): Number of action steps in the horizon.
            hidden_dim (int, optional): Hidden layer size. Defaults to 512.
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.cond_steps=cond_steps
        self.action_dim = action_dim
        self.horizon_steps = horizon_steps
        self.net = nn.Sequential(
            nn.Linear(cond_steps*obs_dim + horizon_steps * action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, horizon_steps * action_dim),
        )

    def forward(self, cond: Dict[str, Tensor], z: Tensor) -> Tensor:
        """Generate actions from observations and noise using the actor network.

        Args:
            cond (Dict[str, Tensor]): Dictionary containing the state tensor with key 'state', which is a tensor of shape (batch, cond_steps, obs_dim)
            z (Tensor): Noise tensor of shape (batch, horizon_steps, action_dim).

        Returns:
            Tensor: Actions of shape (batch, horizon_steps, action_dim).
        """
        s = cond['state']       # (batch, cond_step, obs_dim)
        B = s.shape[0]
        s_flat = s.view(B, -1)  # (batch, cond_step * obs_dim)
        z_flat = z.view(B, -1)  # (batch, horizon*action_dim)
        feature = torch.cat([s_flat, z_flat], dim=-1)
        output = self.net(feature)
        actions: Tensor = output.view(B, self.horizon_steps, self.action_dim)
        return actions

class FQLModel(nn.Module):
    def __init__(self,
                 bc_flow: ReFlow,
                 actor: OneStepActor,
                 critic: CriticObsAct,
                 inference_steps: int,
                 normalize_q_loss: bool,
                 target_q_agg: str,
                 act_low,
                 act_high,
                 device):
        """Initialize the FQL model with behavior cloning flow, actor, critic, and target critic.

        Args:
            bc_flow (ReFlow): Behavior cloning flow model.
            actor (OneStepActor): Actor network for action generation.
            critic (CriticObsAct): Critic network for Q-value estimation.
            inference_steps (int): Number of inference steps for the flow model.
            normalize_q_loss (bool): Whether to normalize the Q-value loss.
            target_q_agg (str): Aggregation for target Q ('min' or 'mean').
            device: Device to run the model on (e.g., 'cuda').
        """
        super().__init__()
        self.bc_flow = bc_flow
        self.actor = actor
        self.critic = critic
        self.target_critic = copy.deepcopy(self.critic)
        self.device = device
        self.inference_steps = int(inference_steps)
        self.normalize_q_loss = bool(normalize_q_loss)
        self.target_q_agg = target_q_agg if target_q_agg in ("min", "mean") else "min"
        # per-dimension action bounds on device, shape (Da,)
        self.act_low = torch.as_tensor(act_low, dtype=torch.float32, device=self.device).view(-1)
        self.act_high = torch.as_tensor(act_high, dtype=torch.float32, device=self.device).view(-1)

    def _clip_actions(self, a: Tensor) -> Tensor:
        """Per-dimension clamp for actions. a: [B, Ta, Da]."""
        low = self.act_low.view(1, 1, -1)
        high = self.act_high.view(1, 1, -1)
        return torch.max(torch.min(a, high), low)

    def forward(self,
                cond: Dict[str, Tensor],
                mode: str = 'onestep') -> Tensor:
        """Generate actions for a batch of observations.

        Args:
            cond: Dict with key 'state'.
            mode: 'onestep' uses distilled actor; 'base_model' uses flow sampler.
        """
        batch_size = cond['state'].shape[0]
        assert mode in ['onestep', 'base_model']
        if mode == 'onestep':
            z = torch.randn(batch_size, self.actor.horizon_steps, self.actor.action_dim, device=self.device)
            actions: Tensor = self.actor(cond, z)
        else:
            actions = self.bc_flow.sample(cond, self.inference_steps, record_intermediate=False, clip_intermediate_actions=False).trajectories
        return actions

    def loss_bc_flow(self, obs: Dict[str, Tensor], actions: Tensor) -> Tensor:
        """BC loss for flow model."""
        (xt, t), v = self.bc_flow.generate_target(actions)
        v_hat = self.bc_flow.network(xt, t, obs)
        return F.mse_loss(v_hat, v)

    def loss_critic(self, obs, actions, next_obs, rewards, terminated, gamma) -> Tuple[Tensor, Dict]:
        """TD loss for double critic."""
        with torch.no_grad():
            z = torch.randn_like(actions, device=self.device)
            next_actions = self.actor.forward(next_obs, z)
            next_actions = self._clip_actions(next_actions)
            next_q1, next_q2 = self.target_critic.forward(next_obs, next_actions)
            if self.target_q_agg == 'min':
                next_q = torch.minimum(next_q1, next_q2)
            else:
                next_q = torch.mean(torch.stack([next_q1, next_q2], dim=0), dim=0)
            target = rewards + gamma * (1 - terminated) * next_q
        q1, q2 = self.critic.forward(obs, actions)
        loss_critic = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        info = {
            'loss_critic': float(loss_critic.item()),
            'q1_mean': float(q1.mean().item()),
            'q2_mean': float(q2.mean().item()),
            'loss_critic_1': float(F.mse_loss(q1, target).item()),
            'loss_critic_2': float(F.mse_loss(q2, target).item()),
        }
        return loss_critic, info

    def loss_actor(self, obs: Dict[str, Tensor], action_batch: Tensor, alpha: float, q_weight: float = 1.0) -> Tuple[Tensor, Dict[str, float]]:
        """Actor loss = BC(flow) + alpha * distill + q_weight * (-Q)."""
        batch_size = obs['state'].shape[0]
        z = torch.randn(batch_size, self.actor.horizon_steps, self.actor.action_dim, device=self.device)
        # one-step actor actions
        a_w = self.actor.forward(obs, z)
        # flow teacher actions with the same noise z (no grad to teacher)
        with torch.no_grad():
            a_theta = self.bc_flow.sample(
                cond=obs,
                inference_steps=self.inference_steps,
                record_intermediate=False,
                clip_intermediate_actions=False,
                z=z,
            ).trajectories
        # Distill on final, bounded actions
        distill_loss = F.mse_loss(self._clip_actions(a_w), self._clip_actions(a_theta))

        # Q term with actions clamped to legal range
        actor_actions = self._clip_actions(a_w)
        q1, q2 = self.critic.forward(obs, actor_actions)
        q = torch.mean(torch.stack([q1, q2], dim=0), dim=0)
        q_loss = -q.mean()
        if self.normalize_q_loss:
            denom = torch.clamp(torch.abs(q).mean().detach(), min=1e-3)
            q_loss = q_loss / denom

        # Flow BC term
        loss_bc_flow = self.loss_bc_flow(obs, action_batch)
        # Scale Q loss with q_weight for stability control
        loss_actor = loss_bc_flow + alpha * distill_loss + q_weight * q_loss
        info = {
            'loss_actor': float(loss_actor.item()),
            'loss_bc_flow': float(loss_bc_flow.item()),
            'q_loss': float(q_loss.item()),
            'distill_loss': float(distill_loss.item()),
            'q': float(q.mean().item()),
            'q_weight': float(q_weight),
            'onestep_expert_bc_loss': float(F.mse_loss(a_w, action_batch).item()),
        }
        return loss_actor, info

    def update_target_critic(self, tau: float) -> None:
        """Soft-update target critic."""
        for tgt, src in zip(self.target_critic.parameters(), self.critic.parameters()):
            tgt.data.copy_(src.data * tau + tgt.data * (1.0 - tau))
                
