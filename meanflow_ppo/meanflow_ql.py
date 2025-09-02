import copy
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd.functional import jvp

from config import Config

# 设置Float32矩阵乘法精度以更好地利用Tensor Core
torch.set_float32_matmul_precision('high')
torch.autograd.set_detect_anomaly(True)


# ========= Small modules =========
class FeatureEmbedding(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 添加LayerNorm，提升稳定性
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),  # 添加激活，提升非线性
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TimeEmbedding(nn.Module):
    """sin/cos time embedding + MLP"""

    def __init__(self, time_dim: int, max_period: int = 10_000):
        super().__init__()
        assert time_dim % 2 == 0, "time_dim must be even"
        half = time_dim // 2
        exponents = torch.arange(half, dtype=torch.float32) / float(half)
        freqs = 1.0 / (max_period ** exponents)
        self.register_buffer("freqs", freqs, persistent=False)
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.LayerNorm(time_dim),  # 添加Norm
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),  # 调整为更对称结构，避免瓶颈
        )
        self.time_dim = time_dim

    def forward(self, t: Tensor) -> Tensor:
        t = t.view(-1).float()  # [B]
        args = t.unsqueeze(-1) * self.freqs.unsqueeze(0)  # [B, half]
        enc = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, time_dim]
        return self.mlp(enc)


# ========= Critic (Double Q over obs + action-chunk) =========
class DoubleCriticObsAct(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, action_horizon: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.total_action_dim = action_dim * action_horizon

        self.obs_encoder = FeatureEmbedding(obs_dim, hidden_dim)

        def make_net():
            return nn.Sequential(
                nn.Linear(hidden_dim + self.total_action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # 添加Norm
                nn.SiLU(),  # 统一为SiLU
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )

        self.q1 = make_net()
        self.q2 = make_net()

    def _prep(self, obs: Tensor, actions: Tensor) -> Tensor:
        act = actions.view(actions.shape[0], -1)  # 统一展平，支持dim=2或3
        obs_encoded = self.obs_encoder(obs)
        return torch.cat([obs_encoded, act], dim=-1)

    def forward(self, obs: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        x = self._prep(obs, actions)
        return self.q1(x), self.q2(x)


# ========= Time-conditioned meanflow_ppo model (predicts velocity [B,H,A]) =========
class MeanTimeCondFlow(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, time_dim: int,
                 pred_horizon: int, obs_horizon: int):
        super().__init__()
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon  # TODO: 若obs_horizon>1，可添加序列encoder

        self.t_embed = TimeEmbedding(time_dim)
        self.r_embed = TimeEmbedding(time_dim)
        self.obs_encoder = FeatureEmbedding(obs_dim, hidden_dim)
        self.noise_embed = FeatureEmbedding(action_dim, hidden_dim)

        joint_in = hidden_dim + hidden_dim + time_dim + time_dim
        self.net = nn.Sequential(
            nn.Linear(joint_in, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 添加Norm
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: Tensor, z: Tensor, r: Tensor, t: Tensor) -> Tensor:
        B, H, A = z.shape
        te = self.t_embed(t)[:, None, :].expand(B, H, -1)  # 优化broadcast
        re = self.r_embed(r)[:, None, :].expand(B, H, -1)
        obs_encoded = self.obs_encoder(obs)[:, None, :].expand(B, H, -1)

        ne = self.noise_embed(z.view(B * H, A)).view(B, H, -1)

        x = torch.cat([obs_encoded, ne, re, te], dim=-1)
        return self.net(x)


# ========= Actor (MeanFlow) =========
class MeanFlowActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.pred_horizon = cfg.pred_horizon
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.model = MeanTimeCondFlow(obs_dim, action_dim, cfg.hidden_dim, cfg.time_dim,
                                      cfg.pred_horizon, cfg.obs_horizon)

        # 定义动作边界
        self.action_scale = 2.0

    @staticmethod
    def sample_t_r(n: int, device) -> Tuple[Tensor, Tensor]:
        t = torch.rand(n, device=device)
        r = torch.rand(n, device=device) * t
        return t, r

    @torch.no_grad()
    def predict_action_chunk(self, obs: Tensor, n_steps: int = 1) -> Tensor:
        self.model.eval()
        device = next(self.parameters()).device
        obs = obs.to(device)
        action_chunk = self.sample_mean_flow(obs, n_steps=n_steps)
        return torch.clamp(action_chunk, -self.action_scale, self.action_scale)

    def sample_mean_flow(self, obs: Tensor, n_steps: int = 1) -> Tensor:
        """使用单个观测进行均值流采样"""
        device = next(self.parameters()).device
        obs = obs.to(device)

        x = torch.randn(obs.size(0), self.pred_horizon, self.action_dim, device=device)
        n_steps = max(1, int(n_steps))
        dt = 1.0 / n_steps

        for i in range(n_steps, 0, -1):
            r = torch.full((x.shape[0],), (i - 1) * dt, device=device)
            t = torch.full((x.shape[0],), i * dt, device=device)
            v = self.model(obs, x, r, t)
            x = x - v * dt
        x = x.clamp(-self.action_scale, self.action_scale)
        return x

    def per_sample_flow_bc_loss(self, obs: Tensor, action_chunk: Tensor) -> Tensor:
        device = next(self.parameters()).device

        z0 = torch.randn_like(action_chunk)
        t, r = self.sample_t_r(action_chunk.shape[0], device=device)
        z = (1 - t.view(-1, 1, 1)) * action_chunk + t.view(-1, 1, 1) * z0
        v = z0 - action_chunk

        z = z.requires_grad_(True)
        t = t.requires_grad_(True)

        v_obs = torch.zeros_like(obs)
        v_z = v
        v_r = torch.zeros_like(r)
        v_t = torch.ones_like(t)

        u_pred, dudt = jvp(lambda *ins: self.model(*ins),
                           (obs, z, r, t),
                           (v_obs, v_z, v_r, v_t),
                           create_graph=True)

        delta = torch.clamp(t - r, min=1e-6).view(-1, 1, 1)
        u_tgt = (v - delta * dudt).detach()
        losses = F.huber_loss(u_pred, u_tgt, reduction='mean')  # [B]
        return losses


# ========= Whole RL model (Actor + Double Q ) =========
class MeanFQL(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.actor = MeanFlowActor(obs_dim, action_dim, cfg)
        self.critic = DoubleCriticObsAct(obs_dim, action_dim, cfg.hidden_dim,
                                         cfg.pred_horizon)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        for p in self.target_actor.parameters():
            p.requires_grad = False
        for p in self.target_critic.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def best_of_n_sampling(self, obs: Tensor, N: int) -> Tensor:
        B = obs.shape[0]

        expanded_obs = obs.unsqueeze(1).repeat(1, N, 1).view(B * N, -1)

        candidates = self.actor.sample_mean_flow(expanded_obs, self.cfg.inference_steps)

        q1_values, q2_values = self.target_critic(expanded_obs, candidates)
        q_values = torch.min(q1_values, q2_values).view(B, N)
        best_indices = torch.argmax(q_values, dim=1)
        candidates = candidates.view(B, N, self.cfg.pred_horizon, self.action_dim)
        batch_indices = torch.arange(B)
        best_action_chunks = candidates[batch_indices, best_indices]
        return best_action_chunks

    # ---------- 修改 1: discounted_returns -> 返回 (returns, alive_mask) ----------
    def discounted_returns(self, rewards: Tensor, dones: Tensor, gamma: float) -> Tuple[Tensor, Tensor]:
        """
        rewards: [B, H, 1] or [B, H]
        dones:   [B, H, 1] or [B, H]  (done at step t+i indicates episode ended at that step)
        返回:
            returns: [B]  -> sum_{i=0..H-1} gamma^i * r_{t+i} * prod_{j< i} (1 - d_{t+j})
            alive_mask: [B] -> 1.0 if NO done happened in first H steps, else 0.0
        """
        gamma = float(max(0.0, min(1.0, gamma)))
        B, H, _ = rewards.shape

        rewards_squeezed = rewards.squeeze(-1)  # [B, H]
        dones_squeezed = dones.squeeze(-1)  # [B, H]

        # 折扣因子 gamma^0 ... gamma^{H-1}
        device = rewards.device
        dtype = rewards.dtype
        factors = (gamma ** torch.arange(H, device=device, dtype=dtype))  # [H]

        discounted_rewards = rewards_squeezed * factors  # [B, H]

        # mask: product_{j< i} (1 - d_j). 右移后 mask[0]=1, mask[1]=1-d0, ...
        mask = torch.cumprod(1.0 - dones_squeezed.float(), dim=1)  # [B, H]
        mask = torch.cat([torch.ones(B, 1, device=device, dtype=dtype), mask[:, :-1]], dim=1)

        # returns
        weighted_returns = torch.sum(discounted_rewards * mask, dim=1)  # [B]

        # alive_mask: 如果前 H 步都没遇到 done，则为 1，否则 0
        alive_mask = mask[:, -1].float()  # product_{j=0..H-1} (1 - d_j)

        return weighted_returns, alive_mask

    # 在 loss_critic 中更新调用方式
    def loss_critic(self, obs: Tensor, actions: Tensor, next_obs: Tensor,
                    rewards: Tensor, dones: Tensor) -> Tuple[Tensor, Dict]:
        B = obs.shape[0]

        with torch.no_grad():
            # 修复：移除重复的采样，直接使用target actor生成next action
            next_actions = self.target_actor.sample_mean_flow(next_obs, n_steps=self.cfg.inference_steps)
            # 添加适当的噪声以提高稳定性
            noise = torch.clamp(torch.randn_like(next_actions) * 0.1, -0.2, 0.2)
            next_actions = torch.clamp(next_actions + noise, -self.actor.action_scale, self.actor.action_scale)

            next_q1, next_q2 = self.target_critic(next_obs, next_actions)
            next_q = torch.min(next_q1, next_q2).view(B)  # [B]

            # 返回 N-step 加权回报并得到 alive mask
            h_step_returns, alive_mask = self.discounted_returns(rewards, dones, self.cfg.gamma)
            # alive_mask==1 表示窗口内没有遇到 done（可以 bootstrapping）
            # future_value 只有在 alive_mask==1 时生效
            future_value = alive_mask * (self.cfg.gamma ** self.cfg.pred_horizon) * next_q
            target = h_step_returns + future_value  # [B]

        # 计算当前Q值
        q1, q2 = self.critic(obs, actions)  # [B,1]
        q1, q2 = q1.view(B), q2.view(B)

        # TD损失 - 使用Huber loss提高稳定性
        td_loss = F.huber_loss(q1, target, reduction='mean') + F.huber_loss(q2, target, reduction='mean')

        info = dict(
            td_loss=td_loss.item(),
            q1_mean=q1.mean().item(),
            q2_mean=q2.mean().item(),
            target_mean=target.mean().item(),
            alive_ratio=alive_mask.mean().item(),
        )
        return td_loss, info

    # 可选的动态权重衰减
    def get_current_bc_alpha(self, current_step: int) -> float:
        """根据训练步数动态调整BC损失权重"""
        if current_step < self.cfg.bc_loss_decay_steps:
            # 线性衰减从initial_weight到final_weight
            progress = current_step / self.cfg.bc_loss_decay_steps
            return self.cfg.bc_loss_initial_weight * (1 - progress) + self.cfg.bc_loss_final_weight * progress
        return self.cfg.bc_loss_final_weight

    def loss_policy(self, obs: Tensor, actions: Tensor, current_step: int = 0) -> Tuple[Tensor, Dict]:
        # 使用当前critic而不是target critic来计算优势
        predict_actions = self.actor.sample_mean_flow(obs, self.cfg.inference_steps)
        q1, q2 = self.critic(obs, predict_actions)
        q_value = torch.min(q1, q2).view(-1)

        # 修复：Q损失应该是负的平均值，鼓励高Q值
        q_loss = -q_value.mean()
        bc_losses = self.actor.per_sample_flow_bc_loss(obs, actions)

        # 动态调整BC权重
        # 修复权重平衡：BC损失应该主导早期训练，Q损失逐渐增强
        # 使用更合理的权重，避免Q损失过小被忽略
        policy_loss = self.cfg.bc_alpha *bc_losses +  q_loss

        info = {
            'policy_loss': policy_loss.item(),
            'bc_loss': bc_losses.item(),
            'q_loss': q_loss.item(),
            'q_value_mean': q_value.mean().item(),
            'bc_alpha': self.cfg.bc_alpha,
        }
        return policy_loss, info

    def update_target(self, tau: float):
        for target_param, source_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                source_param.data * tau + target_param.data * (1.0 - tau)
            )
        for target_param, source_param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                source_param.data * tau + target_param.data * (1.0 - tau)
            )