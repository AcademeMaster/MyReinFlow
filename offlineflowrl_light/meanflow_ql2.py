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
        # MLP，使用Tanh输出，确保动作在[-1,1]范围内
        self.net = nn.Sequential(
            nn.Linear(joint_in, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 添加Norm
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
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
        x.clamp(-self.action_scale, self.action_scale)
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
        self.target_critic = copy.deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad = False

    def discounted_returns(self, rewards: Tensor, gamma: float) -> Tensor:
        gamma = max(0.0, min(1.0, gamma))
        B, H, rew_dim = rewards.shape
        
        # 确保rewards的形状为[B, H]
        rewards_squeezed = rewards.squeeze(-1) if rew_dim == 1 else rewards.view(B, H)
        # 创建折扣因子
        factors = (gamma ** torch.arange(H, device=rewards.device, dtype=rewards.dtype)).unsqueeze(0)
        # 计算加权回报
        weighted_returns = torch.sum(rewards_squeezed * factors, dim=1)  # [B]
        return weighted_returns  # [B]

    def best_of_n_sampling(self, obs: Tensor, N: int) -> Tensor:
        with torch.no_grad():
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


    def loss_critic(self, obs: Tensor, actions: Tensor, next_obs: Tensor,
                    rewards: Tensor, dones: Tensor) -> Tuple[Tensor, Dict]:
        B = obs.shape[0]

        # 使用no_grad计算得到一个固定
        with torch.no_grad():
            # 使用目标网络计算下一状态的Q值
            next_actions = self.actor.sample_mean_flow(next_obs)
            next_q1, next_q2 = self.target_critic(next_obs, next_actions)
            next_q = torch.min(next_q1, next_q2).view(B)

            # 计算H步折扣回报（从t到t+H-1）
            h_step_returns = self.discounted_returns(rewards, self.cfg.gamma)  # [B]
            # 计算bootstrap项
            done = dones.view(-1).float()  # 终止标志 [B]

            future_value = (1.0 - done) * (self.cfg.gamma ** self.cfg.pred_horizon) * next_q

            # 正确的TD目标
            target = h_step_returns + future_value  # [B]

        # 计算当前Q值
        q1, q2 = self.critic(obs, actions)  # [B,1]
        q1, q2 = q1.view(B), q2.view(B)

        # TD损失
        td_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        # CQL正则项优化：动态alpha + 能量加权
        q_dev = torch.abs(q1 - target).mean().detach()  # Q偏差
        adaptive_alpha = self.cfg.cql_alpha * (1 + q_dev / 10.0)  # 示例自适应

        num_samples = self.cfg.cql_num_samples // 2  # 减少以加速
        rep_obs = obs.repeat_interleave(num_samples, dim=0)
        uniform_samples = torch.rand_like(actions.repeat_interleave(num_samples, dim=0),
                                          device=obs.device) * 2.0 * self.actor.action_scale - self.actor.action_scale

        # 计算能量 (使用min(Q1,Q2) for robustness)
        q1_uniform, q2_uniform = self.critic(rep_obs, uniform_samples)
        q_uniform = torch.min(q1_uniform, q2_uniform).view(B, num_samples)
        energy = -q_uniform  # 负Q作为能量
        weights = F.softmax(-energy / self.cfg.cql_temp, dim=1).detach()  # weights基于q_uniform

        # 重新采样动作（保留以增强多样性）
        uniform_samples_reshaped = uniform_samples.view(B, num_samples, -1)
        sampled_indices = torch.multinomial(weights, num_samples, replacement=True)
        batch_indices = torch.arange(B).unsqueeze(1).expand(-1, num_samples)
        sampled_actions = uniform_samples_reshaped[batch_indices, sampled_indices].view(B * num_samples, -1)

        # 计算采样动作的Q值
        q1s, q2s = self.critic(rep_obs, sampled_actions)
        q1s = q1s.view(B, num_samples)
        q2s = q2s.view(B, num_samples)

        temp = self.cfg.cql_temp
        # 加权CQL损失：直接在logsumexp中整合weights
        log_weights = torch.log(weights + 1e-8)  # 避免log(0)
        cql1 = (torch.logsumexp(q1s / temp + log_weights, dim=1) - torch.log(
            torch.tensor(num_samples, dtype=torch.float, device=q1s.device))).mean() * temp - q1.mean()
        cql2 = (torch.logsumexp(q2s / temp + log_weights, dim=1) - torch.log(
            torch.tensor(num_samples, dtype=torch.float, device=q2s.device))).mean() * temp - q2.mean()

        cql_loss = (cql1 + cql2) * adaptive_alpha
        # 总损失
        total_loss = td_loss + cql_loss

        info = dict(
            td_loss=td_loss.item(),
            cql_loss=cql_loss.item(),
        )
        return total_loss, info

    def loss_policy(self, obs: Tensor, actions: Tensor) -> Tuple[Tensor, Dict]:
        # 使用当前critic而不是target critic来计算优势
        predict_actions = self.actor.sample_mean_flow(obs, self.cfg.inference_steps)
        q1, q2 = self.critic(obs, predict_actions)
        q_value = torch.min(q1, q2).view(-1)

        q_loss = -q_value.mean()  # 使用均值确保是标量
        bc_losses = self.actor.per_sample_flow_bc_loss(obs, actions)

        policy_loss = (bc_losses*0 + q_loss).mean()
        info = {
            'policy_loss': policy_loss.item(),
            'bc_loss': bc_losses.mean().item(),
            'q_loss': q_loss.item(),
        }
        return policy_loss, info

    def update_target(self, tau: float):
        for target_param, source_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                source_param.data * tau + target_param.data * (1.0 - tau)
            )