import copy
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd.functional import jvp
from torch.utils.data import Dataset, DataLoader
import lightning as L
from torch import optim
from typing import Dict, Tuple, Any, Optional, List

from config import Config

# 设置Float32矩阵乘法精度以更好地利用Tensor Core
torch.set_float32_matmul_precision('high')


# ========= Small modules =========
class FeatureEmbedding(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
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
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
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
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        self.q1 = make_net()
        self.q2 = make_net()

    def _prep(self, obs: Tensor, actions: Tensor) -> Tensor:
        if actions.dim() == 3:
            act = actions.reshape(actions.shape[0], -1)
        elif actions.dim() == 2:
            act = actions
        else:
            raise ValueError(f"bad actions dim={actions.dim()}")

        obs_encoded = self.obs_encoder(obs)

        return torch.cat([obs_encoded, act], dim=-1)

    def forward(self, obs: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        x = self._prep(obs, actions)
        return self.q1(x), self.q2(x)


# ========= Value Function (V over obs) =========
class ValueFunction(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int):
        super().__init__()
        self.obs_encoder = FeatureEmbedding(obs_dim, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: Tensor) -> Tensor:
        obs_encoded = self.obs_encoder(obs)
        return self.net(obs_encoded)


# ========= Time-conditioned flow model (predicts velocity [B,H,A]) =========
class MeanTimeCondFlow(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, time_dim: int,
                 pred_horizon: int, obs_horizon: int):
        super().__init__()
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon

        self.t_embed = TimeEmbedding(time_dim)
        self.r_embed = TimeEmbedding(time_dim)
        self.obs_encoder = FeatureEmbedding(obs_dim, hidden_dim)
        self.noise_embed = FeatureEmbedding(action_dim, hidden_dim)

        joint_in = hidden_dim + hidden_dim + time_dim + time_dim
        self.net = nn.Sequential(
            nn.Linear(joint_in, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def _norm_z(self, z: Tensor) -> Tensor:
        if z.dim() == 3: return z
        if z.dim() == 2 and z.shape[1] == self.pred_horizon * self.action_dim:
            return z.view(z.shape[0], self.pred_horizon, self.action_dim)
        if z.dim() == 1:
            return z.view(1, self.pred_horizon,
                          self.action_dim) if z.numel() == self.pred_horizon * self.action_dim else z.view(1, 1,
                                                                                                           self.action_dim)
        raise ValueError(f"bad z shape: {z.shape}")

    @staticmethod
    def _norm_time(t: Tensor, B: int) -> Tensor:
        if t.dim() == 0: return t.new_full((B,), float(t))
        if t.dim() == 1:
            if t.numel() == B: return t
            if t.numel() == 1: return t.repeat(B)
            t = t.view(-1)
            return t[:B] if t.numel() >= B else torch.cat([t, t.new_zeros(B - t.numel())], dim=0)
        return t.view(-1)[:B]

    def forward(self, obs: Tensor, z: Tensor, r: Tensor, t: Tensor) -> Tensor:
        z = self._norm_z(z)
        B, H, A = z.shape
        t = self._norm_time(t, B)
        r = self._norm_time(r, B)

        te = self.t_embed(t)  # [B, Td]
        re = self.r_embed(r)  # [B, Td]

        obs_encoded = self.obs_encoder(obs)  # [B, hidden_dim]

        ne = self.noise_embed(z.reshape(B * H, A)).view(B, H, -1)  # [B,H,Hd]

        te = te.unsqueeze(1).repeat(1, H, 1)
        re = re.unsqueeze(1).repeat(1, H, 1)
        obs_encoded = obs_encoded.unsqueeze(1).repeat(1, H, 1)  # [B, H, hidden_dim]

        x = torch.cat([obs_encoded, ne, re, te], dim=-1)  # [B,H,*]
        return self.net(x)  # [B,H,A]


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

    def predict_action_chunk(self, obs: Tensor, n_steps: int = 1) -> Tensor:
        self.model.eval()
        device = next(self.parameters()).device
        obs = obs.to(device)

        with torch.no_grad():
            action_chunk = self.sample_mean_flow(obs, n_steps=n_steps)
            return torch.clamp(action_chunk, -self.action_scale, self.action_scale)

    def sample_mean_flow(self, obs: Tensor, n_steps: int = 1) -> Tensor:
        """使用单个观测进行均值流采样"""
        device = next(self.parameters()).device
        obs = obs.to(device)

        # 使用均匀分布初始化动作序列，范围[-action_scale, action_scale]
        x = (torch.rand(obs.size(0), self.pred_horizon, self.action_dim, device=device) - 0.5) * 2 * self.action_scale
        n_steps = max(1, int(n_steps))
        dt = 1.0 / n_steps

        for i in range(n_steps, 0, -1):
            r = torch.full((x.shape[0],), (i - 1) * dt, device=device)
            t = torch.full((x.shape[0],), i * dt, device=device)
            v = self.model(obs, x, r, t)
            x = x - v * dt

        return x

    def per_sample_flow_bc_loss(self, obs: Tensor, action_chunk: Tensor) -> Tensor:
        device = next(self.parameters()).device

        z0 = torch.randn_like(action_chunk)
        t, r = self.sample_t_r(action_chunk.shape[0], device=device)
        z = (1 - t.view(-1, 1, 1)) * action_chunk + t.view(-1, 1, 1) * z0
        v = z0 - action_chunk

        obs = obs.requires_grad_(True)
        z = z.requires_grad_(True)
        r = r.requires_grad_(True)
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
        losses = F.mse_loss(u_pred, u_tgt, reduction='none').mean(dim=[1, 2])  # [B]
        return losses


# ========= Whole RL model (Actor + Double Q + V + targets) =========
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

        self.vf = ValueFunction(obs_dim, cfg.hidden_dim)

    def discounted_returns(self, rewards: Tensor, gamma: float) -> Tensor:
        gamma = max(0.0, min(1.0, gamma))
        B, H, rew_dim = rewards.shape
        rewards_squeezed = rewards.squeeze(-1) if rew_dim == 1 else rewards.view(B, H)
        factors = (gamma ** torch.arange(H, device=rewards.device, dtype=rewards.dtype)).unsqueeze(0)
        return torch.sum(rewards_squeezed * factors, dim=1)  # [B]

    def best_of_n_sampling(self, obs: Tensor, N: int) -> Tensor:
        with torch.no_grad():
            B = obs.shape[0]

            expanded_obs = obs.unsqueeze(1).repeat(1, N, 1).view(B * N, -1)

            candidates = self.actor.predict_action_chunk(expanded_obs, self.cfg.inference_steps)

            q1_values, q2_values = self.target_critic(expanded_obs, candidates)
            q_values = torch.min(q1_values, q2_values).view(B, N)

            best_indices = torch.argmax(q_values, dim=1)

            candidates = candidates.view(B, N, self.cfg.pred_horizon, self.action_dim)
            batch_indices = torch.arange(B)
            best_action_chunks = candidates[batch_indices, best_indices]
            return best_action_chunks

    def loss_qf(self, obs: Tensor, actions: Tensor, next_obs: Tensor,
                rewards: Tensor, terminated: Tensor) -> Tuple[Tensor, Dict]:
        q1_pred, q2_pred = self.critic(obs, actions)
        q1_pred = q1_pred.squeeze(-1)
        q2_pred = q2_pred.squeeze(-1)
        with torch.no_grad():
            target_vf_pred = self.vf(next_obs).squeeze(-1)

            h_step_returns = self.discounted_returns(rewards, self.cfg.gamma)  # [B]

            done = terminated.view(-1).float()
            q_target = h_step_returns + (1. - done) * (self.cfg.gamma ** self.cfg.pred_horizon) * target_vf_pred

        qf1_loss = F.mse_loss(q1_pred, q_target)
        qf2_loss = F.mse_loss(q2_pred, q_target)
        qf_loss = qf1_loss + qf2_loss

        info = {
            'qf1_loss': qf1_loss.item(),
            'qf2_loss': qf2_loss.item(),
            'qf_loss': qf_loss.item(),
            'q_mean': torch.mean(torch.min(q1_pred, q2_pred)).item(),
        }
        return qf_loss, info

    def loss_vf(self, obs: Tensor, actions: Tensor) -> Tuple[Tensor, Dict]:
        with torch.no_grad():
            q1_pred, q2_pred = self.target_critic(obs, actions)
            q_pred = torch.min(q1_pred, q2_pred).squeeze(-1).detach()

        vf_pred = self.vf(obs).squeeze(-1)
        vf_err = vf_pred - q_pred

        # 正确的分位数回归损失计算
        quantile = self.cfg.quantile
        vf_loss = torch.mean(torch.where(vf_err > 0,
                                         quantile * vf_err,
                                         (quantile - 1) * vf_err))

        info = {
            'vf_loss': vf_loss.item(),
            'v_mean': vf_pred.mean().item(),
        }
        return vf_loss, info

    def loss_policy(self, obs: Tensor, actions: Tensor) -> Tuple[Tensor, Dict]:
        with torch.no_grad():
            q1, q2 = self.critic(obs, actions)
            q_pred = torch.min(q1, q2).squeeze(-1)
            v_pred = self.vf(obs).squeeze(-1)
            adv = q_pred - v_pred

        # 1. 计算优势加权的权重
        # beta 是一个超参数，用于控制权重的尺度，可以设置在 config 中，例如 self.cfg.beta = 10.0
        weights = torch.exp(adv / self.cfg.beta).detach()
        # 2. 对权重进行裁剪，防止梯度爆炸
        weights = torch.clamp(weights, max=100.0)

        # 3. 计算 BC 损失
        bc_losses = self.actor.per_sample_flow_bc_loss(obs, actions)

        # 4. 计算加权后的策略损失
        policy_loss = (weights * bc_losses).mean()

        info = {
            'policy_loss': policy_loss.item(),
            'adv_mean': adv.mean().item(),
            'bc_loss': bc_losses.mean().item(),
            'adv_weights_mean': weights.mean().item(),  # 监控权重大小
        }
        return policy_loss, info

    def update_target(self, tau: float):
        for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
            tp.data.copy_(tp.data * (1 - tau) + p.data * tau)


# ========= LightningModule (Manual optim for different networks) =========
class LitMeanFQL(L.LightningModule):
    def __init__(self, obs_dim: int, action_dim: int, cfg: Config):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.net = MeanFQL(obs_dim, action_dim, cfg)
        self.automatic_optimization = True


    def forward(self, obs: Tensor) -> Tensor:
        return self.net.best_of_n_sampling(obs, self.cfg.N)

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int):
        device = self.device

        obs = batch["observations"].float().to(device)
        obs = obs.squeeze(1) if obs.dim() > 2 else obs
        actions = batch["action_chunks"].float().to(device)
        next_obs = batch["next_observations"].float().to(device)
        next_obs = next_obs.squeeze(1) if next_obs.dim() > 2 else next_obs
        rewards = batch["rewards"].float().to(device)
        rewards = rewards.unsqueeze(-1) if rewards.dim() == 2 else rewards
        terminated = batch["terminations"].float().to(device)

        qf_loss, qf_info = self.net.loss_qf(obs, actions, next_obs, rewards, terminated)
        vf_loss, vf_info = self.net.loss_vf(obs, actions)
        policy_loss, policy_info = self.net.loss_policy(obs, actions)

        total_loss = qf_loss*0.1 + vf_loss*10 + policy_loss
        self.log("train/loss", total_loss, on_step=True, prog_bar=True)
        info = {**qf_info, **vf_info, **policy_info}
        for k, v in info.items():
            self.log(f"train/{k}", v, on_step=True, prog_bar=False)

        return total_loss




    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int):
        device = self.device

        obs = batch["observations"].float().to(device)
        obs = obs.squeeze(1) if obs.dim() > 2 else obs
        actions = batch["action_chunks"].float().to(device)
        next_obs = batch["next_observations"].float().to(device)
        next_obs = next_obs.squeeze(1) if next_obs.dim() > 2 else next_obs
        rewards = batch["rewards"].float().to(device)
        rewards = rewards.unsqueeze(-1) if rewards.dim() == 2 else rewards
        terminated = batch["terminations"].float().to(device)

        qf_loss, qf_info = self.net.loss_qf(obs, actions, next_obs, rewards, terminated)
        vf_loss, vf_info = self.net.loss_vf(obs, actions)
        policy_loss, policy_info = self.net.loss_policy(obs, actions)

        val_loss = qf_loss + vf_loss + policy_loss
        self.log("val/loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.log("val/reward", rewards.mean(), on_step=False, on_epoch=True, prog_bar=True)

        info = {**qf_info, **vf_info, **policy_info}
        for k, v in info.items():
            self.log(f"train/{k}", v, on_step=True, prog_bar=False)

        return val_loss

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int):
        obs = batch["observations"].float().to(self.device)
        obs = obs.squeeze(1) if obs.dim() > 2 else obs
        predicted_actions = self(obs)
        actual_actions = batch["action_chunks"].float().to(self.device)
        action_mse = F.mse_loss(predicted_actions, actual_actions)
        self.log("test/action_mse", action_mse, on_step=False, on_epoch=True)
        return action_mse

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if (batch_idx % self.cfg.target_update_freq) == 0:
            self.net.update_target(self.cfg.tau)

    def configure_optimizers(self):
        # 使用统一的优化器来优化整个网络
        optimizer = optim.Adam(self.net.parameters(), lr=self.cfg.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]

