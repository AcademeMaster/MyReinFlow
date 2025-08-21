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
        self.total_obs_dim = obs_dim

        def make_net():
            return nn.Sequential(
                nn.Linear(self.total_obs_dim + self.total_action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        self.q1 = make_net()
        self.q2 = make_net()

    def _prep(self, obs: Tensor, actions: Tensor) -> Tensor:
        # obs: [B, obs_dim] - 最后一个时间步的观测
        # actions: [B,H,A] or [B,H*A]
        if actions.dim() == 3:
            act = actions.reshape(actions.shape[0], -1)
        elif actions.dim() == 2:
            act = actions
        else:
            raise ValueError(f"bad actions dim={actions.dim()}")

        return torch.cat([obs, act], dim=-1)

    def forward(self, obs: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        x = self._prep(obs, actions)
        return self.q1(x), self.q2(x)


# ========= Time-conditioned flow model (predicts velocity [B,H,A]) =========
class MeanTimeCondFlow(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, time_dim: int, pred_horizon: int):
        super().__init__()
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.pred_horizon = pred_horizon
        self.t_embed = TimeEmbedding(time_dim)
        self.r_embed = TimeEmbedding(time_dim)
        self.obs_embed = FeatureEmbedding(obs_dim, hidden_dim)
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
        if z.dim() == 1:  # Added for single sample
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
        # obs: [B, obs_dim] - 最后一个时间步的观测
        z = self._norm_z(z)
        B, H, A = z.shape
        t = self._norm_time(t, B)
        r = self._norm_time(r, B)

        te = self.t_embed(t)  # [B, Td]
        re = self.r_embed(r)  # [B, Td]
        oe = self.obs_embed(obs)  # [B, Hd]
        ne = self.noise_embed(z.reshape(B * H, A)).view(B, H, -1)  # [B,H,Hd]

        te = te.unsqueeze(1).repeat(1, H, 1)
        re = re.unsqueeze(1).repeat(1, H, 1)
        oe = oe.unsqueeze(1).repeat(1, H, 1)
        x = torch.cat([oe, ne, re, te], dim=-1)  # [B,H,*]
        return self.net(x)  # [B,H,A]


# ========= Actor (MeanFlow) =========
class MeanFlowActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.pred_horizon = cfg.pred_horizon
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.model = MeanTimeCondFlow(obs_dim, action_dim, cfg.hidden_dim, cfg.time_dim, cfg.pred_horizon)
        # 在Actor内部维护观测历史
        self.obs_history = deque(maxlen=cfg.obs_horizon)
        # 用零初始化观测历史
        for _ in range(cfg.obs_horizon):
            self.obs_history.append(torch.zeros(obs_dim))

    def reset_obs_history(self):
        """重置观测历史"""
        self.obs_history.clear()
        for _ in range(self.cfg.obs_horizon):
            self.obs_history.append(torch.zeros(self.obs_dim))

    @staticmethod
    def sample_t_r(n: int, device) -> Tuple[Tensor, Tensor]:
        t = torch.rand(n, device=device)
        r = torch.rand(n, device=device) * t
        return t, r

    def predict_action_chunk(self, obs_cond: Tensor, n_steps: int = 1) -> Tensor:
        self.model.eval()
        device = next(self.parameters()).device
        obs_cond = obs_cond.to(device)
        x = torch.randn(obs_cond.size(0), self.pred_horizon, self.action_dim, device=device)
        return self.sample_mean_flow(obs_cond, x, n_steps=n_steps)

    def select_action(self, obs: Tensor, n_steps: int = 1) -> Tensor:
        """
        为推理选择动作，内部维护观测历史
        输入: obs 形状为 [B, obs_dim] 或 [obs_dim]
        输出: 动作张量 [B, action_dim]
        """
        self.model.eval()
        device = next(self.parameters()).device
        obs = obs.to(device)

        # 确保输入是正确的形状
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # [1, obs_dim]

        # 更新观测历史 - 添加当前观测到历史中
        for i in range(obs.shape[0]):  # 对于每个样本
            self.obs_history.append(obs[i].cpu())  # 存储到CPU以避免设备问题

        # 构建观测序列
        obs_sequence = torch.stack(list(self.obs_history)).unsqueeze(0).to(device)  # [1, obs_horizon, obs_dim]
        if obs.shape[0] > 1:
            # 对于多个样本，复制观测序列
            obs_sequence = obs_sequence.repeat(obs.shape[0], 1, 1)  # [B, obs_horizon, obs_dim]

        # 取最后一个时间步的观测用于条件生成
        obs_cond = obs_sequence[:, -1, :]  # [B, obs_dim]
        x = torch.randn(obs.shape[0], self.pred_horizon, self.action_dim, device=device)
        # 返回第一个动作
        action_chunk = self.sample_mean_flow(obs_cond, x, n_steps=n_steps)
        return action_chunk[:, 0, :]

    def sample_mean_flow(self, obs_cond: Tensor, x: Tensor, n_steps: int = 1) -> Tensor:
        """使用改进的观测序列处理和时间步进的均值流采样"""
        device = next(self.parameters()).device
        obs_cond, x = obs_cond.to(device), x.to(device)

        n_steps = max(1, int(n_steps))
        dt = 1.0 / n_steps

        for i in range(n_steps, 0, -1):
            r = torch.full((x.shape[0],), (i - 1) * dt, device=device)
            t = torch.full((x.shape[0],), i * dt, device=device)
            v = self.model(obs_cond, x, r, t)
            x = x - v * dt

        return torch.clamp(x, -1, 1)

    def flow_bc_loss(self, obs_cond: Tensor, action_chunk: Tensor) -> Tensor:
        """MeanFlow 训练损失（JVP 版）"""
        device = next(self.parameters()).device
        noise = torch.randn_like(action_chunk)
        t, r = self.sample_t_r(action_chunk.shape[0], device=device)
        z = (1 - t.view(-1, 1, 1)) * action_chunk + t.view(-1, 1, 1) * noise
        v = noise - action_chunk

        obs_cond = obs_cond.requires_grad_(True)
        z = z.requires_grad_(True)
        r = r.requires_grad_(True)
        t = t.requires_grad_(True)

        v_obs = torch.zeros_like(obs_cond)
        v_z = v
        v_r = torch.zeros_like(r)
        v_t = torch.ones_like(t)

        u_pred, dudt = jvp(lambda *ins: self.model(*ins),
                           (obs_cond, z, r, t),
                           (v_obs, v_z, v_r, v_t),
                           create_graph=True)

        delta = torch.clamp(t - r, min=1e-6).view(-1, 1, 1)  # Added epsilon for stability
        u_tgt = (v - delta * dudt).detach()
        return F.mse_loss(u_pred, u_tgt)


# ========= Whole RL model (Actor + Double Q + target) =========
class ConservativeMeanFQL(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.actor = MeanFlowActor(obs_dim, action_dim, cfg)
        self.critic = DoubleCriticObsAct(obs_dim, action_dim, cfg.hidden_dim, cfg.pred_horizon)
        self.target_critic = copy.deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad = False

        # 初始化损失统计
        self._td_loss_stats = 1.0
        self._cql_loss_stats = 1.0

    def discounted_returns(self, rewards: Tensor, gamma: float) -> Tensor:
        """
        计算H步折扣回报（从t到t+H-1）
        rewards: [B, H] 包含从t到t+H-1的奖励
        返回: [B] 每个序列的折扣回报
        """
        # 确保gamma在有效范围内
        gamma = max(0.0, min(1.0, gamma))

        B, H = rewards.shape
        factors = (gamma ** torch.arange(H, device=rewards.device, dtype=rewards.dtype)).unsqueeze(0)
        return torch.sum(rewards * factors, dim=1)  # [B]

    def loss_critic(self, obs_cond: Tensor, actions: Tensor, next_obs_cond: Tensor,
                    rewards: Tensor, terminated: Tensor, gamma: float) -> Tuple[Tensor, Dict]:
        """
        Critic损失计算
        obs_cond: [B, obs_dim] - 最后一个时间步的观测
        actions: [B, H, A] - 动作序列
        next_obs_cond: [B, obs_dim] - 下一个状态最后一个时间步的观测
        rewards: [B, H] - 从t到t+H-1的奖励
        terminated: [B, 1] - 终止标志
        gamma: 折扣因子
        """
        B = obs_cond.shape[0]
        with torch.no_grad():
            # 使用目标网络计算下一状态的Q值
            next_actions = self.actor.predict_action_chunk(next_obs_cond, n_steps=self.cfg.inference_steps)
            next_q1, next_q2 = self.target_critic(next_obs_cond, next_actions)
            next_q = torch.min(next_q1, next_q2).view(B)

            # 计算H步折扣回报（从t到t+H-1）
            h_step_returns = self.discounted_returns(rewards, gamma)  # [B]

            # 计算bootstrap项
            term = terminated.view(-1).float()  # 终止标志
            bootstrap = (1.0 - term) * (gamma ** self.cfg.pred_horizon) * next_q

            # 正确的TD目标
            target = h_step_returns + bootstrap  # [B]

        # 计算当前Q值
        q1, q2 = self.critic(obs_cond, actions)  # [B,1]
        q1, q2 = q1.view(B), q2.view(B)

        # TD损失
        td_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        # CQL正则项
        num_samples = self.cfg.cql_num_samples
        rep_obs_cond = obs_cond.repeat_interleave(num_samples, dim=0)
        noise = torch.randn(B * num_samples, self.cfg.pred_horizon, self.action_dim, device=obs_cond.device)
        sampled_actions = self.actor.sample_mean_flow(rep_obs_cond, noise, n_steps=self.cfg.inference_steps)

        # 计算采样动作的Q值
        rep_obs = obs_cond.repeat_interleave(num_samples, dim=0)
        q1s, q2s = self.critic(rep_obs, sampled_actions)
        q1s = q1s.view(B, num_samples)
        q2s = q2s.view(B, num_samples)

        temp = self.cfg.cql_temp
        # CQL损失计算
        cql1 = torch.logsumexp(q1s / temp, dim=1).mean() * temp - q1.mean()
        cql2 = torch.logsumexp(q2s / temp, dim=1).mean() * temp - q2.mean()
        cql = (cql1 + cql2) * self.cfg.cql_alpha

        # 动态平衡TD loss和CQL loss
        if hasattr(self, '_td_loss_stats') and hasattr(self, '_cql_loss_stats'):
            # 使用指数移动平均来跟踪损失统计
            self._td_loss_stats = 0.9 * self._td_loss_stats + 0.1 * td_loss.item()
            self._cql_loss_stats = 0.9 * self._cql_loss_stats + 0.1 * cql.item()
        else:
            # 初始化损失统计
            self._td_loss_stats = td_loss.item()
            self._cql_loss_stats = cql.item()

        # 根据损失统计进行归一化
        if self._cql_loss_stats > 1e-8:  # 避免除零
            cql_scale = self._td_loss_stats / self._cql_loss_stats
            cql_normalized = cql * cql_scale
        else:
            cql_normalized = cql

        total = td_loss + cql_normalized
        info = dict(
            td_loss=td_loss.item(),
            cql_loss=cql_normalized.item(),
            total_critic_loss=total.item(),
            q1_mean=q1.mean().item(),
            q2_mean=q2.mean().item(),
            target_mean=target.mean().item(),
            cql1=cql1.item(),
            cql2=cql2.item(),
            td_loss_raw=td_loss.item(),
            cql_loss_raw=cql.item()
        )
        return total, info

    def loss_actor(self, obs_cond: Tensor, action_batch: Tensor) -> Tuple[Tensor, Dict]:
        """Actor损失计算"""
        # 行为克隆损失
        bc = self.actor.flow_bc_loss(obs_cond, action_batch)

        # Q引导损失
        actor_actions = self.actor.predict_action_chunk(obs_cond, n_steps=self.cfg.inference_steps)
        q1, q2 = self.critic(obs_cond, actor_actions)
        q = torch.min(q1, q2)
        q_loss = -q.mean()

        # 可选：Q损失归一化
        if self.cfg.normalize_q_loss:
            q_loss = q_loss * (1.0 / (torch.abs(q).mean().detach() + 1e-8))

        loss = bc + q_loss
        info = dict(
            loss_actor=loss.item(),
            loss_bc_flow=bc.item(),
            q_loss=q_loss.item(),
            q_mean=q.mean().item()
        )
        return loss, info

    def update_target(self, tau: float):
        """目标网络软更新"""
        for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
            tp.data.copy_(tp.data * (1 - tau) + p.data * tau)


# ========= LightningModule (Auto-optim, official style) =========
class LitConservativeMeanFQL(L.LightningModule):
    def __init__(self, obs_dim: int, action_dim: int, cfg: Config):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.net = ConservativeMeanFQL(obs_dim, action_dim, cfg)
        # 禁用自动优化，因为我们使用多个优化器和频率控制
        self.automatic_optimization = False
        # 初始化观测历史
        self.net.actor.reset_obs_history()

    def forward(self, obs_cond: Tensor) -> Tensor:
        """前向传播：预测动作"""
        return self.net.actor.predict_action_chunk(obs_cond, n_steps=self.cfg.inference_steps)

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int):
        device = self.device

        # 预处理数据：提取最后一个时间步的观测
        obs = batch["observations"].float().to(device)  # [B, obs_horizon, obs_dim]
        obs_cond = obs[:, -1, :]  # [B, obs_dim]

        actions = batch["action_chunks"].float().to(device)  # [B, H, A]
        next_obs = batch["next_observations"].float().to(device)  # [B, obs_horizon, obs_dim]
        next_obs_cond = next_obs[:, -1, :]  # [B, obs_dim]

        rewards = batch["rewards"].float().to(device)  # [B, H]
        terminated = batch["terminations"].float().to(device)  # [B, 1]

        # 获取优化器
        opt_c, opt_a = self.optimizers()

        # Critic step
        self.toggle_optimizer(opt_c)
        loss_c, info_c = self.net.loss_critic(
            obs_cond, actions, next_obs_cond, rewards, terminated, self.cfg.gamma
        )
        self.log("critic/loss", loss_c, on_step=True, prog_bar=True)
        for k, v in info_c.items():
            self.log(f"critic/{k}", v, on_step=True)
        opt_c.zero_grad()
        self.manual_backward(loss_c)
        if self.cfg.grad_clip_value:
            self.clip_gradients(opt_c, gradient_clip_val=self.cfg.grad_clip_value, gradient_clip_algorithm="norm")
        opt_c.step()
        self.untoggle_optimizer(opt_c)

        # Actor step - 控制 actor 更新频率：仅每 N 个 batch 更新
        if (batch_idx % self.cfg.actor_update_freq) == 0:
            self.toggle_optimizer(opt_a)
            loss_a, info_a = self.net.loss_actor(obs_cond, actions)
            self.log("actor/loss", loss_a, on_step=True, prog_bar=True)
            for k, v in info_a.items():
                self.log(f"actor/{k}", v, on_step=True)
            opt_a.zero_grad()
            self.manual_backward(loss_a)
            if self.cfg.grad_clip_value:
                self.clip_gradients(opt_a, gradient_clip_val=self.cfg.grad_clip_value, gradient_clip_algorithm="norm")
            opt_a.step()
            self.untoggle_optimizer(opt_a)

        return {"loss": loss_c}

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int):
        device = self.device

        # 预处理数据
        obs = batch["observations"].float().to(device)
        obs_cond = obs[:, -1, :]
        actions = batch["action_chunks"].float().to(device)
        next_obs = batch["next_observations"].float().to(device)
        next_obs_cond = next_obs[:, -1, :]
        rewards = batch["rewards"].float().to(device)
        terminated = batch["terminations"].float().to(device)

        # 评估critic loss
        val_loss, info = self.net.loss_critic(obs_cond, actions, next_obs_cond, rewards, terminated, self.cfg.gamma)
        self.log("val/critic_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)

        # 评估actor loss
        val_actor_loss, actor_info = self.net.loss_actor(obs_cond, actions)
        self.log("val/actor_loss", val_actor_loss, on_step=False, on_epoch=True, prog_bar=True)

        # 记录其他指标
        for k, v in info.items():
            self.log(f"val/critic_{k}", v, on_step=False, on_epoch=True)
        for k, v in actor_info.items():
            self.log(f"val/actor_{k}", v, on_step=False, on_epoch=True)

        return val_loss

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int):
        obs = batch["observations"].float().to(self.device)
        obs_cond = obs[:, -1, :]
        predicted_actions = self(obs_cond)
        actual_actions = batch["action_chunks"].float().to(self.device)
        # 只比较第一个动作
        action_mse = F.mse_loss(predicted_actions, actual_actions[:, 0, :])
        self.log("test/action_mse", action_mse, on_step=False, on_epoch=True)
        return action_mse

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # 目标网络软更新（按频率）
        if (batch_idx % self.cfg.target_update_freq) == 0:
            self.net.update_target(self.cfg.tau)

    def configure_optimizers(self):
        opt_c = optim.Adam(self.net.critic.parameters(), lr=self.cfg.learning_rate)
        opt_a = optim.Adam(self.net.actor.parameters(), lr=self.cfg.learning_rate)
        sched_c = optim.lr_scheduler.StepLR(opt_c, step_size=10, gamma=0.1)
        sched_a = optim.lr_scheduler.StepLR(opt_a, step_size=10, gamma=0.1)
        return (
            {"optimizer": opt_c, "lr_scheduler": {"scheduler": sched_c, "interval": "epoch"}},
            {"optimizer": opt_a, "lr_scheduler": {"scheduler": sched_a, "interval": "epoch"}},
        )