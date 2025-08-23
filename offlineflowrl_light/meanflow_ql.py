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


class SequenceEncoder(nn.Module):
    """编码观测序列的模块"""

    def __init__(self, obs_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        layers = []
        input_size = obs_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(input_size, hidden_dim))
            layers.append(nn.ReLU())
            input_size = hidden_dim
        layers.append(nn.Linear(input_size, hidden_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, seq_len, obs_dim]
        B, seq_len, _ = x.shape
        x_flat = x.reshape(B * seq_len, -1)
        encoded = self.net(x_flat)
        return encoded.reshape(B, seq_len, -1)


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

        # 修改为处理单个观测的编码器
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
        # obs: [B, obs_dim] - 单个观测
        # actions: [B, H, A] or [B, H*A]
        if actions.dim() == 3:
            act = actions.reshape(actions.shape[0], -1)
        elif actions.dim() == 2:
            act = actions
        else:
            raise ValueError(f"bad actions dim={actions.dim()}")

        # 编码单个观测
        obs_encoded = self.obs_encoder(obs)  # [B, hidden_dim]

        return torch.cat([obs_encoded, act], dim=-1)

    def forward(self, obs: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        x = self._prep(obs, actions)
        return self.q1(x), self.q2(x)


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
        # 修改为处理单个观测的编码器
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
        # obs: [B, obs_dim] - 单个观测
        z = self._norm_z(z)
        B, H, A = z.shape
        t = self._norm_time(t, B)
        r = self._norm_time(r, B)

        te = self.t_embed(t)  # [B, Td]
        re = self.r_embed(r)  # [B, Td]

        # 编码单个观测
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
        # 添加训练步数计数器
        self.training_steps = 0

        # 定义动作边界
        self.action_bound = 2.0

    @staticmethod
    def sample_t_r(n: int, device) -> Tuple[Tensor, Tensor]:
        t = torch.rand(n, device=device)
        r = torch.rand(n, device=device) * t
        return t, r

    def predict_action_chunk(self, obs: Tensor, n_steps: int = 1) -> Tensor:
        self.model.eval()
        device = next(self.parameters()).device
        obs = obs.to(device)
        return self.sample_mean_flow(obs,n_steps=n_steps)*self.action_bound

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
        x = torch.tanh(x)
        return x

    def flow_bc_loss(self, obs: Tensor, action_chunk: Tensor) -> Tensor:
        """MeanFlow 训练损失（JVP 版）"""
        device = next(self.parameters()).device
        noise = torch.randn_like(action_chunk)
        t, r = self.sample_t_r(action_chunk.shape[0], device=device)
        z = (1 - t.view(-1, 1, 1)) * action_chunk + t.view(-1, 1, 1) * noise
        v = noise - action_chunk

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
        return F.mse_loss(u_pred, u_tgt)
    
    def get_bc_weight(self) -> float:
        """
        计算当前BC损失的权重
        从初始权重线性衰减到最终权重
        """
        # 增加训练步数计数
        self.training_steps += 1
        
        # 线性衰减
        initial_weight = self.cfg.bc_loss_initial_weight
        final_weight = self.cfg.bc_loss_final_weight
        decay_steps = self.cfg.bc_loss_decay_steps
        
        if self.training_steps >= decay_steps:
            return final_weight
        
        # 线性插值
        weight = initial_weight - (initial_weight - final_weight) * (self.training_steps / decay_steps)
        return weight


# ========= Whole RL model (Actor + Double Q + target) =========
class ConservativeMeanFQL(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.actor = MeanFlowActor(obs_dim, action_dim, cfg)
        # 修改critic以适应单个观测
        self.critic = DoubleCriticObsAct(obs_dim, action_dim, cfg.hidden_dim,
                                         cfg.pred_horizon)
        self.target_critic = copy.deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad = False

        # 初始化损失统计
        self._td_loss_stats = 1.0
        self._cql_loss_stats = 1.0

    def discounted_returns(self, rewards: Tensor, gamma: float) -> Tensor:
        """
        计算H步折扣回报（从t到t+H-1）
        rewards: [B, H, 1] 包含从t到t+H-1的奖励
        返回: [B] 每个序列的折扣回报
        """
        gamma = max(0.0, min(1.0, gamma))
        # print(f"**************************************************")
        # print(f"rewards shape: {rewards.shape}, gamma: {gamma}")
        B, H, rew_dim = rewards.shape
        # 确保rewards是2D [B, H]
        rewards_squeezed = rewards.squeeze(-1) if rew_dim == 1 else rewards.view(B, H)
        factors = (gamma ** torch.arange(H, device=rewards.device, dtype=rewards.dtype)).unsqueeze(0)
        return torch.sum(rewards_squeezed * factors, dim=1)  # [B]

    def loss_critic(self, obs: Tensor, actions: Tensor, next_obs: Tensor,
                    rewards: Tensor, terminated: Tensor, gamma: float) -> Tuple[Tensor, Dict]:
        """
        Critic损失计算
        obs: [B, obs_dim] - 单个观测
        actions: [B, H, A] - 动作序列
        next_obs: [B, obs_dim] - 下一个观测
        rewards: [B, H, 1] - 从t到t+H-1的奖励
        terminated: [B, 1] - 终止标志
        gamma: 折扣因子
        """
        B = obs.shape[0]
        with torch.no_grad():
            # 使用目标网络计算下一状态的Q值
            next_actions = self.actor.predict_action_chunk(next_obs, n_steps=self.cfg.inference_steps)
            next_q1, next_q2 = self.target_critic(next_obs, next_actions)
            next_q = torch.min(next_q1, next_q2).view(B)

            # 计算H步折扣回报（从t到t+H-1）
            h_step_returns = self.discounted_returns(rewards, gamma)  # [B]

            # 计算bootstrap项
            done = terminated.view(-1).float()  # 终止标志 [B]
  
            future_value = (1.0 - done) * (gamma ** self.cfg.pred_horizon) * next_q

            # 正确的TD目标
            target = h_step_returns + future_value  # [B]

        # 计算当前Q值
        q1, q2 = self.critic(obs, actions)  # [B,1]
        q1, q2 = q1.view(B), q2.view(B)

        # TD损失
        td_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        # CQL正则项
        num_samples = self.cfg.cql_num_samples
        rep_obs = obs.repeat_interleave(num_samples, dim=0)
        sampled_actions = torch.rand(B * num_samples, self.cfg.pred_horizon, self.action_dim,
                                     device=obs.device) * 2 - 1  # [-1, 1]
        # pusher Action Space:Box(-2.0, 2.0, (7,), float32)，这里需要优化根据环境自适应
        sampled_actions=2*sampled_actions
        # 计算采样动作的Q值
        q1s, q2s = self.critic(rep_obs, sampled_actions)
        q1s = q1s.view(B, num_samples)
        q2s = q2s.view(B, num_samples)

        temp = self.cfg.cql_temp
        # CQL损失计算
        cql1 = torch.logsumexp(q1s / temp, dim=1).mean() * temp - q1.mean()
        cql2 = torch.logsumexp(q2s / temp, dim=1).mean() * temp - q2.mean()
        cql_loss = (cql1 + cql2) * self.cfg.cql_alpha



        total = td_loss + cql_loss
        info = dict(
            td_loss=td_loss.item(),
            cql_loss=cql_loss.item(),
            total_critic_loss=total.item(),
            q1_mean=q1.mean().item(),
            q2_mean=q2.mean().item(),
            target_mean=target.mean().item(),
            cql1=cql1.item(),
            cql2=cql2.item(),
        )
        return total, info

    def loss_actor(self, obs: Tensor, action_batch: Tensor) -> Tuple[Tensor, Dict]:
        """Actor损失计算"""
        # 行为克隆损失
        bc_loss = self.actor.flow_bc_loss(obs, action_batch)

        # Q引导损失
        actor_actions = self.actor.predict_action_chunk(obs, n_steps=self.cfg.inference_steps)
        q1, q2 = self.critic(obs, actor_actions)
        q = torch.min(q1, q2)
        q_loss = -q.mean()

        # 计算当前BC损失权重
        bc_weight = self.actor.get_bc_weight()

        # Q损失权重为1 - BC权重
        q_weight = 1.0 - bc_weight

        # 组合损失函数
        loss = q_weight * q_loss + bc_weight * bc_loss
        info = dict(
            loss_actor=loss.item(),
            loss_bc_flow=bc_loss.item(),
            q_loss=q_loss.item(),
            q_mean=q.mean().item(),
            bc_weight=bc_weight,  # 记录当前BC权重
            q_weight=q_weight     # 记录当前Q权重
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

    def forward(self, obs: Tensor) -> Tensor:
        """前向传播：预测动作"""
        return self.net.actor.predict_action_chunk(obs, n_steps=self.cfg.inference_steps)

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int):
        device = self.device

        # 使用单个观测而不是观测序列
        obs = batch["observations"].float().to(device)  # [B, obs_dim]
        obs = obs.squeeze(1)  # 从 [B, 1, obs_dim] 转换为 [B, obs_dim]
        actions = batch["action_chunks"].float().to(device)  # [B, H, A]
        next_obs = batch["next_observations"].float().to(device)  # [B, obs_dim]
        next_obs = next_obs.squeeze(1)  # 从 [B, 1, obs_dim] 转换为 [B, obs_dim]
        rewards = batch["rewards"].float().to(device)  # [B, H]
        terminated = batch["terminations"].float().to(device)  # [B, 1]

        # 获取优化器
        opt_c, opt_a = self.optimizers()

        # Critic step
        self.toggle_optimizer(opt_c)
        loss_c, info_c = self.net.loss_critic(
            obs, actions, next_obs, rewards, terminated, self.cfg.gamma
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
            loss_a, info_a = self.net.loss_actor(obs, actions)
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

        # 使用单个观测而不是观测序列
        obs = batch["observations"].float().to(device)
        obs = obs.squeeze(1)  # 从 [B, 1, obs_dim] 转换为 [B, obs_dim]
        actions = batch["action_chunks"].float().to(device)
        next_obs = batch["next_observations"].float().to(device)
        next_obs = next_obs.squeeze(1)  # 从 [B, 1, obs_dim] 转换为 [B, obs_dim]
        rewards = batch["rewards"].float().to(device)
        terminated = batch["terminations"].float().to(device)

        # 评估critic loss
        val_loss, info = self.net.loss_critic(obs, actions, next_obs, rewards, terminated, self.cfg.gamma)
        self.log("val/critic_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)

        # 评估actor loss
        val_actor_loss, actor_info = self.net.loss_actor(obs, actions)
        self.log("val/actor_loss", val_actor_loss, on_step=False, on_epoch=True, prog_bar=True)

        # 获取BC权重
        bc_weight = actor_info.get("bc_weight", 0.0)  # 默认为0.0，如果不存在则使用该值
        self.log("val/bc_weight", bc_weight, on_step=False, on_epoch=True, prog_bar=True)

        # 记录奖励
        self.log("val/reward", rewards.mean(), on_step=False, on_epoch=True, prog_bar=True)

        # 记录其他指标
        for k, v in info.items():
            self.log(f"val/critic_{k}", v, on_step=False, on_epoch=True)
        for k, v in actor_info.items():
            self.log(f"val/actor_{k}", v, on_step=False, on_epoch=True)

        return val_loss

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int):
        obs = batch["observations"].float().to(self.device)
        obs = obs.squeeze(1)  # 从 [B, 1, obs_dim] 转换为 [B, obs_dim]
        predicted_actions = self(obs)
        actual_actions = batch["action_chunks"].float().to(self.device)
        # 只比较第一个动作
        action_mse = F.mse_loss(predicted_actions, actual_actions)
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