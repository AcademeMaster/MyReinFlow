import copy
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from torch import Tensor
from torch.autograd.functional import jvp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========= Small modules =========
class FeatureEmbedding(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TimeEmbedding(nn.Module):
    def __init__(self, time_dim: int, max_period: int = 10_000):
        super().__init__()
        half_dim = time_dim // 2
        exponents = torch.arange(half_dim, dtype=torch.float32) / half_dim
        freqs = max_period ** -exponents
        self.register_buffer("freqs", freqs, persistent=False)
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.LayerNorm(time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        args = t.flatten().unsqueeze(-1) * self.freqs.unsqueeze(0)
        enc = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.mlp(enc)


# ========= Time-conditioned meanflow_ppo model (predicts velocity [B,H,A]) =========
class MeanTimeCondFlow(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, time_dim: int, ):
        super().__init__()
        self.obs_dim, self.action_dim = obs_dim, action_dim

        self.t_embed = TimeEmbedding(time_dim)
        self.r_embed = TimeEmbedding(time_dim)
        self.obs_encoder = FeatureEmbedding(obs_dim, hidden_dim)
        self.noise_embed = FeatureEmbedding(action_dim, hidden_dim)

        # 简化网络结构，直接连接所有特征
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + time_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: Tensor, z: Tensor, r: Tensor, t: Tensor) -> Tensor:
        # 简化前向传播过程
        obs_encoded = self.obs_encoder(obs)
        noise_encoded = self.noise_embed(z)
        t_encoded = self.t_embed(t)
        r_encoded = self.r_embed(r)

        # 直接拼接所有特征
        x = torch.cat([obs_encoded, noise_encoded, t_encoded, r_encoded], dim=-1)
        return self.net(x)


# ========= Actor (MeanFlow) =========
class MeanFlowActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, action_scale: float = 2.0, horizon: int = 1):
        super().__init__()

        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.horizon = horizon
        # 网络输出维度调整为 action_dim * horizon
        self.output_dim = action_dim * horizon

        self.model = MeanTimeCondFlow(obs_dim, self.output_dim, hidden_dim=256, time_dim=128)

        # 定义动作边界
        self.action_scale = action_scale
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_single_step_action(self, multi_step_action: Tensor, step_idx: int = 0) -> Tensor:
        """从多步action中提取单步action"""
        batch_size = multi_step_action.shape[0]
        start_idx = step_idx * self.action_dim
        end_idx = start_idx + self.action_dim
        return multi_step_action[:, start_idx:end_idx]

    def reshape_multi_step_action(self, flat_action: Tensor) -> Tensor:
        """将扁平化的multi-step action重塑为 [batch, horizon, action_dim]"""
        batch_size = flat_action.shape[0]
        return flat_action.view(batch_size, self.horizon, self.action_dim)

    @torch.no_grad()
    def predict_action_chunk(self, obs: Tensor, n_steps: int = 1) -> Tensor:
        self.model.eval()
        obs = obs.to(self.device)
        action_chunk = self.forward(obs, n_steps=n_steps)
        return torch.clamp(action_chunk, -self.action_scale, self.action_scale)

    def forward(self, obs: Tensor, n_steps: int = 1) -> Tensor:
        """优化的均值流采样 - 避免原地操作以支持梯度计算"""
        obs = obs.to(self.device)
        batch_size = obs.size(0)

        # 预分配初始噪声
        z_0 = torch.randn(batch_size, self.output_dim, device=self.device)

        # 优化步数处理
        n_steps = max(1, int(n_steps))
        dt = 1.0 / n_steps

        # 预分配时间张量，避免循环中重复创建
        time_steps = torch.arange(n_steps, 0, -1, dtype=torch.float32, device=self.device)
        r_values = (time_steps - 1) * dt
        t_values = time_steps * dt

        # 预分配广播后的时间张量
        r_batch = r_values.unsqueeze(1).expand(-1, batch_size).contiguous()  # [n_steps, batch_size]
        t_batch = t_values.unsqueeze(1).expand(-1, batch_size).contiguous()  # [n_steps, batch_size]

        # 优化ODE求解循环 - 避免原地操作
        for i in range(n_steps):
            r = r_batch[i]  # [batch_size]
            t = t_batch[i]  # [batch_size]
            v = self.model(obs, z_0, r, t)
            z_0 = z_0 - v * dt  # 使用非原地操作以保持梯度

        return torch.clamp(z_0, -self.action_scale, self.action_scale)  # 使用非原地操作

    def per_sample_flow_bc_loss(self, obs: Tensor, action: Tensor) -> Tensor:
        """修改BC损失以支持多步action"""
        batch_size = action.shape[0]
        action_dim_total = action.shape[1]

        # 对每个(obs, action)对进行多次采样
        num_samples = 5

        # 扩展batch维度 - 为每个样本创建多个副本
        obs_expanded = obs.unsqueeze(1).repeat(1, num_samples, 1).view(batch_size * num_samples, -1)
        action_expanded = action.unsqueeze(1).repeat(1, num_samples, 1).view(batch_size * num_samples, action_dim_total)

        # 为每个扩展后的样本生成不同的随机噪声
        z_0 = torch.randn_like(action_expanded, device=self.device)
        z_1 = action_expanded
        t = torch.rand(batch_size * num_samples, 1, device=self.device)
        r = torch.rand(batch_size * num_samples, 1, device=self.device) * t
        # 插值计算,t在0到1之间
        z_t = (1 - t) * z_1 + t * z_0
        v = z_0 - z_1

        # 准备梯度计算
        t_scalar = t.squeeze(-1).requires_grad_(True)
        r_scalar = r.squeeze(-1).requires_grad_(True)
        z_t = z_t.requires_grad_(True)

        # JVP向量
        v_obs = torch.zeros_like(obs_expanded)
        v_z = v
        v_r = torch.zeros_like(r_scalar)
        v_t = torch.ones_like(t_scalar)

        # JVP计算
        u_pred, dudt = jvp(lambda obs_in, z_in, r_in, t_in: self.model(obs_in, z_in, r_in, t_in),
                           (obs_expanded, z_t, r_scalar, t_scalar),
                           (v_obs, v_z, v_r, v_t),
                           create_graph=True)

        # 目标计算
        delta = torch.clamp(t_scalar - r_scalar, min=1e-6)[:, None]
        u_tgt = (v - delta * dudt).detach()

        # 计算每个样本的损失，不进行reduction
        per_sample_losses = F.huber_loss(u_pred, u_tgt, reduction='none').mean(dim=-1)

        # 重新组织成 [B, num_samples] 的形状，然后对每个原始样本的多次采样求平均
        per_sample_losses = per_sample_losses.view(batch_size, num_samples)
        sample_averaged_losses = per_sample_losses.mean(dim=1)

        # 最后对整个batch求平均
        return sample_averaged_losses.mean()


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_scale, horizon=1):
        super(Actor, self).__init__()

        self.horizon = horizon
        self.action_dim = action_dim
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim*horizon)

        self.action_scale = action_scale
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  # 将模型移动到指定设备

    def forward(self, state, n_steps=1):
        # 确保输入在正确的设备上
        state = state.to(self.device)
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.action_scale * torch.tanh(self.l3(a))

    def get_single_step_action(self, multi_step_action: Tensor, step_idx: int = 0) -> Tensor:
        """从多步action中提取单步action"""
        batch_size = multi_step_action.shape[0]
        start_idx = step_idx * self.action_dim
        end_idx = start_idx + self.action_dim
        return multi_step_action[:, start_idx:end_idx]

    def reshape_multi_step_action(self, flat_action: Tensor) -> Tensor:
        """将扁平化的multi-step action重塑为 [batch, horizon, action_dim]"""
        batch_size = flat_action.shape[0]
        return flat_action.view(batch_size, self.horizon, self.action_dim)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.Q1_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Q2 architecture
        self.Q2_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.Q1_net(sa)
        q2 = self.Q2_net(sa)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.Q1_net(sa)
        return q1


class TD3_BC(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            alpha=2.5,
            train_mode='offline',
            horizon=1  # 添加horizon参数
    ):

        # 使用多步MeanFlowActor，传入horizon参数
        self.actor=Actor(state_dim, action_dim, action_scale=max_action, horizon=horizon)
        # self.actor = MeanFlowActor(state_dim, action_dim, action_scale=max_action, horizon=horizon).to(device)
        self.actor_target = copy.deepcopy(self.actor)

        # 根据训练模式调整学习率
        actor_lr = 1e-4 if train_mode == "online" else 3e-4
        critic_lr = 3e-4 if train_mode == "online" else 3e-4
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Critic需要处理多步action，所以action维度变为action_dim * horizon
        self.critic = Critic(state_dim, action_dim * horizon).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha
        self.horizon = horizon
        self.device = device  # 添加device属性

        self.total_it = 0
        self.train_mode = train_mode

        # 多步执行相关的状态维护 - 精����版实现
        self.action_cache = None  # 缓存的���������步actions [horizon, action_dim]
        self.cache_index = 0     # 当前使用的action索引
        self.n_step = 1

        # 延迟更新，保证策略学习稳定
        self.lmbda = 0.01
        self.lmbda_freq = policy_freq*10
        # 添加Lambda计算的参数
        self.lmbda_factor = 1e4
        self.lmbda_eps = 1e-6
        self.lmbda_max = 100.0
        self.lmbda_min = 0.001

    def _calculate_multi_step_reward(self, reward_sequences, done_sequences, gamma=None):
        """
        高效计算多步折扣��励 - 向量化实现
        multi_step_rewards = sum_{t=0}^{horizon-1} gamma^t * reward_t, 直到done为止

        Args:
            reward_sequences: [batch_size, horizon, 1] 奖励序列
            done_sequences: [batch_size, horizon, 1] done序列
            gamma: 折扣因子，如果为None则使用self.discount

        Returns:
            multi_step_rewards: [batch_size, 1] 多步折扣奖励
        """
        if gamma is None:
            gamma = self.discount

        batch_size, horizon, _ = reward_sequences.shape

        # 向量化计算折扣因子矩阵 [horizon]
        discount_factors = gamma ** torch.arange(horizon, device=self.device, dtype=torch.float32)

        # 创建episode终止掩码 - 找到每个样本中第一个done的位置
        done_mask = done_sequences.squeeze(-1)  # [batch_size, horizon]

        # 计算累积done掩码，用于处理episode边界
        # 一旦episode结束，后续步骤都应该被屏蔽
        cumulative_done = torch.cumsum(done_mask, dim=1)  # [batch_size, horizon]

        # 创建有效步骤掩码：在episode结束前的步骤为1，结束后为0
        # 但��包含episode结束的那一步
        valid_mask = (cumulative_done <= 1).float()  # [batch_size, horizon]

        # 应用掩码到奖励
        masked_rewards = reward_sequences.squeeze(-1) * valid_mask  # [batch_size, horizon]

        # 向量化计算折扣奖励：每个样本的所有步骤同时计算
        discounted_rewards = masked_rewards * discount_factors.unsqueeze(0)  # [batch_size, horizon]

        # 沿horizon维度求和得到多步奖励
        multi_step_rewards = discounted_rewards.sum(dim=1, keepdim=True)  # [batch_size, 1]

        return multi_step_rewards

    def _calculate_multi_step_done(self, done_sequences):
        """
        高效计算多步done标志 - 向量化实现

        Args:
            done_sequences: [batch_size, horizon, 1] done序列

        Returns:
            multi_step_done: [batch_size, 1] ��步done标志
        """
        # 向量化处理：检��每个序列是否有任何done=True的步骤
        done_mask = done_sequences.squeeze(-1)  # [batch_size, horizon]

        # 如果序列中任何一步done，则整个序列done
        # 使用max操作：如果任何位置为1，则结果为1
        multi_step_done = done_mask.max(dim=1, keepdim=True)[0]  # [batch_size, 1]

        return multi_step_done

    def select_action(self, state, n_step=None):
        """智能动作选择函数 - 修复状态维度问题"""
        # 更新推理步数
        if n_step is not None:
            self.n_step = n_step

        # 检查是否需要重新推理
        if self.action_cache is None or self.cache_index >= self.horizon:
            with torch.no_grad():
                # 确保state是正确的形状 [batch_size, state_dim]
                if isinstance(state, np.ndarray):
                    if state.ndim == 1:
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    elif state.ndim == 2 and state.shape[0] == 1:
                        state_tensor = torch.FloatTensor(state).to(device)
                    else:
                        state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0).to(device)
                else:
                    state_tensor = torch.FloatTensor(state).to(device)
                    if state_tensor.ndim == 1:
                        state_tensor = state_tensor.unsqueeze(0)

                # 直接生成多步action并重塑
                multi_step_action = self.actor(state_tensor, self.n_step)
                self.action_cache = self.actor.reshape_multi_step_action(multi_step_action)
                self.cache_index = 0

        # 获取并返回当前action
        current_action = self.action_cache[0, self.cache_index].cpu().numpy()
        self.cache_index += 1
        return current_action

    def reset_action_cache(self):

        self.action_cache = None
        self.cache_index = 0

    def set_horizon(self, horizon):

        self.horizon = horizon
        self.reset_action_cache()  # 重置缓存以应用新的horizon

    def set_inference_steps(self, n_step):

        self.n_step = n_step

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # 统一使用多步采样 - 当horizon=1时自然退化为单步
        state, action_chunks, next_state, reward_sequences, done_sequences = replay_buffer.sample(batch_size)

        # 统一处理action维度 - 扁平化为 [batch_size, horizon * action_dim]
        action = action_chunks.view(batch_size, -1)

        # 统一计算多步奖励和done标志
        reward = self._calculate_multi_step_reward(reward_sequences, done_sequences)
        done = self._calculate_multi_step_done(done_sequences)

        # 预计算折扣因子，避免重复计算
        discount_factor = self.discount ** self.horizon

        with torch.no_grad():
            # 统一的噪声处理
            noise = torch.randn_like(action) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)

            # 修复重复调用问题
            next_action = self.actor_target(next_state) + noise
            next_action = next_action.clamp(-self.max_action, self.max_action)

            # 统一的目标Q值计算
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            
            # 使用标准的Bellman方程计算
            target_Q = reward + (1.0 - done) * discount_factor * target_Q

        # 当前Q值估计
        current_Q1, current_Q2 = self.critic(state, action)

        # 优化critic损失计算 - 使用Huber损失提高鲁棒性
        critic_loss = F.huber_loss(current_Q1, target_Q) + F.huber_loss(current_Q2, target_Q)

        # 优化critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # 添加梯度裁剪以提高训练稳定性
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # 优化Lambda自适应更新
        if self.total_it % self.lmbda_freq == 0:
            with torch.no_grad():  # Lambda更新不需要梯度
                new_lmbda = 1.0 / (critic_loss.detach() * self.lmbda_factor + self.lmbda_eps)
                new_lmbda = torch.clamp(new_lmbda, self.lmbda_min, self.lmbda_max)
                self.lmbda = 0.9 * self.lmbda + 0.1 * new_lmbda.item()

        # 延迟策略更新
        if self.total_it % self.policy_freq == 0:
            pi = self.actor(state)
            Q = self.critic.Q1(state, pi)
            # 优化actor损失计算
            if self.train_mode == "offline":
                lmbda = self.alpha / Q.abs().mean().detach()
                bc_loss = self.actor.per_sample_flow_bc_loss(state, action)
                actor_loss = -lmbda * Q.mean() + bc_loss
            else:  # online模式 - 移除有问题的lambda计算
                actor_loss = -Q.mean()

            # 优化actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # 添加梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            # 优化目标网络更新 - 使用更高效的原地操作
            with torch.no_grad():
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.mul_(1 - self.tau).add_(param, alpha=self.tau)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.mul_(1 - self.tau).add_(param, alpha=self.tau)

    def save(self, filename):
        # 确保目录存在
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # 保存critic网络和其优化器
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        # 保存actor网络和其优化器
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

        # 保存target网络和其优化器（如��需要）
        torch.save(self.critic_target.state_dict(), filename + "_critic_target")
        torch.save(self.actor_target.state_dict(), filename + "_actor_target")

    def load(self, filename):
        # 检查文件是否存在
        actor_path = filename + "_actor"
        critic_path = filename + "_critic"

        if not os.path.exists(actor_path):
            print(f"找不到actor模型文件: {actor_path}")
            return

        if not os.path.exists(critic_path):
            print(f"找不到critic模型文��: {critic_path}")
            return

        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
