import copy
from typing import Dict, Tuple, Optional, List, Any
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from dataclasses import dataclass


@dataclass
class Config:
    """配置参数类"""
    hidden_dim: int = 256
    time_dim: int = 64
    pred_horizon: int = 5
    learning_rate: float = 3e-4
    grad_clip_value: float = 1.0
    cql_alpha: float = 1.0
    cql_temp: float = 1.0
    tau: float = 0.005
    gamma: float = 0.99
    inference_steps: int = 10
    normalize_q_loss: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class DoubleCriticObsAct(nn.Module):
    """双Q网络Critic，处理多步动作序列输入"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, action_horizon: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.action_horizon = action_horizon

        total_action_dim = action_dim * action_horizon

        # Q1网络
        self.q1_net = nn.Sequential(
            nn.Linear(obs_dim + total_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Q2网络
        self.q2_net = nn.Sequential(
            nn.Linear(obs_dim + total_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        """计算状态-动作序列的双Q值"""
        # 处理actions的形状，确保它是3维张量 [B, horizon, action_dim]
        if actions.dim() > 3:
            # 如果超过3维，将其压缩为3维
            actions = actions.view(-1, self.action_horizon, self.action_dim)
        elif actions.dim() == 2:
            # 如果是2维，假设它已经是展平的，需要恢复形状
            batch_size = actions.shape[0]
            actions = actions.view(batch_size, self.action_horizon, self.action_dim)
        elif actions.dim() != 3:
            raise ValueError(f"Unexpected actions dimension: {actions.dim()}")

        batch_size, horizon, _ = actions.shape
        actions_flat = actions.reshape(batch_size, -1)

        # 确保obs是正确的形状 [B, obs_dim]
        if obs.dim() > 2:
            obs = obs.reshape(batch_size, -1)
        elif obs.dim() < 2:
            obs = obs.unsqueeze(0)

        combined = torch.cat([obs, actions_flat], dim=-1)

        q1 = self.q1_net(combined)
        q2 = self.q2_net(combined)
        return q1, q2

    def q1(self, obs: Tensor, actions: Tensor) -> Tensor:
        """仅计算Q1值"""
        # 处理actions的形状，确保它是3维张量 [B, horizon, action_dim]
        if actions.dim() > 3:
            # 如果超过3维，将其压缩为3维
            actions = actions.view(-1, self.action_horizon, self.action_dim)
        elif actions.dim() == 2:
            # 如果是2维，假设它已经是展平的，需要恢复形状
            batch_size = actions.shape[0]
            actions = actions.view(batch_size, self.action_horizon, self.action_dim)
        elif actions.dim() != 3:
            raise ValueError(f"Unexpected actions dimension: {actions.dim()}")

        batch_size, horizon, _ = actions.shape
        actions_flat = actions.reshape(batch_size, -1)

        # 确保obs是正确的形状 [B, obs_dim]
        if obs.dim() > 2:
            obs = obs.reshape(batch_size, -1)
        elif obs.dim() < 2:
            obs = obs.unsqueeze(0)

        combined = torch.cat([obs, actions_flat], dim=-1)
        return self.q1_net(combined)


class ImprovedTimeEmbedding(nn.Module):
    """改进的时间特征嵌入模块，使用正弦位置编码"""

    def __init__(self, time_dim: int, max_period: int = 10000):
        super().__init__()
        self.time_dim = time_dim
        self.max_period = max_period

        # 确保时间维度是偶数
        assert time_dim % 2 == 0, "time_dim must be even"

        # 创建正弦位置编码参数
        half_dim = time_dim // 2
        exponents = torch.arange(half_dim, dtype=torch.float32) / half_dim
        self.freqs = 1.0 / (max_period ** exponents)

        # 可学习的变换层
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim)
        )

    def forward(self, t: Tensor) -> Tensor:
        """嵌入时间特征"""
        # 确保输入是float32类型
        t = t.float()

        if t.dim() == 1:
            t = t.unsqueeze(-1)
        elif t.dim() > 2:
            # 如果维度超过2，将其压缩为2维
            t = t.view(t.shape[0], -1)

        device = t.device

        # 计算正弦位置编码
        freqs = self.freqs.to(device)
        args = t * freqs[None, :]

        # 生成正弦和余弦编码
        sin_enc = torch.sin(args)
        cos_enc = torch.cos(args)

        # 拼接并应用MLP
        encoding = torch.cat([sin_enc, cos_enc], dim=-1)
        return self.mlp(encoding)


class FeatureEmbedding(nn.Module):
    """通用特征嵌入模块"""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        """嵌入特征"""
        return self.net(x)


class ImprovedMeanTimeConditionedFlowModel(nn.Module):
    """改进的带时间条件约束的流匹配模型"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, time_dim: int, pred_horizon: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.pred_horizon = pred_horizon

        # 模块化嵌入层
        self.time_embed = ImprovedTimeEmbedding(time_dim)
        self.obs_embed = FeatureEmbedding(obs_dim, hidden_dim)
        self.noise_embed = FeatureEmbedding(action_dim, hidden_dim)

        # 联合处理模块 - 使用更深的网络
        self.joint_processor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + time_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, obs: Tensor, z: Tensor, r: Tensor, t: Tensor) -> Tensor:
        """
        前向传播计算速度场

        参数:
            obs: 观测张量 [B, obs_dim]
            z: 混合样本 [B, pred_horizon, action_dim]
            r: 参考时间步 [B] 或 [1]
            t: 当前时间步 [B] 或 [1]

        返回:
            速度场 [B, pred_horizon, action_dim]
        """
        # 确保所有张量在正确的设备上
        device = z.device
        
        # 确保观测张量维度正确
        if obs.dim() > 2:
            obs = obs.view(obs.shape[0], -1)

        # 确保z张量维度正确
        if z.dim() > 3:
            z = z.view(-1, self.pred_horizon, self.action_dim)
        elif z.dim() < 3:
            # 如果是2维张量，尝试重构为3维
            if z.shape[-1] == self.action_dim * self.pred_horizon:
                z = z.view(-1, self.pred_horizon, self.action_dim)
            else:
                raise ValueError(f"Unexpected z dimension: {z.dim()}")

        # 获取正确的B, H, A值（从z张量获取批量大小）
        B, H, A = z.shape

        # 处理时间张量，确保它们与z的批量大小匹配
        def process_time_tensor(time_tensor, batch_size):
            if time_tensor.dim() == 0:
                # 标量情况
                return time_tensor.unsqueeze(0).repeat(batch_size)
            elif time_tensor.dim() == 1:
                if time_tensor.shape[0] == 1:
                    # 单个值需要广播到批量大小
                    return time_tensor.repeat(batch_size)
                elif time_tensor.shape[0] == batch_size:
                    # 已经是正确的大小
                    return time_tensor
                else:
                    # 其他情况，截取或填充
                    if time_tensor.shape[0] > batch_size:
                        return time_tensor[:batch_size]
                    else:
                        padding = time_tensor.new_zeros(batch_size - time_tensor.shape[0])
                        return torch.cat([time_tensor, padding])
            else:
                # 多维张量，展平并调整到正确的大小
                time_tensor = time_tensor.view(-1)
                if time_tensor.shape[0] >= batch_size:
                    return time_tensor[:batch_size]
                else:
                    repeats = (batch_size + time_tensor.shape[0] - 1) // time_tensor.shape[0]
                    return time_tensor.repeat(repeats)[:batch_size]

        t = process_time_tensor(t, B)
        r = process_time_tensor(r, B)

        # 嵌入时间特征
        t_emb = self.time_embed(t)  # [B, time_dim]
        r_emb = self.time_embed(r)  # [B, time_dim]

        # 嵌入观测特征
        obs_emb = self.obs_embed(obs)  # [B, hidden_dim]

        # 嵌入噪声特征
        noise_emb = self.noise_embed(z.reshape(B * H, A))  # [B*H, hidden_dim]
        noise_emb = noise_emb.view(B, H, -1)  # [B, H, hidden_dim]

        # 扩展观测和时间嵌入以匹配时间维度
        obs_emb = obs_emb.unsqueeze(1).repeat(1, H, 1)  # [B, H, hidden_dim]
        t_emb = t_emb.unsqueeze(1).repeat(1, H, 1)      # [B, H, time_dim]
        r_emb = r_emb.unsqueeze(1).repeat(1, H, 1)      # [B, H, time_dim]

        # 合并特征并预测速度场
        combined = torch.cat([obs_emb, noise_emb, r_emb, t_emb], dim=-1)
        return self.joint_processor(combined)


class ImprovedMeanFlowPolicyAgent(nn.Module):
    """改进的MeanFlow策略代理"""

    def __init__(self, obs_dim: int, action_dim: int, config: Config):
        super().__init__()
        self.config = config
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = config.hidden_dim
        self.time_dim = config.time_dim
        self.pred_horizon = config.pred_horizon
        self.learning_rate = config.learning_rate
        self.grad_clip_value = config.grad_clip_value

        self.model = ImprovedMeanTimeConditionedFlowModel(
            self.obs_dim, self.action_dim, self.hidden_dim, self.time_dim, self.pred_horizon
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4  # 添加权重衰减
        )

    @staticmethod
    def sample_t_r(n_samples: int, device: str = 'cpu') -> Tuple[Tensor, Tensor]:
        """采样时间点t和r"""
        t = torch.rand(n_samples, device=device, dtype=torch.float32)
        r = torch.rand(n_samples, device=device, dtype=torch.float32) * t
        return t, r

    @torch.no_grad()
    def predict_action_chunk(self, batch: Dict[str, Tensor], n_steps: int = 1) -> Tensor:
        """预测动作块"""
        self.model.eval()
        device = next(self.model.parameters()).device

        observations = batch["observations"].to(device)
        obs_cond = observations[:, -1, :]  # 使用最新的观测作为条件

        # 生成初始噪声
        noise = torch.randn(
            observations.size(0),
            self.pred_horizon,
            self.action_dim,
            dtype=torch.float32,
            device=device
        )

        # 使用MeanFlow采样
        return self.sample_mean_flow(obs_cond, noise, n_steps=n_steps)

    @torch.no_grad()
    def select_action(self, batch: Dict[str, Tensor], n_steps: int = 1) -> Tensor:
        """根据环境观测选择单个动作"""
        action_chunk = self.predict_action_chunk(batch, n_steps=n_steps)
        return action_chunk[:, 0, :]  # 返回动作块中的第一个动作

    @torch.no_grad()
    def sample_mean_flow(self, obs_cond: Tensor, noise: Tensor, n_steps: int = 1) -> Tensor:
        """使用MeanFlow进行采样生成动作"""
        device = next(self.model.parameters()).device
        obs_cond, x = obs_cond.to(device), noise.to(device)
        dt = 1.0 / n_steps

        for i in range(n_steps, 0, -1):
            r = torch.full((x.shape[0],), (i - 1) * dt, device=device, dtype=torch.float32)
            t = torch.full((x.shape[0],), i * dt, device=device, dtype=torch.float32)
            velocity = self.model(obs_cond, x, r, t)
            x = x - velocity * dt

        return x

    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        """改进的前向传播，计算MeanFlow损失"""
        device = next(self.model.parameters()).device
        observations = batch["observations"].to(device)
        actions = batch["actions"].to(device)

        # 生成噪声和时间步
        noise = torch.randn_like(actions, device=device, dtype=torch.float32)
        t, r = self.sample_t_r(actions.shape[0], device=device)

        # 创建混合样本
        z = (1 - t.view(-1, 1, 1)) * actions + t.view(-1, 1, 1) * noise

        # 使用最新的观测作为条件，并确保维度正确
        obs_cond = observations[:, -1, :]
        
        # 确保所有张量维度一致
        batch_size = z.shape[0]
        if obs_cond.shape[0] != batch_size:
            if obs_cond.shape[0] == 1:
                obs_cond = obs_cond.repeat(batch_size, 1)
            else:
                obs_cond = obs_cond[:batch_size] if obs_cond.shape[0] > batch_size else torch.cat([
                    obs_cond, obs_cond.new_zeros(batch_size - obs_cond.shape[0], obs_cond.shape[1])
                ])

        # 确保时间张量维度正确
        if t.shape[0] != batch_size:
            if t.shape[0] == 1:
                t = t.repeat(batch_size)
            else:
                t = t[:batch_size] if t.shape[0] > batch_size else torch.cat([
                    t, t.new_zeros(batch_size - t.shape[0])
                ])
                
        if r.shape[0] != batch_size:
            if r.shape[0] == 1:
                r = r.repeat(batch_size)
            else:
                r = r[:batch_size] if r.shape[0] > batch_size else torch.cat([
                    r, r.new_zeros(batch_size - r.shape[0])
                ])

        # 直接计算流匹配损失
        predicted_velocity = self.model(obs_cond, z, r, t)

        # 目标速度场: 从噪声到真实动作的方向
        target_velocity = noise - actions

        return F.mse_loss(predicted_velocity, target_velocity)


class ConservativeMeanFQLModel(nn.Module):
    """保守性MeanFQL模型，添加CQL正则化处理离线RL"""

    def __init__(self, actor: ImprovedMeanFlowPolicyAgent, critic: DoubleCriticObsAct,
                 config: Config):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.target_critic = copy.deepcopy(self.critic)
        self.config = config
        self.device = torch.device(config.device)

        # 优化器
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=config.learning_rate
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=config.learning_rate
        )

        # 移动到设备
        self.to(self.device)

    def to(self, device):
        """重写to方法以确保所有组件都移动到设备"""
        super().to(device)
        self.actor.to(device)
        self.critic.to(device)
        self.target_critic.to(device)
        return self

    def forward(self, cond: Dict[str, Tensor]) -> Tensor:
        """前向传播，生成动作"""
        return self.actor.predict_action_chunk(cond, n_steps=self.config.inference_steps)

    def _compute_discounted_rewards(self, rewards: Tensor, gamma: float) -> Tensor:
        """计算折扣奖励总和"""
        # 确保使用正确的动作视野长度
        # print(f"_compute_discounted_rewards - 输入 rewards 形状: {rewards.shape}")
        # 展平rewards张量，使其形状为 [batch_size, reward_dim]
        if rewards.dim() > 2:
            rewards = rewards.view(rewards.shape[0], -1)
        # print(f"_compute_discounted_rewards - 展平后 rewards 形状: {rewards.shape}")
        
        h = min(self.actor.pred_horizon, rewards.shape[1])  # 确保不超过实际奖励序列长度
        batch_size = rewards.shape[0]
        
        step_indices = torch.arange(h, device=rewards.device, dtype=torch.float32)
        discount_factors = gamma ** step_indices  # 形状 [h]
        
        # 确保discount_factors形状为 [1, h] 以便广播
        discount_factors = discount_factors.unsqueeze(0)  # 形状 [1, h]
        
        # 计算折扣奖励总和，结果应该是一个形状为 [batch_size] 的张量
        discounted_sum = torch.sum(
            discount_factors * rewards[:, :h],  # [batch_size, h] * [1, h] => [batch_size, h]
            dim=1  # 在时间维度上求和，得到 [batch_size]
        )
        # print(f"_compute_discounted_rewards - 输出 discounted_sum 形状: {discounted_sum.shape}")
        return discounted_sum  # 返回形状 [batch_size]

    def loss_critic(self, obs: Tensor, actions: Tensor, next_obs: Tensor,
                    rewards: Tensor, terminated: Tensor, gamma: float) -> Tuple[Tensor, Dict]:
        """计算critic损失，添加CQL正则化"""
        batch_size = obs.shape[0]

        with torch.no_grad():
            # 获取下一状态的动作
            batch = {
                "observations": next_obs,
                "actions": actions,  # 这里只是占位，实际不会使用
            }
            next_actions = self.actor.predict_action_chunk(batch)
            next_actions = torch.clamp(next_actions, -1, 1)

            # 计算目标Q值
            next_q1, next_q2 = self.target_critic(next_obs, next_actions)
            next_q = torch.min(next_q1, next_q2).view(batch_size)  # 确保形状为 [batch_size]

            # 计算折扣奖励
            discounted_rewards = self._compute_discounted_rewards(rewards, gamma)  # 形状 [batch_size]

            # 处理terminated张量
            terminated = terminated.view(batch_size)  # 确保形状为 [batch_size]
            
            # 组合目标Q值 (考虑终止状态)
            future_q = (1 - terminated.float()) * (gamma ** self.actor.pred_horizon) * next_q
            
            # 调试信息
            # print(f"discounted_rewards shape: {discounted_rewards.shape}")
            # print(f"future_q shape: {future_q.shape}")
            
            # 计算最终目标，所有张量都应该是 [batch_size] 形状
            target = discounted_rewards + future_q

        # 计算当前Q值
        q1, q2 = self.critic(obs, actions)
        print(f"loss_critic - q1 形状: {q1.shape}")
        print(f"loss_critic - q2 形状: {q2.shape}")
        
        # 确保目标张量维度匹配Q值维度
        target = target.view(batch_size, 1)  # 调整为 [batch_size, 1] 以匹配Q值形状
        print(f"loss_critic - target 重塑后形状: {target.shape}")
            
        td_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        print(f"loss_critic - td_loss 形状: {td_loss.shape}")

        # CQL保守性正则化
        # 从当前策略采样动作
        with torch.no_grad():
            sampled_actions = self.actor.sample_mean_flow(
                obs[:, -1, :],  # 使用最新的观测
                torch.randn(batch_size, self.actor.pred_horizon, actions.shape[-1],
                            device=self.device, dtype=torch.float32),
                n_steps=10
            )

        # 计算当前数据和采样数据的Q值
        q1_sampled, q2_sampled = self.critic(obs, sampled_actions)

        # CQL正则化项
        cql_loss1 = torch.logsumexp(q1_sampled / self.config.cql_temp, dim=0).mean() * self.config.cql_temp - q1.mean()
        cql_loss2 = torch.logsumexp(q2_sampled / self.config.cql_temp, dim=0).mean() * self.config.cql_temp - q2.mean()
        cql_loss = (cql_loss1 + cql_loss2) * self.config.cql_alpha

        # 组合损失
        total_loss = td_loss + cql_loss

        # 调试信息
        loss_critic_info = {
            'td_loss': td_loss.item(),
            'cql_loss': cql_loss.item(),
            'total_critic_loss': total_loss.item(),
            'q1_mean': q1.mean().item(),
            'q2_mean': q2.mean().item(),
            'target_mean': target.mean().item(),
        }

        return total_loss, loss_critic_info

    def loss_actor(self, obs: Tensor, action_batch: Tensor, alpha: float = 1.0) -> Tuple[Tensor, Dict]:
        """计算actor损失"""
        # 获取Q损失
        batch = {"observations": obs, "actions": action_batch}
        actor_actions = self.actor.predict_action_chunk(batch)
        q1, q2 = self.critic(obs, actor_actions)
        q = torch.min(q1, q2)

        q_loss = -q.mean()
        if self.config.normalize_q_loss:
            lam = 1 / torch.abs(q).mean().detach()
            q_loss = lam * q_loss

        # 获取BC损失
        loss_bc_flow = self.actor(batch)
        loss_actor = loss_bc_flow + alpha * q_loss

        # 调试信息
        loss_actor_info = {
            'loss_actor': loss_actor.item(),
            'loss_bc_flow': loss_bc_flow.item(),
            'q_loss': q_loss.item(),
            'q_mean': q.mean().item(),
        }

        return loss_actor, loss_actor_info

    def update_target_critic(self, tau: float):
        """更新目标critic网络"""
        for target_param, source_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                source_param.data * tau + target_param.data * (1.0 - tau)
            )


class ReplayBuffer(Dataset):
    """简单的经验回放缓冲区"""

    def __init__(self, capacity: int = 1000000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, transition: Dict[str, np.ndarray]):
        """添加转换到缓冲区"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def add_batch(self, transitions: List[Dict[str, np.ndarray]]):
        """批量添加转换"""
        for transition in transitions:
            self.add(transition)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """随机采样一批数据"""
        indices = np.random.randint(0, len(self.buffer), batch_size)
        samples = [self.buffer[i] for i in indices]

        # 转换为张量
        batch = {}
        for key in samples[0].keys():
            # 确保numpy数组是float32类型，然后再转换为torch张量
            np_array = np.stack([s[key] for s in samples])
            if np_array.dtype != np.float32:
                np_array = np_array.astype(np.float32)
            batch[key] = torch.from_numpy(np_array)

        return batch


def train_offline_rl(
        model: ConservativeMeanFQLModel,
        dataloader: DataLoader,
        num_epochs: int,
        config: Config
) -> Dict[str, List[float]]:
    """完整的离线强化学习训练循环"""
    device = model.device
    metrics = defaultdict(list)

    for epoch in range(num_epochs):
        epoch_metrics = defaultdict(float)
        num_batches = 0

        for batch in dataloader:
            # 转移到设备
            obs = batch["observations"].to(device)
            actions = batch["actions"].to(device)
            next_obs = batch["next_observations"].to(device)
            rewards = batch["rewards"].to(device)
            terminated = batch["terminated"].to(device)

            # 更新critic
            critic_loss, critic_info = model.loss_critic(
                obs, actions, next_obs, rewards, terminated, config.gamma
            )

            model.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.critic.parameters(), max_norm=config.grad_clip_value)
            model.critic_optimizer.step()

            # 更新actor (较少频率)
            if num_batches % 2 == 0:  # 每2个batch更新一次actor
                actor_loss, actor_info = model.loss_actor(obs, actions, alpha=1.0)

                model.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.actor.model.parameters(), max_norm=config.grad_clip_value)
                model.actor_optimizer.step()

                # 记录actor指标
                for k, v in actor_info.items():
                    epoch_metrics[k] += v

            # 更新目标网络
            model.update_target_critic(config.tau)

            # 记录critic指标
            for k, v in critic_info.items():
                epoch_metrics[k] += v

            num_batches += 1

        # 计算epoch平均指标
        for k in epoch_metrics:
            epoch_metrics[k] /= num_batches
            metrics[k].append(epoch_metrics[k])

        # 打印进度
        if epoch % 10 == 0:
            print(f"Epoch {epoch}:")
            for k, v in epoch_metrics.items():
                print(f"  {k}: {v:.4f}")

    return metrics


# 示例使用代码
def main():
    """示例主函数，展示如何使用这些类"""
    # 创建配置
    config = Config()

    # 假设的环境维度
    obs_dim = 10
    action_dim = 4
    action_horizon = 5

    # 创建模型组件
    actor = ImprovedMeanFlowPolicyAgent(obs_dim, action_dim, config)
    critic = DoubleCriticObsAct(obs_dim, action_dim, config.hidden_dim, action_horizon)

    # 创建完整模型
    model = ConservativeMeanFQLModel(actor, critic, config)

    # 创建示例数据加载器
    buffer = ReplayBuffer(capacity=1000)

    # 添加一些示例数据
    for _ in range(1000):
        transition = {
            "observations": np.random.randn(1, obs_dim).astype(np.float32),
            "actions": np.random.randn(1, action_horizon, action_dim).astype(np.float32),
            "next_observations": np.random.randn(1, obs_dim).astype(np.float32),
            "rewards": np.random.randn(1, action_horizon).astype(np.float32),  # 与动作序列长度一致
            "terminated": np.random.choice([0, 1], size=(1, 1)).astype(np.float32)  # 保持二维
        }
        buffer.add(transition)

    dataloader = DataLoader(buffer, batch_size=32, shuffle=True)

    # 训练模型
    metrics = train_offline_rl(model, dataloader, num_epochs=100, config=config)

    print("训练完成!")
    return metrics


if __name__ == "__main__":
    main()