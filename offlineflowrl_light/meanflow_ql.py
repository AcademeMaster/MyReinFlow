
import copy
from typing import Dict, Tuple, List
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd.functional import jvp
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
    inference_steps: int = 1  # Optimized for one-step as per MeanFlow paper
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

    def _prepare_inputs(self, obs: Tensor, actions: Tensor) -> Tensor:
        """准备输入：处理形状并拼接"""
        if actions.dim() > 3:
            actions = actions.view(-1, self.action_horizon, self.action_dim)
        elif actions.dim() == 2:
            batch_size = actions.shape[0]
            actions = actions.view(batch_size, self.action_horizon, self.action_dim)
        elif actions.dim() != 3:
            raise ValueError(f"Unexpected actions dimension: {actions.dim()}")

        batch_size, horizon, _ = actions.shape
        actions_flat = actions.reshape(batch_size, -1)

        if obs.dim() > 2:
            obs = obs.reshape(batch_size, -1)
        elif obs.dim() < 2:
            obs = obs.unsqueeze(0)

        return torch.cat([obs, actions_flat], dim=-1)

    def forward(self, obs: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        """计算状态-动作序列的双Q值"""
        combined = self._prepare_inputs(obs, actions)
        q1 = self.q1_net(combined)
        q2 = self.q2_net(combined)
        return q1, q2

    def q1(self, obs: Tensor, actions: Tensor) -> Tensor:
        """仅计算Q1值"""
        combined = self._prepare_inputs(obs, actions)
        return self.q1_net(combined)


class ImprovedTimeEmbedding(nn.Module):
    """改进的时间特征嵌入模块，使用正弦位置编码"""

    def __init__(self, time_dim: int, max_period: int = 10000):
        super().__init__()
        self.time_dim = time_dim
        self.max_period = max_period

        assert time_dim % 2 == 0, "time_dim must be even"

        half_dim = time_dim // 2
        exponents = torch.arange(half_dim, dtype=torch.float32) / half_dim
        self.freqs = 1.0 / (max_period ** exponents)

        self.mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim)
        )

    def forward(self, t: Tensor) -> Tensor:
        """嵌入时间特征"""
        t = t.float()

        if t.dim() == 1:
            t = t.unsqueeze(-1)
        elif t.dim() > 2:
            t = t.view(t.shape[0], -1)

        device = t.device
        freqs = self.freqs.to(device)
        args = t * freqs[None, :]

        sin_enc = torch.sin(args)
        cos_enc = torch.cos(args)

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

        self.time_embed = ImprovedTimeEmbedding(time_dim)
        self.obs_embed = FeatureEmbedding(obs_dim, hidden_dim)
        self.noise_embed = FeatureEmbedding(action_dim, hidden_dim)

        self.joint_processor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + time_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def _normalize_obs(self, obs: Tensor) -> Tensor:
        """规范化obs形状为 [B, obs_dim]"""
        if obs.dim() > 2:
            obs = obs.reshape(obs.shape[0], -1)
        elif obs.dim() < 2:
            obs = obs.unsqueeze(0)
        return obs

    def _normalize_z(self, z: Tensor) -> Tensor:
        """规范化z形状为 [B, pred_horizon, action_dim]"""
        if z.dim() > 3:
            z = z.view(-1, self.pred_horizon, self.action_dim)
        elif z.dim() < 3:
            if z.dim() == 2 and z.shape[1] == self.action_dim * self.pred_horizon:
                z = z.view(-1, self.pred_horizon, self.action_dim)
            else:
                raise ValueError(f"Unexpected z shape: {z.shape}")
        return z

    def _normalize_time(self, time_tensor: Tensor, target_batch_size: int) -> Tensor:
        """规范化时间张量 (t 或 r) 形状为 [B]"""
        if time_tensor.dim() == 0:
            time_tensor = time_tensor.unsqueeze(0).repeat(target_batch_size)
        elif time_tensor.dim() == 1:
            if time_tensor.shape[0] != target_batch_size:
                if time_tensor.shape[0] == 1:
                    time_tensor = time_tensor.repeat(target_batch_size)
                else:
                    # 截断或填充到目标大小
                    if time_tensor.shape[0] > target_batch_size:
                        time_tensor = time_tensor[:target_batch_size]
                    else:
                        pad = torch.zeros(target_batch_size - time_tensor.shape[0], device=time_tensor.device, dtype=time_tensor.dtype)
                        time_tensor = torch.cat([time_tensor, pad])
        elif time_tensor.dim() > 1:
            time_tensor = time_tensor.view(-1)[:target_batch_size]
        return time_tensor

    def forward(self, obs: Tensor, z: Tensor, r: Tensor, t: Tensor) -> Tensor:
        """
        前向传播计算速度场

        输入形状假设:
            obs: [B, obs_dim] 或兼容形状 (自动规范化)
            z: [B, pred_horizon, action_dim] 或兼容形状 (自动规范化)
            r: [B] 或兼容形状 (自动规范化)
            t: [B] 或兼容形状 (自动规范化)

        返回:
            速度场 [B, pred_horizon, action_dim]
        """
        obs = self._normalize_obs(obs)
        z = self._normalize_z(z)

        B, H, A = z.shape

        t = self._normalize_time(t, B)
        r = self._normalize_time(r, B)

        t_emb = self.time_embed(t)
        r_emb = self.time_embed(r)

        obs_emb = self.obs_embed(obs)

        noise_emb = self.noise_embed(z.reshape(B * H, A))
        noise_emb = noise_emb.view(B, H, -1)

        obs_emb = obs_emb.unsqueeze(1).repeat(1, H, 1)
        t_emb = t_emb.unsqueeze(1).repeat(1, H, 1)
        r_emb = r_emb.unsqueeze(1).repeat(1, H, 1)

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
        obs_cond = observations[:, -1, :]

        noise = torch.randn(
            observations.size(0),
            self.pred_horizon,
            self.action_dim,
            dtype=torch.float32,
            device=device
        )

        return self.sample_mean_flow(obs_cond, noise, n_steps=n_steps)

    @torch.no_grad()
    def select_action(self, batch: Dict[str, Tensor], n_steps: int = 1) -> Tensor:
        """根据环境观测选择单个动作"""
        action_chunk = self.predict_action_chunk(batch, n_steps=n_steps)
        return action_chunk[:, 0, :]

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

        return torch.clamp(x, -1, 1)

    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        """改进的前向传播，计算MeanFlow损失"""
        device = next(self.model.parameters()).device
        observations = batch["observations"].to(device)
        actions = batch["actions"].to(device)

        noise = torch.randn_like(actions, device=device, dtype=torch.float32)
        t, r = self.sample_t_r(actions.shape[0], device=device)

        z = (1 - t.view(-1, 1, 1)) * actions + t.view(-1, 1, 1) * noise

        obs_cond = observations[:, -1, :]
        if obs_cond.dim() == 1:
            obs_cond = obs_cond.unsqueeze(0)

        v = noise - actions

        # 确保输入张量需要梯度
        obs_cond = obs_cond.requires_grad_(True)
        z = z.requires_grad_(True)
        r = r.requires_grad_(True)
        t = t.requires_grad_(True)

        # 计算Jacobian-Vector Product
        u, dudt = jvp(
            func=self.model,
            inputs=(obs_cond, z, r, t),
            v=(torch.zeros_like(obs_cond, requires_grad=False), v, torch.zeros_like(r, requires_grad=False), torch.ones_like(t, requires_grad=False)),
            create_graph=True  # 保持梯度计算图以支持反向传播
        )

        # 计算目标速度场
        delta_t = (t - r).unsqueeze(1).unsqueeze(1)
        u_tgt = v - delta_t * dudt
        u_tgt = u_tgt.detach()

        # 使用JVP的primal输出作为预测速度场
        predicted_velocity = u

        # 计算损失
        loss = F.mse_loss(predicted_velocity, u_tgt)

        return loss


class ConservativeMeanFQLModel(nn.Module):
    """保守性MeanFQL模型，添加CQL正则化处理离线RL"""

    def __init__(self, obs_dim: int, action_dim: int, config: Config):
        super().__init__()
        self.actor = ImprovedMeanFlowPolicyAgent(obs_dim, action_dim, config)
        self.critic = DoubleCriticObsAct(obs_dim, action_dim, config.hidden_dim, config.pred_horizon)


        self.target_critic = copy.deepcopy(self.critic)
        self.config = config
        self.device = torch.device(config.device)

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=config.learning_rate
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=config.learning_rate
        )

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
        if rewards.dim() > 2:
            rewards = rewards.view(rewards.shape[0], -1)

        h = min(self.actor.pred_horizon, rewards.shape[1])
        batch_size = rewards.shape[0]

        step_indices = torch.arange(h, device=rewards.device, dtype=torch.float32)
        discount_factors = gamma ** step_indices
        discount_factors = discount_factors.unsqueeze(0)

        discounted_sum = torch.sum(
            discount_factors * rewards[:, :h],
            dim=1
        )
        return discounted_sum

    def loss_critic(self, obs: Tensor, actions: Tensor, next_obs: Tensor,
                    rewards: Tensor, terminated: Tensor, gamma: float) -> Tuple[Tensor, Dict]:
        """计算critic损失，添加CQL正则化"""
        batch_size = obs.shape[0]

        with torch.no_grad():
            batch = {
                "observations": next_obs,
                "actions": actions,  # 占位
            }
            next_actions = self.actor.predict_action_chunk(batch)
            next_actions = torch.clamp(next_actions, -1, 1)

            next_q1, next_q2 = self.target_critic(next_obs, next_actions)
            next_q = torch.min(next_q1, next_q2).view(batch_size)

            discounted_rewards = self._compute_discounted_rewards(rewards, gamma)

            terminated = terminated.view(batch_size)

            future_q = (1 - terminated.float()) * (gamma ** self.actor.pred_horizon) * next_q

            target = discounted_rewards + future_q

        q1, q2 = self.critic(obs, actions)

        target = target.view(batch_size, 1)

        td_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        # CQL保守性正则化
        num_action_samples = 10
        with torch.no_grad():
            obs_cond = obs[:, -1, :]
            repeated_obs_cond = obs_cond.repeat_interleave(num_action_samples, dim=0)
            sampled_noise = torch.randn(
                batch_size * num_action_samples, self.actor.pred_horizon, self.actor.action_dim,
                device=self.device, dtype=torch.float32
            )
            sampled_actions = self.actor.sample_mean_flow(
                repeated_obs_cond, sampled_noise, n_steps=10
            )

        repeated_obs = obs.repeat_interleave(num_action_samples, dim=0)
        q1_sampled, q2_sampled = self.critic(repeated_obs, sampled_actions)
        q1_sampled = q1_sampled.view(batch_size, num_action_samples)
        q2_sampled = q2_sampled.view(batch_size, num_action_samples)

        cql_loss1 = torch.logsumexp(q1_sampled / self.config.cql_temp, dim=1).mean() * self.config.cql_temp - q1.mean()
        cql_loss2 = torch.logsumexp(q2_sampled / self.config.cql_temp, dim=1).mean() * self.config.cql_temp - q2.mean()
        cql_loss = (cql_loss1 + cql_loss2) * self.config.cql_alpha

        total_loss = td_loss + cql_loss

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
        batch = {"observations": obs, "actions": action_batch}
        actor_actions = self.actor.predict_action_chunk(batch)
        q1, q2 = self.critic(obs, actor_actions)
        q = torch.min(q1, q2)

        q_loss = -q.mean()
        if self.config.normalize_q_loss:
            lam = 1 / torch.abs(q).mean().detach()
            q_loss = lam * q_loss

        loss_bc_flow = self.actor(batch)
        loss_actor = loss_bc_flow + alpha * q_loss

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

        batch = {}
        for key in samples[0].keys():
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
            obs = batch["observations"].to(device)
            actions = batch["actions"].to(device).squeeze(1)
            next_obs = batch["next_observations"].to(device)
            rewards = batch["rewards"].to(device).squeeze(1)
            terminated = batch["terminated"].to(device).squeeze()

            critic_loss, critic_info = model.loss_critic(
                obs, actions, next_obs, rewards, terminated, config.gamma
            )

            model.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.critic.parameters(), max_norm=config.grad_clip_value)
            model.critic_optimizer.step()

            if num_batches % 2 == 0:
                actor_loss, actor_info = model.loss_actor(obs, actions, alpha=1.0)

                model.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.actor.model.parameters(), max_norm=config.grad_clip_value)
                model.actor_optimizer.step()

                for k, v in actor_info.items():
                    epoch_metrics[k] += v

            model.update_target_critic(config.tau)

            for k, v in critic_info.items():
                epoch_metrics[k] += v

            num_batches += 1

        for k in epoch_metrics:
            epoch_metrics[k] /= num_batches
            metrics[k].append(epoch_metrics[k])

        if epoch % 10 == 0:
            print(f"Epoch {epoch}:")
            for k, v in epoch_metrics.items():
                print(f"  {k}: {v:.4f}")

    return metrics


# 示例使用代码
def main():
    """示例主函数，展示如何使用这些类"""
    config = Config()

    obs_dim = 10
    action_dim = 4
    action_horizon = 5


    model = ConservativeMeanFQLModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=config
    )

    buffer = ReplayBuffer(capacity=1000)

    for _ in range(1000):
        transition = {
            "observations": np.random.randn(1, obs_dim).astype(np.float32),
            "actions": np.random.randn(1, action_horizon, action_dim).astype(np.float32),
            "next_observations": np.random.randn(1, obs_dim).astype(np.float32),
            "rewards": np.random.randn(1, action_horizon).astype(np.float32),
            "terminated": np.random.choice([0, 1], size=(1, 1)).astype(np.float32)
        }
        buffer.add(transition)

    dataloader = DataLoader(buffer, batch_size=32, shuffle=True)

    metrics = train_offline_rl(model, dataloader, num_epochs=100, config=config)

    print("训练完成!")
    return metrics


if __name__ == "__main__":
    main()
