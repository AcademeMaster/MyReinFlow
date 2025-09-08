import random
import copy
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from collections import deque
from torch import Tensor

from torch.func import jvp


# ========= Device Management =========
def get_device(prefer_cuda: bool = True) -> torch.device:
    """统一的设备获取函数"""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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



# ========= Time-conditioned meanflow_ppo model (predicts velocity [B,H,A]) =========
class MeanFlowActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int ):
        super().__init__()
        self.obs_dim, self.action_dim = obs_dim, action_dim

        self.t_embed = FeatureEmbedding(obs_dim, hidden_dim)
        self.r_embed = FeatureEmbedding(obs_dim, hidden_dim)
        self.obs_encoder = FeatureEmbedding(obs_dim, hidden_dim)
        self.noise_embed = FeatureEmbedding(action_dim, hidden_dim)

        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: Tensor, z: Tensor, r: Tensor, t: Tensor) -> Tensor:
        # 简化前向传播过程
        obs_encoded = self.obs_encoder(obs)
        noise_encoded = self.noise_embed(z)

        # 将r和t扩展为与obs相同的维度
        r_expanded = r.unsqueeze(-1).expand(-1, self.obs_dim)
        t_expanded = t.unsqueeze(-1).expand(-1, self.obs_dim)
            
        t_encoded = self.t_embed(t_expanded)
        r_encoded = self.r_embed(r_expanded)

        # 直接拼接所有特征
        x = torch.cat([obs_encoded, noise_encoded, t_encoded, r_encoded], dim=-1)
        return self.net(x)



# ========= Actor (MeanFlow) =========
class MeanFlowPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, max_action: float = 2.0, device: torch.device = None):
        super().__init__()

        self.action_dim = action_dim
        self.obs_dim = obs_dim

        self.model = MeanFlowActor(obs_dim, self.action_dim, hidden_dim=256)

        # 定义动作边界
        self.max_action = max_action
        # 使用传入的device或默认device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 将模型移动到指定设备
        self.to(self.device)

    def forward(self, obs: Tensor, n_steps: int = 1) -> Tensor:
        """优化的均值流采样 - 避免原地操作以支持梯度计算"""
        obs = obs.to(self.device)
        batch_size = obs.size(0)

        # 预分配初始噪声
        z_0 = torch.randn(batch_size, self.action_dim, device=self.device)

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

        return torch.clamp(z_0, -self.max_action, self.max_action)  # 使用非原地操作

    def per_sample_flow_bc_loss(self, obs: Tensor, action: Tensor) -> Tensor:
        """修改BC损失以支持多步action"""
        # 确保输入张量在正确的设备上
        obs = obs.to(self.device)
        action = action.to(self.device)
        
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

        # JVP计算 - 移除create_graph参数
        u_pred, dudt = jvp(lambda obs_in, z_in, r_in, t_in: self.model(obs_in, z_in, r_in, t_in),
                           (obs_expanded, z_t, r_scalar, t_scalar),
                           (v_obs, v_z, v_r, v_t))

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

if __name__ == '__main__':
    # 统一设备管理
    device = get_device(prefer_cuda=True)
    print(f"使用设备: {device}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if device.type == 'cuda':
        print(f"GPU设备名称: {torch.cuda.get_device_name()}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 测试参数设置
    obs_dim = 17  # 观察空间维度
    action_dim = 6  # 动作空间维度
    max_action = 1.0  # 最大动作值
    batch_size = 32  # 批次大小
    
    # 初始化模型，传入统一的device
    actor = MeanFlowPolicy(obs_dim=obs_dim, action_dim=action_dim, max_action=max_action, device=device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)

    test_obs = torch.randn(batch_size, obs_dim).to(device)
    test_actions = torch.randn(batch_size, action_dim).clamp(-max_action, max_action).to(device)
    
    # 测试前向传播
    print("测试前向传播...")
    output_actions = actor(test_obs)
    print(f"输入观察维度: {test_obs.shape}")
    print(f"输出动作维度: {output_actions.shape}")
    print(f"动作范围: [{output_actions.min().item():.3f}, {output_actions.max().item():.3f}]")
    
    # 测试BC损失计算
    print("\n测试BC损失计算...")
    loss = actor.per_sample_flow_bc_loss(test_obs, test_actions)
    print(f"BC损失值: {loss.item():.3f}")
    
    print("\n所有测试完成!")
