from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
import os

import torch


@dataclass
class Config:
    """配置参数容器"""
    # 训练参数
    num_epochs: int = 100
    batch_size: int = 2048
    learning_rate: float = 3e-4  # 降低学习率以提高稳定性
    eval_interval: int = 10
    checkpoint_dir: str = "./checkpoint_t"
    
    # 环境参数
    dataset_name: str = "mujoco/pusher/expert-v0"
    
    # 序列参数
    obs_horizon: int = 1  # 观测序列长度，设置为1表示只使用当前观测
    pred_horizon: int = 16
    # action_horizon: int = 2
    inference_steps: int = 1  # 推理步数
    window_stride: int = 1  # 滑动窗口步长
    
    # 模型参数
    hidden_dim: int = 512
    time_dim: int = 64
    N: int = 10  # Best-of-N采样的N值
    
    # 训练参数
    gamma: float = 0.99
    tau: float = 0.005
    target_update_freq: int = 1
    actor_update_freq: int = 1
    

    # BC损失权重衰减参数
    bc_loss_initial_weight: float = 1.0  # BC损失初始权重
    bc_loss_final_weight: float = 0.01    # BC损失最终权重
    bc_loss_decay_steps: int = 1000     # 衰减步数
    

    # CQL参数
    cql_alpha: float = 10.0 # 过大会使模型过于保守，从而导致模型难以收敛到最优策略，过小的会使模型忽略保守性，导致分布偏移问题。
    cql_temp: float = 10.0
    cql_num_samples: int = 10

    
    # 更新频率参数
    q_update_period: int = 1  # 更频繁地更新Q网络
    v_update_period: int = 2
    policy_update_period: int = 1  # 更频繁地更新策略网络
    
    # 其他参数
    normalize_q_loss: bool = False
    grad_clip_value: float = 1.0  # 添加梯度裁剪以提高稳定性
    num_workers: int = 0
    test_episodes: int = 20
    max_steps: int = 1000
    
    # 动作维度（在初始化时设置）
    observation_dim: int = 0
    action_dim: int = 0
    
    # Accelerator相关参数
    mixed_precision: str = "no"
    gradient_accumulation_steps: int = 1

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        """初始化后处理"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def __repr__(self):
        """以可读方式显示所有配置"""
        return "\n".join(f"{k}: {v}" for k, v in asdict(self).items())