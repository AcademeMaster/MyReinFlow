from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
import os

import torch


@dataclass
class Config:
    """配置参数容器"""
    # 训练参数
    num_epochs: int = 100
    batch_size: int = 2048  # 减小batch size以提高训练稳定性
    learning_rate: float = 1e-4  # 进一步降低学习率
    eval_interval: int = 10
    checkpoint_dir: str = "./checkpoint_t"
    bc_alpha: float = 5.0  # 增加BC损失权重，确保模仿学习主导早期训练
    # 环境参数
    dataset_name: str = "mujoco/ant/expert-v0"
    
    # 序列参数
    obs_horizon: int = 1  # 观测序列长度，设置为1表示只使用���前观测
    pred_horizon: int = 4
    # action_horizon: int = 2
    inference_steps: int = 5  # 增加推理步数以提高采样质量
    window_stride: int = 1  # 滑动窗口步长
    
    # 模型参数
    hidden_dim: int = 512  # 减小隐藏层维度，避免过拟合
    time_dim: int = 64
    N: int = 8  # 减少Best-of-N的N值，加快训练

    # 训练参数
    gamma: float = 0.99
    tau: float = 0.01  # 增加软更新率
    target_update_freq: int = 2  # 更频繁更新目标网络
    actor_update_freq: int = 2  # 减少actor更新频率


    # BC损失权重衰减参数
    bc_loss_initial_weight: float = 10.0  # BC损失初始权重
    bc_loss_final_weight: float = 1.0    # BC损失最终权重
    bc_loss_decay_steps: int = 2000     # 衰减步数


    # CQL参数
    cql_alpha: float = 5.0  # 减小CQL系数，避免过于保守
    cql_temp: float = 1.0   # 降低温度参数
    cql_num_samples: int = 10

    
    # 更新频率参数
    q_update_period: int = 1  # 更频繁地更新Q网络
    v_update_period: int = 2
    policy_update_period: int = 2  # 降低策略更新频率

    # 其他参数
    normalize_q_loss: bool = False
    grad_clip_value: float = 0.5  # 减小梯度裁剪阈值
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
    
    # 可选的优化参数
    adaptive_alpha: bool = False  # 是否使用自适应alpha
    entropy_coeff: float = 0.0  # 熵正则化系数，0表示不使用

    # 梯度裁剪
    max_grad_norm: float = 0.5  # 减小梯度裁剪阈值

    def __post_init__(self):
        """初始化后处理"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def __repr__(self):
        """以可读方式显��所有配置"""
        return "\n".join(f"{k}: {v}" for k, v in asdict(self).items())