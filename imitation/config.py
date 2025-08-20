from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
import os

import torch


@dataclass
class Config:
    """配置参数容器"""
    # 训练参数
    num_epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-4
    eval_interval: int = 10
    checkpoint_dir: str = "./checkpoint_t"
    
    # 环境参数
    dataset_name: str = "mujoco/pusher/expert-v0"
    
    # 序列参数
    obs_horizon: int = 1
    pred_horizon: int = 64
    action_horizon: int = 32
    inference_steps: int = 1
    
    # 模型参数
    time_dim: int = 32
    hidden_dim: int = 256
    sigma: float = 0.0
    
    # 归一化
    normalize_data: bool = True
    
    # 测试参数
    test_episodes: int = 5
    max_steps: int = 300
    
    # 动作维度（在初始化时设置）
    action_dim: int = 0
    
    # Accelerator相关参数
    mixed_precision: str = "no"
    gradient_accumulation_steps: int = 1


    grad_clip_value: float = 1.0
    cql_alpha: float = 1.0
    cql_temp: float = 1.0
    tau: float = 0.005
    gamma: float = 0.99

    normalize_q_loss: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        """初始化后处理"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def __repr__(self):
        """以可读方式显示所有配置"""
        return "\n".join(f"{k}: {v}" for k, v in asdict(self).items())
