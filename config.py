from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
import os

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
    pred_horizon: int = 16
    action_horizon: int = 8
    inference_steps: int = 20
    
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
    
    def __post_init__(self):
        """初始化后处理"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def __repr__(self):
        """以可读方式显示所有配置"""
        return "\n".join(f"{k}: {v}" for k, v in asdict(self).items())