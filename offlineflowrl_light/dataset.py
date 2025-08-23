import minari
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import lightning as L
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import collections
from config import Config  # 确保Config类在config.py中定义
from torch import Tensor


class SlidingWindowDataset(Dataset):
    def __init__(self, minari_datasets: List, config: Config):
        self.config = config
        self.episode_data = []
        self.window_indices = []
        
        # 收集所有数据
        for minari_dataset in minari_datasets:
            for episode in minari_dataset.iterate_episodes():
                self.episode_data.append(episode)
                
        # 创建滑动窗口索引
        self._create_sliding_windows()
        
    def _create_sliding_windows(self):
        """创建滑动窗口索引"""
        self.window_indices = []
        for episode_idx, episode in enumerate(self.episode_data):
            episode_length = len(episode.observations)
            # 确保窗口不会超出序列边界
            for start in range(0, episode_length  - self.config.pred_horizon,
                              self.config.window_stride):
                self.window_indices.append((episode_idx, start))
                
    def __len__(self):
        """返回数据集大小"""
        return len(self.window_indices)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """获取单个样本"""
        episode_idx, start = self.window_indices[idx]
        episode = self.episode_data[episode_idx]
        
        # 提取当前观测 [obs_dim]
        current_obs = episode.observations[start]
        
        # 提取下一个观测 [obs_dim]
        next_obs = episode.observations[start + 1] if start + 1 < len(episode.observations) else episode.observations[start]
        
        # 提取动作序列 [pred_horizon, action_dim]

        action_seq = episode.actions[start:start + self.config.pred_horizon]
        
        # 提取奖励序列 [pred_horizon, 1]
        reward_seq = episode.rewards[start:start + self.config.pred_horizon].reshape(-1, 1)
        
        # 只取动作序列最后一个时间步的终止标志 [1]
        terminated = episode.terminations[start + self.config.pred_horizon - 1] if \
            start + self.config.pred_horizon - 1 < len(episode.terminations) else True

        # 计算有效长度
        valid_length = min(self.config.pred_horizon, len(episode.rewards) - start)
        
        return {
            "observations": current_obs.astype(np.float32),
            "next_observations": next_obs.astype(np.float32),
            "action_chunks": action_seq.astype(np.float32),
            "rewards": reward_seq.astype(np.float32),
            "terminations": np.array([terminated], dtype=np.float32),  # 只返回一个标量值
            "valid_length": np.array(valid_length, dtype=np.int32),
        }


class SlidingWindowCollator:
    """滑动窗口数据集的批处理函数 - 优化内存使用"""

    def __init__(self, config: Config):
        self.config = config

    def __call__(self, batch: List[Dict[str, np.ndarray]]) -> Dict[str, Tensor]:
        """将样本列表合并为批次"""
        if len(batch) == 0:
            return {}
            
        batch_size = len(batch)
        
        # 初始化批次张量
        observations = torch.zeros(
            batch_size, 
            batch[0]["observations"].shape[-1],  # [batch_size, obs_dim]
            dtype=torch.float32
        )
        next_observations = torch.zeros(
            batch_size, 
            batch[0]["next_observations"].shape[-1],  # [batch_size, obs_dim]
            dtype=torch.float32
        )
        action_chunks = torch.zeros(
            batch_size, 
            self.config.pred_horizon, 
            batch[0]["action_chunks"].shape[-1],
            dtype=torch.float32
        )
        rewards = torch.zeros(batch_size, self.config.pred_horizon, 1, dtype=torch.float32)
        terminations = torch.zeros(batch_size, 1, dtype=torch.float32)
        valid_length = torch.zeros(batch_size, dtype=torch.long)
        
        # 填充批次数据
        for i, item in enumerate(batch):
            observations[i] = torch.from_numpy(item["observations"])
            next_observations[i] = torch.from_numpy(item["next_observations"])
            action_chunks[i] = torch.from_numpy(item["action_chunks"])
            rewards[i] = torch.from_numpy(item["rewards"]).float()
            terminations[i] = torch.from_numpy(item["terminations"]).float()
            valid_length[i] = torch.tensor(item["valid_length"].item())

        # 为了与模型兼容，将观测扩展为序列格式 [batch_size, 1, obs_dim]
        observations_seq = observations.unsqueeze(1)  # [batch_size, 1, obs_dim]
        next_observations_seq = next_observations.unsqueeze(1)  # [batch_size, 1, obs_dim]
            
        return {
            "observations": observations_seq,
            "next_observations": next_observations_seq,
            "action_chunks": action_chunks,
            "rewards": rewards,
            "terminations": terminations,
            "valid_length": valid_length,
        }



class MinariDataModule(L.LightningDataModule):
    """Minari数据集模块，用于PyTorch Lightning训练"""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str = None):
        """设置数据集"""
        # 加载Minari数据集
        minari_dataset = minari.load_dataset(self.config.dataset_name)
        
        # 创建滑动窗口数据集
        full_dataset = SlidingWindowDataset([minari_dataset], self.config)
        
        # 划分训练集和验证集
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )

    def train_dataloader(self):
        """训练数据加载器"""
        persistent = self.config.num_workers > 0
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=SlidingWindowCollator(self.config),
            num_workers=self.config.num_workers,
            persistent_workers=persistent
        )

    def val_dataloader(self):
        """验证数据加载器"""
        persistent = self.config.num_workers > 0
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=SlidingWindowCollator(self.config),
            num_workers=self.config.num_workers,
            persistent_workers=persistent
        )

    def test_dataloader(self):
        """测试数据加载器"""
        persistent = self.config.num_workers > 0
        return DataLoader(
            self.val_dataset,  # 使用验证集进行测试
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=SlidingWindowCollator(self.config),
            num_workers=self.config.num_workers,
            persistent_workers=persistent
        )

# 测试代码
if __name__ == "__main__":
    # 测试数据集功能
    print("测试SlidingWindowDataset功能...")

    # 创建配置
    config = Config()
    # 加载pusher数据集
    # Action：SpaceBox(-2.0, 2.0, (7,), float32)
    # Observation：SpaceBox(-inf, inf, (23,), float64)
    dataset = minari.load_dataset(config.dataset_name)
    # 创建滑动窗口数据集
    sliding_dataset = SlidingWindowDataset([dataset], config)  # 传递列表参数

    print(f"数据集大小: {len(sliding_dataset)}")

    # 测试获取数据
    if len(sliding_dataset) > 0:
        sample = sliding_dataset[0]
        print(f"观测数据维度: {sample['observations'].shape}")
        print(f"未来观测数据维度: {sample['next_observations'].shape}")
        print(f"动作块维度: {sample['action_chunks'].shape}")
        print(f"奖励序列维度: {sample['rewards'].shape}")
        print(f"终止标志维度: {sample['terminations'].shape}")
        print(f"有效长度: {sample['valid_length']}")


        # 测试批次处理
        dataloader = DataLoader(sliding_dataset, batch_size=2, collate_fn=SlidingWindowCollator(config))
        batch = next(iter(dataloader))
        print(f"批次观测数据维度: {batch['observations'].shape}")
        print(f"批次未来观测数据维度: {batch['next_observations'].shape}")
        print(f"批次动作块维度: {batch['action_chunks'].shape}")
        print(f"批次奖励序列维度: {batch['rewards'].shape}")
        print(f"批次终止标志维度: {batch['terminations'].shape}")
        print(f"批次有效长度维度: {batch['valid_length'].shape}")

    print("\n数据集测试完成!")