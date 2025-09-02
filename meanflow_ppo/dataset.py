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
            # 修复：确保 start + pred_horizon < episode_length，这样我们才能访问 observations[start + pred_horizon]
            # 同时确保有足够的数据用于动作和奖励序列
            max_start = episode_length - self.config.pred_horizon - 1
            if max_start >= 0:  # 确保episode足够长
                for start in range(0, max_start + 1, self.config.window_stride):
                    self.window_indices.append((episode_idx, start))

    def __len__(self):
        """返回数据集大小"""
        return len(self.window_indices)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """获取单个样本 - 已修正"""
        episode_idx, start = self.window_indices[idx]
        episode = self.episode_data[episode_idx]
        H = self.config.pred_horizon

        # 1. 当前观测 (s_t) [obs_dim]
        current_obs = episode.observations[start]

        # 2. 动作序列 (a_t, ..., a_{t+H-1}) [H, action_dim]
        action_seq = episode.actions[start:start + H]

        # 3. 奖励序列 (r_t, ..., r_{t+H-1}) [H, 1]
        reward_seq = episode.rewards[start:start + H].reshape(-1, 1)

        # 4. [新增] 终止信号序列 (d_t, ..., d_{t+H-1}) [H, 1]
        #    这是为了正确计算 N-步回报，处理窗口内的早期终止
        terminations_seq = episode.terminations[start:start + H]
        truncations_seq = episode.truncations[start:start + H]
        dones_seq = np.logical_or(terminations_seq, truncations_seq).reshape(-1, 1)

        # 5. [修正] H 步之后的下一个观测 (s_{t+H}) [obs_dim]
        final_obs_idx = start + H
        # 由于 _create_sliding_windows 的逻辑��这里不会越界
        final_next_obs = episode.observations[final_obs_idx]



        return {
            "observations": current_obs.astype(np.float32),
            "next_observations": final_next_obs.astype(np.float32),
            "action_chunks": action_seq.astype(np.float32),
            "rewards": reward_seq.astype(np.float32),
            "dones": dones_seq.astype(np.float32),  # 用于 N-步回报计算
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
        dones = torch.zeros(batch_size, self.config.pred_horizon, 1, dtype=torch.float32)
        final_dones = torch.zeros(batch_size, 1, dtype=torch.float32)
        
        # 填充批次数据
        for i, item in enumerate(batch):
            observations[i] = torch.from_numpy(item["observations"])
            next_observations[i] = torch.from_numpy(item["next_observations"])
            action_chunks[i] = torch.from_numpy(item["action_chunks"])
            rewards[i] = torch.from_numpy(item["rewards"]).float()
            dones[i] = torch.from_numpy(item["dones"]).float()


        # 确保观测数据维度正确，不需要额外处理
        observations_seq = observations  # [batch_size, obs_dim]
        next_observations_seq = next_observations  # [batch_size, obs_dim]
            
        return {
            "observations": observations_seq,
            "next_observations": next_observations_seq,
            "action_chunks": action_chunks,
            "rewards": rewards,
            "dones": dones,

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
    try:
        dataset = minari.load_dataset(config.dataset_name)
        # 创建滑动窗口数据集
        sliding_dataset = SlidingWindowDataset([dataset], config)  # 传递列表参数

        print(f"数据集大小: {len(sliding_dataset)}")
        
        # 测试获取单个样本
        if len(sliding_dataset) > 0:
            sample = sliding_dataset[0]
            print("样本结构:")
            for key, value in sample.items():
                print(f"  {key}: shape={value.shape if hasattr(value, 'shape') else 'N/A'}, dtype={value.dtype if hasattr(value, 'dtype') else type(value)}")
        
        # 测试MinariDataModule
        print("测试MinariDataModule功能...")
        data_module = MinariDataModule(config)
        data_module.setup()
        train_loader = data_module.train_dataloader()
        
        print(f"训练集大小: {len(data_module.train_dataset)}")
        print(f"验证集大小: {len(data_module.val_dataset)}")
        
        # 测试获取一个批次
        batch = next(iter(train_loader))
        print("批次结构:")
        for key, value in batch.items():
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")

        # 测试数据加载器迭代
        batch_count = 0
        for batch in tqdm(train_loader, desc="训练数据加载器"):
            batch_count += 1
            if batch_count >= 3:  # 只测试前3个批次
                break
        print(f"成功迭代 {batch_count} 个批次")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
