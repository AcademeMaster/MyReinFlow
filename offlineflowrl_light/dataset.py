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


class SlidingWindowDataset(Dataset):
    """滑动窗口数据集，提供平移的观测窗口"""

    def __init__(self, dataset, config: Config):
        """
        参数:
            dataset: Minari数据集
            config: Config对象
        """
        self.config = config
        self.episodes = []  # 存储所有样本

        # 处理每个episode
        for episode in tqdm(dataset, desc="创建滑动窗口数据集"):
            ep_len = len(episode.actions)

            # 跳过太短的序列（至少需要1个观测）
            if ep_len < 1:
                continue

            # 准备数据 - 不进行归一化
            obs = episode.observations.astype(np.float32)
            actions = episode.actions.astype(np.float32)
            rewards = episode.rewards.astype(np.float32)
            terminations = episode.terminations.astype(np.float32)

            # 确保滑动步长不超过动作序列长度
            window_stride = min(config.window_stride, ep_len)

            # 创建滑动窗口 - 包含所有可能的时间点
            for start_idx in range(0, ep_len, window_stride):
                # 1. 准备当前观测窗口
                # 计算观测窗口的起始和结束索引
                obs_start = max(0, start_idx - config.obs_horizon + 1)
                obs_end = start_idx + 1
                current_obs = obs[obs_start:obs_end]

                # 如果观测窗口不足，使用最近的观测进行填充
                if len(current_obs) < config.obs_horizon:
                    padding_count = config.obs_horizon - len(current_obs)
                    # 使用最近的观测进行填充
                    padding = np.tile(current_obs[-1], (padding_count, 1))
                    current_obs = np.concatenate([padding, current_obs])

                # 2. 准备未来观测窗口（动作块执行后的状态）
                # 计算未来观测窗口的起始索引
                next_obs_start = start_idx + config.pred_horizon

                # 处理边界情况
                if next_obs_start >= ep_len:
                    # 整个未来观测窗口都需要填充
                    next_obs = np.tile(obs[-1], (config.obs_horizon, 1))
                else:
                    next_obs_end = min(next_obs_start + config.obs_horizon, ep_len)
                    next_obs = obs[next_obs_start:next_obs_end]

                    # 如果未来观测窗口不足，使用最后一个有效观测填充
                    if len(next_obs) < config.obs_horizon:
                        padding_count = config.obs_horizon - len(next_obs)
                        padding = np.tile(next_obs[-1], (padding_count, 1))
                        next_obs = np.concatenate([next_obs, padding])

                # 3. 准备动作块
                # 计算动作块的结束索引
                act_end = min(start_idx + config.pred_horizon, ep_len)
                action_chunk = actions[start_idx:act_end]
                actual_action_length = len(action_chunk)

                # 创建填充后的动作块
                padded_action_chunk = np.zeros((config.pred_horizon, actions.shape[1]), dtype=actions.dtype)
                if actual_action_length > 0:
                    padded_action_chunk[:actual_action_length] = action_chunk

                # 4. 准备奖励序列 - 统一为2D (pred_horizon, 1)
                reward_seq = rewards[start_idx:act_end].reshape(-1, 1)  # 变为列向量
                padded_reward_seq = np.zeros((config.pred_horizon, 1), dtype=rewards.dtype)
                if actual_action_length > 0:
                    padded_reward_seq[:actual_action_length] = reward_seq
                    # 超出部分填充最后一个实际奖励
                    padded_reward_seq[actual_action_length:] = reward_seq[-1]
                else:
                    # 如果没有实际奖励，填充0
                    padded_reward_seq[:] = 0

                # 5. 准备终止标志序列 - 统一为2D (pred_horizon, 1)
                termination_seq = terminations[start_idx:act_end].reshape(-1, 1)  # 变为列向量
                padded_termination_seq = np.zeros((config.pred_horizon, 1), dtype=terminations.dtype)
                if actual_action_length > 0:
                    padded_termination_seq[:actual_action_length] = termination_seq
                    # 超出部分填充1
                    padded_termination_seq[actual_action_length:] = 1
                else:
                    # 如果没有实际终止标志，填充1
                    padded_termination_seq[:] = 1

                # 6. 计算有效长度
                valid_length = actual_action_length

                # 7. 添加到数据集
                self.episodes.append({
                    "observations": current_obs,  # 历史观测序列 (obs_horizon, obs_dim)
                    "next_observations": next_obs,  # 未来观测序列 (obs_horizon, obs_dim)
                    "action_chunks": padded_action_chunk,  # 动作块 (pred_horizon, act_dim)
                    "rewards": padded_reward_seq,  # 奖励序列 (pred_horizon, 1)
                    "terminations": padded_termination_seq,  # 终止标志 (pred_horizon, 1)
                    "valid_length": valid_length,
                    "start_idx": start_idx,
                    "episode_length": ep_len,
                    "is_terminal": (start_idx + config.pred_horizon >= ep_len)  # 标记是否为末端
                })

    def __len__(self):
        """返回数据集大小"""
        return len(self.episodes)

    def __getitem__(self, idx):
        """获取单个样本"""
        item = self.episodes[idx]
        return {
            "observations": torch.as_tensor(item["observations"], dtype=torch.float32),
            "next_observations": torch.as_tensor(item["next_observations"], dtype=torch.float32),
            "action_chunks": torch.as_tensor(item["action_chunks"], dtype=torch.float32),
            "rewards": torch.as_tensor(item["rewards"], dtype=torch.float32),
            "terminations": torch.as_tensor(item["terminations"], dtype=torch.float32),
            "valid_length": torch.tensor(item["valid_length"], dtype=torch.long),
            "start_idx": torch.tensor(item["start_idx"], dtype=torch.long),
            "episode_length": torch.tensor(item["episode_length"], dtype=torch.long),
            "is_terminal": torch.tensor(item["is_terminal"], dtype=torch.bool)
        }



class SlidingWindowCollator:
    """滑动窗口数据集的批处理函数 - 优化内存使用"""

    def __init__(self, config: Config):
        self.config = config

    def __call__(self, batch):
        if len(batch) == 0:
            return {}

        # 获取维度信息
        obs_dim = batch[0]["observations"].shape[-1]  # 观测维度
        obs_horizon = self.config.obs_horizon  # 观测窗口长度
        act_dim = batch[0]["action_chunks"].shape[-1]  # 动作维度
        pred_horizon = self.config.pred_horizon  # 预测时域长度

        # 预先分配张量
        batch_size = len(batch)

        # 观测窗口: (batch_size, obs_horizon, obs_dim)
        observations = torch.zeros((batch_size, obs_horizon, obs_dim), dtype=torch.float32)

        # 未来观测窗口: (batch_size, obs_horizon, obs_dim)
        next_observations = torch.zeros((batch_size, obs_horizon, obs_dim), dtype=torch.float32)

        # 动作块: (batch_size, pred_horizon, act_dim)
        action_chunks = torch.zeros((batch_size, pred_horizon, act_dim), dtype=torch.float32)

        # 奖励序列: (batch_size, pred_horizon, 1)
        rewards = torch.zeros((batch_size, pred_horizon, 1), dtype=torch.float32)

        # 终止标志序列: (batch_size, pred_horizon, 1)
        terminations = torch.zeros((batch_size, pred_horizon, 1), dtype=torch.float32)

        # 其他标量字段
        valid_length = torch.zeros(batch_size, dtype=torch.long)
        start_idx = torch.zeros(batch_size, dtype=torch.long)
        episode_length = torch.zeros(batch_size, dtype=torch.long)
        is_terminal = torch.zeros(batch_size, dtype=torch.bool)

        for i, item in enumerate(batch):
            # 观测窗口
            observations[i] = item["observations"]

            # 未来观测窗口
            next_observations[i] = item["next_observations"]

            # 动作块
            action_chunks[i] = item["action_chunks"]

            # 奖励序列
            rewards[i] = item["rewards"]

            # 终止标志序列
            terminations[i] = item["terminations"]

            # 其他标量字段
            valid_length[i] = item["valid_length"]
            start_idx[i] = item["start_idx"]
            episode_length[i] = item["episode_length"]
            is_terminal[i] = item["is_terminal"]

        return {
            "observations": observations,
            "next_observations": next_observations,
            "action_chunks": action_chunks,
            "rewards": rewards,
            "terminations": terminations,
            "valid_length": valid_length,
            "start_idx": start_idx,
            "episode_length": episode_length,
            "is_terminal": is_terminal
        }



class MinariDataModule(L.LightningDataModule):
    """Minari数据集模块，用于PyTorch Lightning训练"""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None

        self.collator = SlidingWindowCollator(config)
        self.normalizer = None

    def prepare_data(self):
        """准备数据集 - 在训练前调用"""
        # 加载Minari数据集
        minari_dataset = minari.load_dataset(self.config.dataset_name)
        episodes = list(minari_dataset.iterate_episodes())

        # 分割episode为训练/验证集 (80/20)
        num_episodes = len(episodes)
        train_size = int(0.8 * num_episodes)
        train_episodes = episodes[:train_size]
        val_episodes = episodes[train_size:]

        # 先创建训练集以计算统计量
        self.train_dataset = SlidingWindowDataset(train_episodes, config=self.config)




        # 验证集使用训练集的统计量，避免数据泄漏
        self.val_dataset = SlidingWindowDataset(val_episodes, config=self.config)

    def train_dataloader(self):
        """训练数据加载器"""
        persistent = self.config.num_workers > 0
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.collator,
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
            collate_fn=self.collator,
            num_workers=self.config.num_workers,
            persistent_workers=persistent
        )

    def test_dataloader(self):
        """测试数据加载器 - 使用验证集"""
        return self.val_dataloader()




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
    sliding_dataset = SlidingWindowDataset(dataset, config)

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
        print(f"是否为末端: {sample['is_terminal']}")

        # 测试批次处理
        dataloader = DataLoader(sliding_dataset, batch_size=2, collate_fn=SlidingWindowCollator(config))
        batch = next(iter(dataloader))
        print(f"批次观测数据维度: {batch['observations'].shape}")
        print(f"批次未来观测数据维度: {batch['next_observations'].shape}")
        print(f"批次动作块维度: {batch['action_chunks'].shape}")
        print(f"批次奖励序列维度: {batch['rewards'].shape}")
        print(f"批次终止标志维度: {batch['terminations'].shape}")
        print(f"批次有效长度维度: {batch['valid_length'].shape}")
        print(f"批次末端标记维度: {batch['is_terminal'].shape}")

    print("\n数据集测试完成!")