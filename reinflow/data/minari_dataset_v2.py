import torch
import numpy as np
from torch.utils.data import Dataset
import minari
import gymnasium as gym
from gymnasium import spaces
import time
import os


class MinariDataset(Dataset):
    """Minari数据集处理类 - 基于官方示例改进
    
    参数:
        dataset_name (str): Minari数据集名称
        horizon_steps (int): 动作序列的时间步长
        device (torch.device): 计算设备
        normalize (bool): 是否归一化数据
        max_samples (int, optional): 最大样本数量
        seed (int, optional): 随机种子
    """
    def __init__(self, dataset_name, horizon_steps, device, normalize=True, max_samples=None, seed=None):
        super().__init__()
        self.dataset_name = dataset_name
        self.horizon_steps = horizon_steps
        self.device = device
        self.normalize = normalize
        self.max_samples = max_samples
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # 加载Minari数据集
        print(f"Loading Minari dataset: {dataset_name}")
        start_time = time.time()
        try:
            self.minari_dataset = minari.load_dataset(dataset_name)
            print(f"Dataset loaded in {time.time() - start_time:.2f} seconds")
            print(f"Total episodes: {len(self.minari_dataset)}")
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            raise
        
        # 获取环境信息
        self.env = self.minari_dataset.recover_environment()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        # 获取空间维度
        if isinstance(self.observation_space, spaces.Box):
            self.obs_dim = int(np.prod(self.observation_space.shape))
        else:
            raise ValueError(f"Unsupported observation space: {type(self.observation_space)}")
        
        if isinstance(self.action_space, spaces.Box):
            self.action_dim = int(np.prod(self.action_space.shape))
            self.is_discrete = False
            self.act_min = float(self.action_space.low.min())
            self.act_max = float(self.action_space.high.max())
        elif isinstance(self.action_space, spaces.Discrete):
            self.action_dim = int(self.action_space.n)
            self.is_discrete = True
            self.act_min = 0
            self.act_max = self.action_dim - 1
        else:
            raise ValueError(f"Unsupported action space: {type(self.action_space)}")
        
        print(f"Environment: {self.env.spec.id}")
        print(f"Observation dim: {self.obs_dim}")
        print(f"Action dim: {self.action_dim}")
        print(f"Action space type: {'Discrete' if self.is_discrete else 'Continuous'}")
        print(f"Action range: [{self.act_min}, {self.act_max}]")
        
        # 处理数据
        self.process_data()
        
        # 计算统计信息用于归一化
        if self.normalize:
            self.compute_normalization_stats()
    
    def process_data(self):
        """处理Minari数据集中的episode数据 - 基于官方示例改进"""
        print("Processing dataset...")
        start_time = time.time()
        
        # 存储所有数据
        self.episodes_data = []
        
        # 跟踪数据质量问题
        nan_episodes = 0
        short_episodes = 0
        valid_episodes = 0
        
        # 处理每个episode
        for i, episode in enumerate(self.minari_dataset):
            # 检查数据质量
            if np.isnan(episode.observations).any() or np.isnan(episode.actions).any():
                nan_episodes += 1
                continue
            
            if len(episode.observations) < self.horizon_steps + 1:
                short_episodes += 1
                continue
            
            # 存储episode数据
            self.episodes_data.append({
                'observations': episode.observations,
                'actions': episode.actions,
                'rewards': episode.rewards,
                'terminations': episode.terminations,
                'truncations': episode.truncations
            })
            
            valid_episodes += 1
            
            # 如果达到最大样本数，则停止
            if self.max_samples is not None and valid_episodes >= self.max_samples:
                break
        
        # 报告数据质量
        print(f"Valid episodes: {valid_episodes}")
        if nan_episodes > 0:
            print(f"Warning: {nan_episodes} episodes contained NaN values and were skipped")
        if short_episodes > 0:
            print(f"Warning: {short_episodes} episodes were too short (< {self.horizon_steps + 1} steps) and were skipped")
        
        # 生成样本索引 - 每个样本是一个(episode_idx, start_idx)对
        self.sample_indices = []
        for ep_idx, episode in enumerate(self.episodes_data):
            obs_len = len(episode['observations'])
            # 对于每个episode，生成可能的起始索引
            max_start_idx = obs_len - self.horizon_steps
            for start_idx in range(max_start_idx):
                self.sample_indices.append((ep_idx, start_idx))
        
        # 如果没有足够的样本，创建备用样本
        if len(self.sample_indices) == 0:
            print("Warning: No valid samples found. Creating minimal sample.")
            # 创建一个简单的随机样本
            dummy_obs = np.random.randn(self.horizon_steps + 1, self.obs_dim).astype(np.float32)
            if self.is_discrete:
                dummy_actions = np.random.randint(0, self.action_dim, size=(self.horizon_steps,))
            else:
                dummy_actions = np.random.randn(self.horizon_steps, self.action_dim).astype(np.float32)
            
            self.episodes_data = [{
                'observations': dummy_obs,
                'actions': dummy_actions,
                'rewards': np.zeros(self.horizon_steps),
                'terminations': np.zeros(self.horizon_steps, dtype=bool),
                'truncations': np.zeros(self.horizon_steps, dtype=bool)
            }]
            
            self.sample_indices = [(0, 0)]
        
        print(f"Dataset processed in {time.time() - start_time:.2f} seconds")
        print(f"Total training samples: {len(self.sample_indices)}")
    
    def compute_normalization_stats(self):
        """计算归一化统计信息"""
        print("Computing normalization statistics...")
        
        # 收集所有观测和动作
        all_obs = []
        all_act = []
        
        for episode in self.episodes_data:
            all_obs.append(episode['observations'])
            all_act.append(episode['actions'])
        
        all_obs = np.concatenate(all_obs, axis=0)
        all_act = np.concatenate(all_act, axis=0)
        
        # 计算均值和标准差
        self.obs_mean = np.mean(all_obs, axis=0).astype(np.float32)
        self.obs_std = np.std(all_obs, axis=0).astype(np.float32) + 1e-6  # 添加小值以避免除零
        
        if not self.is_discrete:
            self.act_mean = np.mean(all_act, axis=0).astype(np.float32)
            self.act_std = np.std(all_act, axis=0).astype(np.float32) + 1e-6
        else:
            # 离散动作不需要归一化
            self.act_mean = np.zeros(1, dtype=np.float32)
            self.act_std = np.ones(1, dtype=np.float32)
        
        print(f"Observation mean: {self.obs_mean[:min(5, len(self.obs_mean))]}...")
        print(f"Observation std: {self.obs_std[:min(5, len(self.obs_std))]}...")
        if not self.is_discrete:
            print(f"Action mean: {self.act_mean}")
            print(f"Action std: {self.act_std}")
    
    def normalize_obs(self, obs):
        """归一化观测"""
        if self.normalize:
            return (obs - self.obs_mean) / self.obs_std
        return obs
    
    def normalize_act(self, act):
        """归一化动作"""
        if self.normalize and not self.is_discrete:
            return (act - self.act_mean) / self.act_std
        return act
    
    def denormalize_act(self, act):
        """反归一化动作"""
        if self.normalize and not self.is_discrete:
            return act * self.act_std + self.act_mean
        return act
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.sample_indices)
    
    def __getitem__(self, idx):
        """获取数据样本"""
        episode_idx, start_idx = self.sample_indices[idx]
        episode = self.episodes_data[episode_idx]
        
        # 获取观测和动作序列
        obs = episode['observations'][start_idx].copy()
        next_obs = episode['observations'][start_idx + self.horizon_steps].copy()
        actions = episode['actions'][start_idx:start_idx + self.horizon_steps].copy()
        
        # 确保数据类型正确
        obs = obs.astype(np.float32)
        next_obs = next_obs.astype(np.float32)
        if not self.is_discrete:
            actions = actions.astype(np.float32)
        
        # 归一化
        obs = self.normalize_obs(obs)
        next_obs = self.normalize_obs(next_obs)
        actions = self.normalize_act(actions)
        
        # 转换为PyTorch张量
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32)
        
        if self.is_discrete:
            actions_tensor = torch.tensor(actions, dtype=torch.long)
        else:
            actions_tensor = torch.tensor(actions, dtype=torch.float32)
        
        return {
            'state': obs_tensor,
            'next_state': next_obs_tensor,
            'actions': actions_tensor
        }


def collate_fn(batch):
    """数据批处理函数 - 基于官方示例"""
    return {
        "state": torch.stack([item["state"] for item in batch]),
        "next_state": torch.stack([item["next_state"] for item in batch]),
        "actions": torch.stack([item["actions"] for item in batch])
    }


def move_cond_to_device(cond, device):
    """将条件字典移动到指定设备"""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in cond.items()}
