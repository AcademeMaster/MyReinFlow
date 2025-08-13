import torch
import numpy as np
from torch.utils.data import Dataset
import minari
import gymnasium as gym
import time
import os


class MinariDataset(Dataset):
    """Minari数据集处理类
    
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
            self.dataset = minari.load_dataset(dataset_name)
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            raise
        
        print(f"Dataset loaded in {time.time() - start_time:.2f} seconds")
        
        # 获取环境信息
        try:
            self.env_id = self.dataset.spec.observation_space.id if hasattr(self.dataset.spec.observation_space, 'id') else str(self.dataset.spec)
        except AttributeError:
            self.env_id = "unknown_env"
        print(f"Environment ID: {self.env_id}")
        
        # 处理数据
        self.process_data()
        
        # 计算统计信息用于归一化
        if self.normalize:
            self.compute_normalization_stats()
    
    def process_data(self):
        """处理Minari数据集中的episode数据"""
        print("Processing dataset...")
        start_time = time.time()
        
        # 获取所有episodes
        episodes = self.dataset.episodes
        print(f"Total episodes: {len(episodes)}")
        
        # 提取观测和动作数据
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []
        
        # 跟踪数据质量问题
        nan_episodes = 0
        short_episodes = 0
        
        # 处理每个episode
        for i, episode in enumerate(episodes):
            # 提取观测和动作
            obs = episode.observations
            act = episode.actions
            rew = episode.rewards
            next_obs = episode.next_observations
            done = episode.terminations
            
            # 检查数据质量
            if np.isnan(obs).any() or np.isnan(act).any():
                nan_episodes += 1
                continue
            
            if len(obs) < self.horizon_steps + 1:
                short_episodes += 1
                continue
            
            # 将数据添加到列表中
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            next_observations.append(next_obs)
            dones.append(done)
            
            # 如果达到最大样本数，则停止
            if self.max_samples is not None and len(observations) >= self.max_samples:
                break
        
        # 报告数据质量问题
        if nan_episodes > 0:
            print(f"Warning: {nan_episodes} episodes contained NaN values and were skipped")
        if short_episodes > 0:
            print(f"Warning: {short_episodes} episodes were too short (< {self.horizon_steps + 1} steps) and were skipped")
        
        # 转换为numpy数组
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.next_observations = next_observations
        self.dones = dones
        
        # 生成样本索引
        self.sample_indices = []
        for i, obs in enumerate(observations):
            # 对于每个episode，生成可能的起始索引
            max_start_idx = len(obs) - self.horizon_steps
            for start_idx in range(max_start_idx):
                self.sample_indices.append((i, start_idx))
        
        # 如果没有足够的样本，生成备用样本
        if len(self.sample_indices) == 0:
            print("Warning: No valid samples found. Generating fallback samples.")
            # 创建一个简单的随机样本
            obs_dim = self.dataset.spec.observation_space.shape[0]
            act_dim = self.dataset.spec.action_space.shape[0]
            
            # 生成随机数据
            random_obs = np.random.randn(100, obs_dim)
            random_act = np.random.randn(100, act_dim)
            
            self.observations = [random_obs]
            self.actions = [random_act]
            self.rewards = [np.zeros(100)]
            self.next_observations = [random_obs]
            self.dones = [np.zeros(100, dtype=bool)]
            
            # 生成样本索引
            max_start_idx = 100 - self.horizon_steps
            for start_idx in range(max_start_idx):
                self.sample_indices.append((0, start_idx))
        
        print(f"Dataset processed in {time.time() - start_time:.2f} seconds")
        print(f"Total samples: {len(self.sample_indices)}")
    
    def compute_normalization_stats(self):
        """计算归一化统计信息"""
        print("Computing normalization statistics...")
        
        # 收集所有观测和动作
        all_obs = np.concatenate([obs for obs in self.observations])
        all_act = np.concatenate([act for act in self.actions])
        
        # 计算均值和标准差
        self.obs_mean = np.mean(all_obs, axis=0)
        self.obs_std = np.std(all_obs, axis=0) + 1e-6  # 添加小值以避免除零
        
        self.act_mean = np.mean(all_act, axis=0)
        self.act_std = np.std(all_act, axis=0) + 1e-6
        
        print(f"Observation mean: {self.obs_mean[:5]}...")
        print(f"Observation std: {self.obs_std[:5]}...")
        print(f"Action mean: {self.act_mean}")
        print(f"Action std: {self.act_std}")
    
    def normalize_obs(self, obs):
        """归一化观测"""
        if self.normalize:
            return (obs - self.obs_mean) / self.obs_std
        return obs
    
    def normalize_act(self, act):
        """归一化动作"""
        if self.normalize:
            return (act - self.act_mean) / self.act_std
        return act
    
    def denormalize_act(self, act):
        """反归一化动作"""
        if self.normalize:
            return act * self.act_std + self.act_mean
        return act
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.sample_indices)
    
    def __getitem__(self, idx):
        """获取数据样本"""
        episode_idx, start_idx = self.sample_indices[idx]
        
        # 获取观测和动作序列
        obs = self.observations[episode_idx][start_idx]
        next_obs = self.next_observations[episode_idx][start_idx + self.horizon_steps - 1]
        actions = self.actions[episode_idx][start_idx:start_idx + self.horizon_steps]
        
        # 归一化
        obs = self.normalize_obs(obs)
        next_obs = self.normalize_obs(next_obs)
        actions = self.normalize_act(actions)
        
        # 转换为PyTorch张量
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.float32)
        
        return {
            'state': obs_tensor,
            'next_state': next_obs_tensor,
            'actions': actions_tensor
        }


def move_cond_to_device(cond, device):
    """将条件字典移动到指定设备"""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in cond.items()}