import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset




class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling reinforcement learning training data.
    
    This buffer supports multiple data formats including D4RL and Minari datasets.
    """

    def __init__(self, state_dim: int, action_dim: int, max_size: int = int(1e6)):
        """
        Initialize the replay buffer.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            max_size: Maximum capacity of the buffer
        """
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray, 
            reward: float, done: bool):
        """
        Add a single transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            reward: Reward received
            done: Done flag (1 for terminal state, 0 otherwise)
        """
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        """
        Randomly sample a batch of transitions from the buffer.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of torch tensors (states, actions, next_states, rewards, dones)
        """
        # 确保不会尝试采样超过缓冲区中实际存储的样本数
        current_size = min(self.size, self.max_size)
        if current_size == 0:
            raise ValueError("Replay buffer is empty, cannot sample")
        if batch_size > current_size:
            raise ValueError(f"Cannot sample {batch_size} items from buffer of size {current_size}")
            
        indices = np.random.randint(0, current_size, size=batch_size)

        return (
            torch.FloatTensor(self.state[indices]).to(self.device),
            torch.FloatTensor(self.action[indices]).to(self.device),
            torch.FloatTensor(self.next_state[indices]).to(self.device),
            torch.FloatTensor(self.reward[indices]).to(self.device),
            torch.FloatTensor(self.done[indices]).to(self.device)
        )

    def convert_D4RL(self, dataset: dict):
        """
        Load data from a D4RL format dataset (for backward compatibility).
        
        Args:
            dataset: D4RL format dataset dictionary
        """
        self.state = dataset['observations'].astype(np.float32)
        self.action = dataset['actions'].astype(np.float32)
        self.next_state = dataset['next_observations'].astype(np.float32)
        self.reward = dataset['rewards'].reshape(-1, 1).astype(np.float32)
        self.done = dataset['terminals'].reshape(-1, 1).astype(np.float32)
        self.size = min(self.state.shape[0], self.max_size)
        # 确保数组大小不超过max_size
        if self.state.shape[0] > self.max_size:
            self.state = self.state[:self.max_size]
            self.action = self.action[:self.max_size]
            self.next_state = self.next_state[:self.max_size]
            self.reward = self.reward[:self.max_size]
            self.done = self.done[:self.max_size]

    def convert_minari(self, dataset):
        """
        Load data from a Minari dataset with optimized performance

        Args:
            dataset: Minari dataset object
        """
        # 预计算总数据量
        total_steps = sum(len(ep.observations) - 1 for ep in dataset.iterate_episodes())
        
        # 确保不超过缓冲区最大容量
        total_steps = min(total_steps, self.max_size)

        # 预分配内存
        self.state = np.empty((self.max_size, *dataset.observation_space.shape), dtype=np.float32)
        self.action = np.empty((self.max_size, *dataset.action_space.shape), dtype=np.float32)
        self.next_state = np.empty((self.max_size, *dataset.observation_space.shape), dtype=np.float32)
        self.reward = np.empty((self.max_size, 1), dtype=np.float32)
        self.done = np.empty((self.max_size, 1), dtype=np.bool_)

        index = 0
        for episode in dataset.iterate_episodes():
            obs = episode.observations
            actions = episode.actions
            rewards = episode.rewards
            terminations = episode.terminations
            truncations = episode.truncations

            # 计算当前episode的有效步数
            ep_len = len(obs) - 1
            # 统一处理terminated和truncated为done
            next_state_dones = np.logical_or(terminations[1:], truncations[1:])
            
            # 确保不超过缓冲区剩余空间
            if index + ep_len > self.max_size:
                ep_len = self.max_size - index
                if ep_len <= 0:
                    break

            # 批量填充数据（向量化操作）
            self.state[index:index + ep_len] = obs[:-1][:ep_len]
            self.action[index:index + ep_len] = actions[:ep_len]
            self.next_state[index:index + ep_len] = obs[1:][:ep_len]
            self.reward[index:index + ep_len] = rewards[:ep_len].reshape(-1, 1)
            # 确保done标志的长度与ep_len一致，使用min确保不越界
            done_flags = next_state_dones[:ep_len].reshape(-1, 1)
            actual_len = min(ep_len, len(done_flags))
            self.done[index:index + actual_len] = done_flags[:actual_len]

            index += ep_len
            # 如果缓冲区已满，停止填充
            if index >= self.max_size:
                break

        self.size = index

    def normalize_states(self, eps: float = 1e-3):
        """
        Normalize states in the buffer using mean and standard deviation.
        
        Args:
            eps: Small epsilon value to prevent division by zero
            
        Returns:
            Tuple of normalization parameters (mean, std)
        """
        mean = self.state.mean(0, keepdims=True)
        std = self.state.std(0, keepdims=True) + eps
        self.state = (self.state - mean) / std
        self.next_state = (self.next_state - mean) / std
        return mean, std