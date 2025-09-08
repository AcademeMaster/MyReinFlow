import numpy as np
import torch


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling reinforcement learning training data.
    
    This buffer supports multiple data formats including D4RL and Minari datasets.
    Enhanced with action chunking and multi-step rewards for diffusion model training
    and multi-step offline reinforcement learning.
    """

    def __init__(self, state_dim: int, action_dim: int, max_size: int = int(1e6),
                 horizon: int = 1):
        """
        Initialize the replay buffer.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            max_size: Maximum capacity of the buffer
            horizon: Number of consecutive steps for action chunking and reward accumulation
        """
        # 参数验证
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")

        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.horizon = horizon

        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

        # 存储episode边界信息，用于正确处理action chunking和multi-step rewards
        self.episode_starts = np.zeros(max_size, dtype=np.bool_)
        self.episode_ends = np.zeros(max_size, dtype=np.bool_)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray, 
            reward: float, done: bool = None, terminated: bool = None, truncated: bool = None,
            episode_start: bool = False):
        """
        Add a single transition to the buffer.
        支持gymnasium格式的terminated和truncated参数

        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            reward: Reward received
            done: Done flag (for backward compatibility)
            terminated: Episode terminated naturally (gymnasium format)
            truncated: Episode truncated due to time limit (gymnasium format)
            episode_start: Whether this is the start of a new episode
        """
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward

        # 处理gymnasium格式的terminated和truncated
        if terminated is not None and truncated is not None:
            done_flag = terminated or truncated
        elif done is not None:
            done_flag = done
        else:
            raise ValueError("Must provide either 'done' or both 'terminated' and 'truncated'")

        self.done[self.ptr] = done_flag
        self.episode_starts[self.ptr] = episode_start
        self.episode_ends[self.ptr] = done_flag

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def _get_action_chunk(self, start_idx: int) -> np.ndarray:
        """
        Get action chunk starting from start_idx.
        正确处理episode边界，确保不跨越episode

        Args:
            start_idx: Starting index for action chunk

        Returns:
            Action chunk of shape (horizon, action_dim)
        """
        current_size = min(self.size, self.max_size)
        actions = []
        episode_ended = False
        last_valid_action = None

        for i in range(self.horizon):
            idx = (start_idx + i) % current_size

            # 检查是否跨越了episode边界
            if i > 0:
                prev_idx = (start_idx + i - 1) % current_size
                # 如果前一步是episode结束，标记episode已结束
                if self.episode_ends[prev_idx]:
                    episode_ended = True
                # 如果当前步是新episode开始，也标记episode已结束（跨越边界）
                if self.episode_starts[idx]:
                    episode_ended = True

            if episode_ended and last_valid_action is not None:
                # 用最后一个有效action填充剩余位置
                actions.append(last_valid_action.copy())
            else:
                # 正常情况，使用当前action
                actions.append(self.action[idx])
                last_valid_action = self.action[idx]

        return np.stack(actions)

    def sample(self, batch_size: int, return_chunks: bool = True):
        """
        高效的批量采样方法 - 大幅优化版本
        减少循环，使用向量化操作和内存预分配
        """
        current_size = min(self.size, self.max_size)
        if current_size == 0:
            raise ValueError("Replay buffer is empty, cannot sample")

        # 批量采样索引
        indices = np.random.choice(current_size, size=batch_size, replace=batch_size > current_size)

        # 预分配输出数组，避免动态增长
        action_chunks = np.empty((batch_size, self.horizon, self.action.shape[1]), dtype=np.float32)
        reward_sequences = np.empty((batch_size, self.horizon, 1), dtype=np.float32)
        done_sequences = np.empty((batch_size, self.horizon, 1), dtype=np.float32)

        # 计算所有序列的索引矩阵 [batch_size, horizon]
        horizon_offsets = np.arange(self.horizon)
        all_indices = (indices[:, None] + horizon_offsets) % current_size  # 广播操作

        # 批量检查episode边界 - 向量化版本
        episode_ends_batch = self.episode_ends[all_indices]  # [batch_size, horizon]
        episode_starts_batch = self.episode_starts[all_indices]  # [batch_size, horizon]

        # 找到每个样本序列中的边界位置
        for b in range(batch_size):
            seq_indices = all_indices[b]

            # 快速检查边界：如果有边界，找到第一个边界位置
            boundaries = np.where(np.logical_or(
                episode_ends_batch[b, :-1],  # 前一步结束
                episode_starts_batch[b, 1:]   # 当前步开始
            ))[0]

            if len(boundaries) > 0:
                # 有边界，截断序列
                valid_len = boundaries[0] + 1

                # 填充有效部分
                action_chunks[b, :valid_len] = self.action[seq_indices[:valid_len]]
                reward_sequences[b, :valid_len] = self.reward[seq_indices[:valid_len]]
                done_sequences[b, :valid_len] = self.done[seq_indices[:valid_len]]

                # 填充无效部分
                if valid_len < self.horizon and valid_len > 0:
                    # 用最后有效值填充action
                    last_action = self.action[seq_indices[valid_len-1]]
                    action_chunks[b, valid_len:] = last_action
                    reward_sequences[b, valid_len:] = 0.0
                    done_sequences[b, valid_len:] = 1.0
            else:
                # 无边界，直接使用整个序列
                action_chunks[b] = self.action[seq_indices]
                reward_sequences[b] = self.reward[seq_indices]
                done_sequences[b] = self.done[seq_indices]

        # 批量转换为tensor
        return (
            torch.from_numpy(self.state[indices]).to(self.device),
            torch.from_numpy(action_chunks).to(self.device),
            torch.from_numpy(self.next_state[indices]).to(self.device),
            torch.from_numpy(reward_sequences).to(self.device),
            torch.from_numpy(done_sequences).to(self.device)
        )

    def _get_reward_sequence(self, start_idx: int) -> np.ndarray:
        """
        获取从start_idx开始的reward序列，供算法自己计算多步奖励
        Buffer职责：只提供原始数据，不做任何计算

        Args:
            start_idx: Starting index

        Returns:
            Reward sequence of shape (horizon, 1)
        """
        current_size = min(self.size, self.max_size)
        rewards = []

        for i in range(self.horizon):
            idx = (start_idx + i) % current_size

            # 检查episode边界
            if i > 0:
                prev_idx = (start_idx + i - 1) % current_size
                if self.episode_ends[prev_idx] or self.episode_starts[idx]:
                    # episode结束，用0填充剩余位置
                    rewards.append(np.array([0.0]))
                    continue

            rewards.append(self.reward[idx])

        return np.stack(rewards)

    def _get_done_sequence(self, start_idx: int) -> np.ndarray:
        """
        获取从start_idx开始的done序列，供算法判断episode边界
        Buffer职责：只提供原始数据

        Args:
            start_idx: Starting index

        Returns:
            Done sequence of shape (horizon, 1)
        """
        current_size = min(self.size, self.max_size)
        dones = []

        for i in range(self.horizon):
            idx = (start_idx + i) % current_size

            # 检查episode边界
            if i > 0:
                prev_idx = (start_idx + i - 1) % current_size
                if self.episode_ends[prev_idx] or self.episode_starts[idx]:
                    # episode结束，标记为done
                    dones.append(np.array([1.0]))
                    continue

            dones.append(self.done[idx])

        return np.stack(dones)

    def sample_sequential(self, batch_size: int, sequence_length: int):
        """
        Sample sequential trajectories for diffusion model training.
        确保序列不跨越episode边界

        Args:
            batch_size: Number of sequences to sample
            sequence_length: Length of each sequence

        Returns:
            Tuple of (states, actions, rewards, dones) sequences
        """
        current_size = min(self.size, self.max_size)
        if current_size < sequence_length:
            raise ValueError(f"Buffer size {current_size} is smaller than sequence length {sequence_length}")

        sequences = {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': []
        }

        attempts = 0
        max_attempts = batch_size * 10  # 防止无限循环

        while len(sequences['states']) < batch_size and attempts < max_attempts:
            attempts += 1

            # 随机选择起始点
            start_idx = np.random.randint(0, current_size)

            # 检查从这个起始点开始的序列是否跨越episode边界
            valid_sequence = True
            seq_indices = []

            for i in range(sequence_length):
                idx = (start_idx + i) % current_size
                seq_indices.append(idx)

                # 检查是否跨���episode边界
                if i > 0:
                    prev_idx = (start_idx + i - 1) % current_size
                    if self.episode_ends[prev_idx] or self.episode_starts[idx]:
                        valid_sequence = False
                        break

            if valid_sequence:
                sequences['states'].append(self.state[seq_indices])
                sequences['actions'].append(self.action[seq_indices])
                sequences['rewards'].append(self.reward[seq_indices])
                sequences['dones'].append(self.done[seq_indices])

        if len(sequences['states']) == 0:
            raise ValueError("Could not find any valid sequential samples")

        return (
            torch.FloatTensor(np.stack(sequences['states'])).to(self.device),
            torch.FloatTensor(np.stack(sequences['actions'])).to(self.device),
            torch.FloatTensor(np.stack(sequences['rewards'])).to(self.device),
            torch.FloatTensor(np.stack(sequences['dones'])).to(self.device)
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

        # 初始化episode边界信息
        self.episode_starts = np.zeros(self.max_size, dtype=np.bool_)
        self.episode_ends = np.zeros(self.max_size, dtype=np.bool_)

        # 设置episode边界
        self.episode_starts[0] = True  # 第一个样本是episode��始
        for i in range(min(self.size - 1, self.max_size - 1)):
            if self.done[i, 0]:
                self.episode_ends[i] = True
                if i + 1 < self.size:
                    self.episode_starts[i + 1] = True

        # 确保数组大小不超过max_size
        if self.state.shape[0] > self.max_size:
            self.state = self.state[:self.max_size]
            self.action = self.action[:self.max_size]
            self.next_state = self.next_state[:self.max_size]
            self.reward = self.reward[:self.max_size]
            self.done = self.done[:self.max_size]
            self.episode_starts = self.episode_starts[:self.max_size]
            self.episode_ends = self.episode_ends[:self.max_size]

    def convert_minari(self, dataset):
        """
        Load data from a Minari dataset with optimized performance.
        支持gymnasium格式的terminated和truncated

        Args:
            dataset: Minari dataset object
        """
        # 预计算总数据量
        total_steps = sum(len(ep.observations) - 1 for ep in dataset.iterate_episodes())
        total_steps = min(total_steps, self.max_size)

        # 预分配内存
        self.state = np.empty((self.max_size, *dataset.observation_space.shape), dtype=np.float32)
        self.action = np.empty((self.max_size, *dataset.action_space.shape), dtype=np.float32)
        self.next_state = np.empty((self.max_size, *dataset.observation_space.shape), dtype=np.float32)
        self.reward = np.empty((self.max_size, 1), dtype=np.float32)
        self.done = np.empty((self.max_size, 1), dtype=np.float32)  # 统一使用float32
        self.episode_starts = np.zeros(self.max_size, dtype=np.bool_)
        self.episode_ends = np.zeros(self.max_size, dtype=np.bool_)

        index = 0
        for episode in dataset.iterate_episodes():
            obs = episode.observations
            actions = episode.actions
            rewards = episode.rewards
            terminations = episode.terminations
            truncations = episode.truncations

            ep_len = len(obs) - 1
            # 将terminated和truncated合并为done (gymnasium格式支持)
            next_state_dones = np.logical_or(terminations[1:], truncations[1:])
            
            if index + ep_len > self.max_size:
                ep_len = self.max_size - index
                if ep_len <= 0:
                    break

            # 批量填充数据（向量化操作）
            self.state[index:index + ep_len] = obs[:-1][:ep_len]
            self.action[index:index + ep_len] = actions[:ep_len]
            self.next_state[index:index + ep_len] = obs[1:][:ep_len]
            self.reward[index:index + ep_len] = rewards[:ep_len].reshape(-1, 1)
            done_flags = next_state_dones[:ep_len].reshape(-1, 1).astype(np.float32)
            actual_len = min(ep_len, len(done_flags))
            self.done[index:index + actual_len] = done_flags[:actual_len]

            # 设置episode边界
            self.episode_starts[index] = True
            self.episode_ends[index + actual_len - 1] = True

            index += ep_len
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
        if self.size == 0:
            raise ValueError("Cannot normalize states from empty buffer")

        # 只对有效数据计算统计量
        valid_states = self.state[:self.size]
        valid_next_states = self.next_state[:self.size]

        mean = valid_states.mean(0, keepdims=True)
        std = valid_states.std(0, keepdims=True) + eps

        # 标准化所有状态
        self.state[:self.size] = (valid_states - mean) / std
        self.next_state[:self.size] = (valid_next_states - mean) / std

        return mean, std

    def get_statistics(self):
        """
        获取buffer的统计信息，用于调试

        Returns:
            Dictionary with buffer statistics
        """
        if self.size == 0:
            return {"size": 0, "episodes": 0}

        valid_data = self.size
        episode_starts_count = np.sum(self.episode_starts[:valid_data])
        episode_ends_count = np.sum(self.episode_ends[:valid_data])

        return {
            "size": valid_data,
            "episodes": episode_starts_count,
            "episode_starts": episode_starts_count,
            "episode_ends": episode_ends_count,
            "horizon": self.horizon,
            "done_ratio": np.mean(self.done[:valid_data])
        }
