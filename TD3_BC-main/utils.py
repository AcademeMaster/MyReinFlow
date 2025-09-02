import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class EpisodeDataset(Dataset):
    def __init__(self, episodes):
        self.episodes = episodes

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        return self.episodes[idx]


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
        indices = np.random.randint(0, self.size, size=batch_size)

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
        self.size = self.state.shape[0]

    def convert_minari(self, dataset):
        """
        Load data from a Minari dataset.
        
        Args:
            dataset: Minari dataset object
        """
        # Create a temporary dataset class for the episodes

        
        def collate_fn(batch):
            """
            Convert episode data to training format.
            
            Args:
                batch: List of episodes
                
            Returns:
                Dictionary with processed data
            """
            # Pre-calculate total transitions for efficiency
            total_transitions = sum(max(0, len(episode.observations) - 1) for episode in batch)
            
            # Pre-allocate arrays
            all_observations = np.empty((total_transitions, ) + batch[0].observations.shape[1:], 
                                      dtype=np.float32)
            all_actions = np.empty((total_transitions, ) + batch[0].actions.shape[1:], 
                                 dtype=np.float32)
            all_next_observations = np.empty((total_transitions, ) + batch[0].observations.shape[1:], 
                                           dtype=np.float32)
            all_rewards = np.empty((total_transitions, ), dtype=np.float32)
            all_dones = np.empty((total_transitions, ), dtype=np.float32)
            
            idx = 0
            for episode in batch:
                observations = episode.observations
                actions = episode.actions
                rewards = episode.rewards
                terminations = episode.terminations if hasattr(episode, 'terminations') else np.zeros(len(observations), dtype=bool)
                truncations = episode.truncations if hasattr(episode, 'truncations') else np.zeros(len(observations), dtype=bool)

                # Process each transition (s_t, a_t, s_{t+1}, r_t, done_t)
                episode_length = len(observations) - 1
                for i in range(episode_length):
                    all_observations[idx] = observations[i]
                    all_actions[idx] = actions[i]
                    all_next_observations[idx] = observations[i + 1]
                    all_rewards[idx] = rewards[i]

                    # Calculate done flag (1 for terminal, 0 otherwise)
                    done = (terminations[i] if i < len(terminations) else False) or \
                           (truncations[i] if i < len(truncations) else False)
                    all_dones[idx] = float(done)
                    idx += 1

            return {
                "observations": all_observations[:idx],
                "actions": all_actions[:idx],
                "next_observations": all_next_observations[:idx],
                "rewards": all_rewards[:idx],
                "dones": all_dones[:idx]
            }

        # Use DataLoader to process dataset in batches to save memory
        episodes = list(dataset.iterate_episodes())
        batch_size = min(1000, len(episodes))  # Limit batch size to avoid memory issues
        
        # Wrap episodes in a Dataset and then use DataLoader
        episode_dataset = EpisodeDataset(episodes)
        dataloader = DataLoader(episode_dataset, batch_size=batch_size, collate_fn=collate_fn)

        # Process data in batches to save memory
        all_states = []
        all_actions = []
        all_next_states = []
        all_rewards = []
        all_dones = []
        
        for batch in dataloader:
            all_states.append(batch["observations"])
            all_actions.append(batch["actions"])
            all_next_states.append(batch["next_observations"])
            all_rewards.append(batch["rewards"])
            all_dones.append(batch["dones"])

        # Concatenate all batch data
        self.state = np.concatenate(all_states, axis=0)
        self.action = np.concatenate(all_actions, axis=0)
        self.next_state = np.concatenate(all_next_states, axis=0)
        self.reward = np.concatenate(all_rewards, axis=0).reshape(-1, 1)
        self.done = np.concatenate(all_dones, axis=0).reshape(-1, 1)
        self.size = self.state.shape[0]

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