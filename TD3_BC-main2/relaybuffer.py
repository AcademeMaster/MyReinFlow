import torch
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
from torch.utils.data import Dataset, DataLoader
import copy


class TorchDataset(Dataset):
    """PyTorch Dataset class for reinforcement learning transitions."""

    def __init__(self, data: Dict[str, Union[np.ndarray, torch.Tensor]]):
        """
        Initialize the dataset.
        
        Args:
            data: Dictionary containing the dataset with keys like 'observations', 'actions', etc.
        """
        super().__init__()
        self.data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                self.data[key] = torch.from_numpy(value).float()
            elif isinstance(value, torch.Tensor):
                self.data[key] = value.float()
            else:
                raise TypeError(f"Unsupported data type: {type(value)}")
        
        self.size = len(self.data['observations'])
        self.frame_stack = None  # Number of frames to stack; set outside the class.
        self.p_aug = None  # Image augmentation probability; set outside the class.
        self.return_next_actions = False  # Whether to additionally return next actions; set outside the class.
        
        # Compute terminal and initial locations
        if 'terminals' in self.data:
            terminal_indices = torch.nonzero(self.data['terminals'] > 0).flatten()
            self.terminal_locs = terminal_indices.cpu().numpy()
            initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1]) if len(self.terminal_locs) > 0 else np.array([0])
            self.initial_locs = initial_locs
        else:
            self.terminal_locs = np.array([])
            self.initial_locs = np.array([0])

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset."""
        return {key: value[idx] for key, value in self.data.items()}

    def get_random_idxs(self, num_idxs: int) -> np.ndarray:
        """Return `num_idxs` random indices."""
        return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size: int, idxs: Optional[np.ndarray] = None) -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        
        batch = self.get_subset(idxs)
        
        if self.frame_stack is not None:
            # Stack frames
            initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
            obs_list = []  # Will be [ob[t - frame_stack + 1], ..., ob[t]]
            next_obs_list = []  # Will be [ob[t - frame_stack + 2], ..., ob[t], next_ob[t]]
            
            for i in reversed(range(self.frame_stack)):
                # Use the initial state if the index is out of bounds
                cur_idxs = np.maximum(idxs - i, initial_state_idxs)
                obs_list.append(self.data['observations'][cur_idxs])
                if i != self.frame_stack - 1:
                    next_obs_list.append(self.data['observations'][cur_idxs])
            
            next_obs_list.append(self.data['next_observations'][idxs])
            
            # Concatenate observations along the last dimension
            batch['observations'] = torch.cat(obs_list, dim=-1)
            batch['next_observations'] = torch.cat(next_obs_list, dim=-1)
        
        if self.p_aug is not None and np.random.rand() < self.p_aug:
            # Apply random-crop image augmentation
            batch = self.augment(batch, ['observations', 'next_observations'])
        
        return batch

    def sample_sequence(self, batch_size: int, sequence_length: int, discount: float) -> Dict[str, torch.Tensor]:
        """Sample a sequence of transitions."""
        idxs = np.random.randint(self.size - sequence_length + 1, size=batch_size)
        
        data = {k: v[idxs] for k, v in self.data.items()}
        
        # Initialize arrays for sequences
        rewards = torch.zeros((*data['rewards'].shape, sequence_length), dtype=torch.float32)
        masks = torch.ones((*data['masks'].shape, sequence_length), dtype=torch.float32)
        valid = torch.ones((*data['masks'].shape, sequence_length), dtype=torch.float32)
        observations = torch.zeros((*data['observations'].shape[:-1], sequence_length, data['observations'].shape[-1]), 
                                   dtype=torch.float32)
        next_observations = torch.zeros((*data['observations'].shape[:-1], sequence_length, data['observations'].shape[-1]), 
                                        dtype=torch.float32)
        actions = torch.zeros((*data['actions'].shape[:-1], sequence_length, data['actions'].shape[-1]), 
                              dtype=torch.float32)
        terminals = torch.zeros((*data['terminals'].shape, sequence_length), dtype=torch.float32)
        next_actions = torch.zeros((*data['actions'].shape[:-1], sequence_length, data['actions'].shape[-1]), 
                                   dtype=torch.float32)
        
        # Fill in each of the sequence_length dimensions
        for i in range(sequence_length):
            cur_idxs = idxs + i
            
            if i == 0:
                rewards[..., 0] = self.data['rewards'][cur_idxs]
                masks[..., 0] = self.data['masks'][cur_idxs]
                terminals[..., 0] = self.data['terminals'][cur_idxs]
            else:
                rewards[..., i] = rewards[..., i - 1] + self.data['rewards'][cur_idxs] * (discount ** i)
                masks[..., i] = torch.minimum(masks[..., i - 1], self.data['masks'][cur_idxs])
                terminals[..., i] = torch.maximum(terminals[..., i - 1], self.data['terminals'][cur_idxs])
                valid[..., i] = (1.0 - terminals[..., i - 1])
            
            actions[..., i, :] = self.data['actions'][cur_idxs]
            next_observations[..., i, :] = self.data['next_observations'][cur_idxs]
            observations[..., i, :] = self.data['observations'][cur_idxs]
            next_actions[..., i, :] = self.data['actions'][torch.minimum(
                torch.from_numpy(cur_idxs + 1), torch.tensor(self.size - 1))]
        
        return {
            'observations': data['observations'].clone(),
            'full_observations': observations,
            'actions': actions,
            'masks': masks,
            'rewards': rewards,
            'terminals': terminals,
            'valid': valid,
            'next_observations': next_observations,
            'next_actions': next_actions,
        }

    def get_subset(self, idxs: np.ndarray) -> Dict[str, torch.Tensor]:
        """Return a subset of the dataset given the indices."""
        result = {key: value[idxs] for key, value in self.data.items()}
        if self.return_next_actions:
            # WARNING: This is incorrect at the end of the trajectory. Use with caution.
            next_action_idxs = np.minimum(idxs + 1, self.size - 1)
            result['next_actions'] = self.data['actions'][next_action_idxs]
        return result

    def augment(self, batch: Dict[str, torch.Tensor], keys: List[str]) -> Dict[str, torch.Tensor]:
        """Apply image augmentation to the given keys."""
        # Simple random crop augmentation for demonstration
        # In practice, you might want to use more sophisticated augmentation techniques
        padding = 3
        batch_size = len(batch[keys[0]])
        
        for key in keys:
            if key in batch and len(batch[key].shape) == 4:  # Image data (B, H, W, C)
                # Apply random crop
                h, w = batch[key].shape[1], batch[key].shape[2]
                crop_h, crop_w = h - 2 * padding, w - 2 * padding
                
                # Random crop positions
                top = torch.randint(0, 2 * padding + 1, (batch_size,))
                left = torch.randint(0, 2 * padding + 1, (batch_size,))
                
                cropped = []
                for i in range(batch_size):
                    cropped.append(batch[key][i, top[i]:top[i]+crop_h, left[i]:left[i]+crop_w, :])
                batch[key] = torch.stack(cropped, dim=0)
        
        return batch


class ReplayBuffer(TorchDataset):
    """Replay buffer class.
    
    This class extends TorchDataset to support adding transitions.
    """

    @classmethod
    def create(cls, transition: Dict[str, Any], size: int):
        """Create a replay buffer from the example transition.
        
        Args:
            transition: Example transition (dict).
            size: Size of the replay buffer.
        """
        def create_buffer(example):
            if isinstance(example, np.ndarray):
                return np.zeros((size, *example.shape), dtype=example.dtype)
            elif isinstance(example, torch.Tensor):
                return torch.zeros((size, *example.shape), dtype=example.dtype)
            else:
                raise TypeError(f"Unsupported transition element type: {type(example)}")
        
        buffer_dict = {key: create_buffer(value) for key, value in transition.items()}
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset: Dict[str, Any], size: int):
        """Create a replay buffer from the initial dataset.
        
        Args:
            init_dataset: Initial dataset.
            size: Size of the replay buffer.
        """
        def create_buffer(init_buffer):
            if isinstance(init_buffer, np.ndarray):
                buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
                buffer[:len(init_buffer)] = init_buffer[:size] if len(init_buffer) > size else init_buffer
                return buffer
            elif isinstance(init_buffer, torch.Tensor):
                buffer = torch.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
                buffer[:len(init_buffer)] = init_buffer[:size] if len(init_buffer) > size else init_buffer
                return buffer
            else:
                raise TypeError(f"Unsupported dataset element type: {type(init_buffer)}")
        
        buffer_dict = {key: create_buffer(value) for key, value in init_dataset.items()}
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = len(init_dataset['observations'])
        return dataset

    def __init__(self, data: Dict[str, Union[np.ndarray, torch.Tensor]]):
        super().__init__(data)
        self.max_size = len(self.data['observations'])
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition: Dict[str, Any]):
        """Add a transition to the replay buffer."""
        for key, value in transition.items():
            if key in self.data:
                if isinstance(value, np.ndarray):
                    self.data[key][self.pointer] = torch.from_numpy(value)
                elif isinstance(value, torch.Tensor):
                    self.data[key][self.pointer] = value
                else:
                    self.data[key][self.pointer] = torch.tensor(value)
        
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        """Clear the replay buffer."""
        self.size = self.pointer = 0


def add_history(dataset: TorchDataset, history_length: int) -> TorchDataset:
    """Add history to the dataset."""
    size = dataset.size
    
    if 'terminals' in dataset.data:
        terminal_locs = torch.nonzero(dataset.data['terminals'] > 0).flatten().cpu().numpy()
        initial_locs = np.concatenate([[0], terminal_locs[:-1] + 1]) if len(terminal_locs) > 0 else np.array([0])
        assert terminal_locs[-1] == size - 1 if len(terminal_locs) > 0 else True
    
    idxs = np.arange(size)
    initial_state_idxs = initial_locs[np.searchsorted(initial_locs, idxs, side='right') - 1]
    
    obs_history_list = []
    act_history_list = []
    
    for i in reversed(range(1, history_length)):
        cur_idxs = np.maximum(idxs - i, initial_state_idxs)
        outside = (idxs - i < initial_state_idxs)[..., None]
        
        # Process observations
        obs_cur = dataset.data['observations'][cur_idxs]
        obs_masked = obs_cur * (~outside) if isinstance(obs_cur, torch.Tensor) else obs_cur * (~outside)
        obs_history_list.append(obs_masked)
        
        # Process actions
        act_cur = dataset.data['actions'][cur_idxs]
        act_masked = act_cur * (~outside) if isinstance(act_cur, torch.Tensor) else act_cur * (~outside)
        act_history_list.append(act_masked)
    
    # Stack history along new dimension
    observation_history = torch.stack(obs_history_list, dim=-2)
    action_history = torch.stack(act_history_list, dim=-2)
    
    # Create new dataset with history
    new_data = copy.deepcopy(dataset.data)
    new_data['observation_history'] = observation_history
    new_data['action_history'] = action_history
    
    return TorchDataset(new_data)