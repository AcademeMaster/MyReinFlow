"""
Datasets for stitched state-action sequences, with optional Q-learning signals.

This module provides:
  - Batch, Transition, TransitionWithReturn namedtuples
  - StitchedSequenceDataset: loads states/actions/(images) from .npz or .pkl with
    a 1-D traj_lengths array that concatenates episodes end-to-end.
  - StitchedSequenceQLearningDataset: extends the above with rewards and dones,
    supports skipping truncated last steps and optional Monte Carlo returns.

Expected keys in the dataset file:
  - "traj_lengths": 1-D array of per-episode lengths
  - "states":      [sum(T_i), obs_dim]
  - "actions":     [sum(T_i), action_dim]
  - optional "images":  [sum(T_i), C, H, W]
  - optional (for Q-learning): "rewards" [sum(T_i)], "terminals" [sum(T_i)]
"""

from __future__ import annotations

import logging
import random
import pickle
from collections import namedtuple
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

log = logging.getLogger(__name__)

# Simple containers compatible with training loops
Batch = namedtuple("Batch", ["actions", "conditions"])  # actions: Tensor [Ta, Da] or [B, Ta, Da]
Transition = namedtuple("Transition", ["actions", "conditions", "rewards", "dones"])  # per-timestep
TransitionWithReturn = namedtuple(
    "TransitionWithReturn",
    ["actions", "conditions", "rewards", "dones", "reward_to_gos"],
)


class StitchedSequenceDataset(torch.utils.data.Dataset):
    """
    Load stitched trajectories of states/actions/images, and 1-D array of traj_lengths, from .npz or .pkl.

    Sampling uses the first max_n_episodes episodes in order (no random episode selection).

    Each sample returns a Batch(actions, conditions) where:
      - actions: Tensor [horizon_steps, action_dim]
      - conditions: dict with keys:
          'state': Tensor [cond_steps, obs_dim]
          optionally 'rgb': Tensor [img_cond_steps, C, H, W]
    """

    def __init__(
        self,
        dataset_path: str,
        horizon_steps: int = 64,
        cond_steps: int = 1,
        img_cond_steps: int = 1,
        max_n_episodes: int = -1,
        use_img: bool = False,
        device: str | torch.device = "cuda:0",
    ) -> None:
        assert img_cond_steps <= cond_steps, "consider using more cond_steps than img_cond_steps"

        self.horizon_steps = int(horizon_steps)
        self.cond_steps = int(cond_steps)
        self.img_cond_steps = int(img_cond_steps)
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.use_img = bool(use_img)
        self.max_n_episodes = int(max_n_episodes)
        self.dataset_path = dataset_path

        # Load dataset file
        if dataset_path.endswith(".npz"):
            dataset = np.load(dataset_path, allow_pickle=False)  # only np arrays
        elif dataset_path.endswith(".pkl"):
            with open(dataset_path, "rb") as f:
                dataset = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")

        if self.max_n_episodes == -1:
            self.max_n_episodes = int(len(dataset["traj_lengths"]))
            log.info(
                f"max_n_episodes specified as -1, fall back to maximum value {self.max_n_episodes}"
            )
        traj_lengths = np.asarray(dataset["traj_lengths"])[: self.max_n_episodes]
        if traj_lengths.ndim != 1:
            raise ValueError("traj_lengths must be a 1-D array")
        total_num_steps = int(np.sum(traj_lengths))
        # Set up indices for sampling
        self.indices = self.make_indices(traj_lengths.tolist(), self.horizon_steps)

        # Extract states and actions up to max_n_episodes
        self.states = torch.from_numpy(np.asarray(dataset["states"])[:total_num_steps]).float().to(self.device)
        self.actions = torch.from_numpy(np.asarray(dataset["actions"])[:total_num_steps]).float().to(self.device)

        log.info(f"Successfully loaded dataset from {dataset_path}")
        n_eps = min(self.max_n_episodes, len(traj_lengths))
        if n_eps <= 0:
            raise ValueError("number of episodes less than 1, check dataset content")
        log.info(f"Number of episodes: {n_eps}")
        log.info(f"States shape/type: {self.states.shape, self.states.dtype}")
        log.info(f"Actions shape/type: {self.actions.shape, self.actions.dtype}")
        if self.use_img:
            self.images = torch.from_numpy(np.asarray(dataset["images"])[:total_num_steps]).to(self.device)
            log.info(f"Images shape/type: {self.images.shape, self.images.dtype}")
        log.info(f"Finished creating {self.__class__.__name__} from {dataset_path}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> object:
        """
        Returns a window ending at `start` with history padding at the beginning of an episode.
        """
        start, num_before_start = self.indices[idx]
        end = start + self.horizon_steps

        # history states up to current step (inclusive)
        states = self.states[(start - num_before_start) : (start + 1)]
        actions = self.actions[start:end]

        # stack last cond_steps observations, repeat earliest when insufficient
        states = torch.stack(
            [states[max(num_before_start - t, 0)] for t in reversed(range(self.cond_steps))]
        )
        conditions: Dict[str, torch.Tensor] = {"state": states}
        if self.use_img:
            images = self.images[(start - num_before_start) : end]
            images = torch.stack(
                [images[max(num_before_start - t, 0)] for t in reversed(range(self.img_cond_steps))]
            )
            conditions["rgb"] = images
        return Batch(actions, conditions)

    def make_indices(self, traj_lengths: Sequence[int], horizon_steps: int) -> List[Tuple[int, int]]:
        """
        Build index list for sampling.
        Each entry: (global_step_index, steps_before_within_same_traj).
        """
        indices: List[Tuple[int, int]] = []
        cur_traj_index = 0
        for traj_length in traj_lengths:
            traj_length = int(traj_length)
            max_start = cur_traj_index + traj_length - horizon_steps
            indices += [(i, i - cur_traj_index) for i in range(cur_traj_index, max_start + 1)]
            cur_traj_index += traj_length
        return indices

    def set_train_val_split(self, train_split: float) -> List[int]:
        """Randomly split indices, keep train indices inside self, return val indices (as integer positions)."""
        num_train = int(len(self.indices) * float(train_split))
        # sample actual index tuples, then map back to positions
        train_indices = random.sample(self.indices, num_train)
        train_set = set(train_indices)
        val_positions = [i for i in range(len(self.indices)) if self.indices[i] not in train_set]
        self.indices = train_indices
        return val_positions


class StitchedSequenceQLearningDataset(StitchedSequenceDataset):
    """
    Extends StitchedSequenceDataset to include rewards and dones for Q-learning.

    Returns Transition(actions, conditions, rewards, dones) per item.
    Skips last step of truncated episodes (terminal=False at final step) to ensure next state exists.
    """

    def __init__(
        self,
        dataset_path: str,
        max_n_episodes: int = 10000,
        discount_factor: float = 1.0,
        device: str | torch.device = "cuda:0",
        get_mc_return: bool = False,
        **kwargs,
    ) -> None:
        # peek to get lengths and signals first
        if dataset_path.endswith(".npz"):
            dataset = np.load(dataset_path, allow_pickle=False)
        elif dataset_path.endswith(".pkl"):
            with open(dataset_path, "rb") as f:
                dataset = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")

        traj_lengths = np.asarray(dataset["traj_lengths"])[: int(max_n_episodes)]
        total_num_steps = int(np.sum(traj_lengths))

        self.discount_factor = float(discount_factor)
        self.device = torch.device(device) if not isinstance(device, torch.device) else device

        # rewards and dones (terminals)
        self.rewards = torch.from_numpy(np.asarray(dataset["rewards"])[:total_num_steps]).float().to(self.device)
        log.info(f"Rewards shape/type: {self.rewards.shape, self.rewards.dtype}")
        self.dones = torch.from_numpy(np.asarray(dataset["terminals"])[:total_num_steps]).float().to(self.device)
        log.info(f"Dones shape/type: {self.dones.shape, self.dones.dtype}")

        # now call parent to load states/actions and build indices (overridden below)
        super().__init__(
            dataset_path=dataset_path,
            max_n_episodes=int(max_n_episodes),
            device=self.device,
            **kwargs,
        )
        log.info(f"Total number of transitions using: {len(self)}")

        # optional discounted reward-to-go per trajectory
        self.get_mc_return = bool(get_mc_return)
        if self.get_mc_return:
            self.reward_to_go = torch.zeros_like(self.rewards)
            cumulative = np.cumsum(traj_lengths)
            prev = 0
            for traj_end in tqdm(cumulative, desc="Computing reward-to-go"):
                traj_rewards = self.rewards[prev:traj_end]
                returns = torch.zeros_like(traj_rewards)
                prev_return = torch.tensor(0.0, device=self.rewards.device)
                for t in range(len(traj_rewards)):
                    returns[-t - 1] = traj_rewards[-t - 1] + self.discount_factor * prev_return
                    prev_return = returns[-t - 1]
                self.reward_to_go[prev:traj_end] = returns
                prev = int(traj_end)
            log.info("Computed reward-to-go for each trajectory.")

    def make_indices(self, traj_lengths: Sequence[int], horizon_steps: int) -> List[Tuple[int, int]]:
        """Skip last step of truncated episodes (terminal=False on final step)."""
        num_skip = 0
        indices: List[Tuple[int, int]] = []
        cur_traj_index = 0
        for traj_length in traj_lengths:
            traj_length = int(traj_length)
            max_start = cur_traj_index + traj_length - horizon_steps
            # if final step is not terminal => truncation; skip last possible start
            if not bool(self.dones[cur_traj_index + traj_length - 1]):
                max_start -= 1
                num_skip += 1
            indices += [(i, i - cur_traj_index) for i in range(cur_traj_index, max_start + 1)]
            cur_traj_index += traj_length
        log.info(f"Number of transitions skipped due to truncation: {num_skip}")
        return indices

    def __getitem__(self, idx: int) -> object:
        start, num_before_start = self.indices[idx]
        end = start + self.horizon_steps

        states = self.states[(start - num_before_start) : (start + 1)]
        actions = self.actions[start:end]
        rewards = self.rewards[start : (start + 1)]
        dones = self.dones[start : (start + 1)]

        # next states for action horizon; if cross-episode it is fine since dones prevents bootstrapping
        if idx < len(self.indices) - self.horizon_steps:
            next_states = self.states[(start - num_before_start + self.horizon_steps) : (start + 1 + self.horizon_steps)]
        else:
            next_states = torch.zeros_like(states)

        # stack obs history
        states = torch.stack(
            [states[max(num_before_start - t, 0)] for t in reversed(range(self.cond_steps))]
        )
        next_states = torch.stack(
            [next_states[max(num_before_start - t, 0)] for t in reversed(range(self.cond_steps))]
        )

        conditions: Dict[str, torch.Tensor] = {"state": states, "next_state": next_states}
        if self.use_img:
            images = self.images[(start - num_before_start) : end]
            images = torch.stack(
                [images[max(num_before_start - t, 0)] for t in reversed(range(self.img_cond_steps))]
            )
            conditions["rgb"] = images

        if self.get_mc_return:
            reward_to_gos = self.reward_to_go[start : (start + 1)]
            return TransitionWithReturn(actions, conditions, rewards, dones, reward_to_gos)
        else:
            return Transition(actions, conditions, rewards, dones)
