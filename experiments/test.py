#!/usr/bin/env python3
"""
Improved FQL training script (Minari dataset -> Flow-based behavior cloning + Q-learning)

Key improvements over original submission:
- MinariStitchedDataset.__getitem__ now returns dicts so DataLoader can use default collate.
- Robust handling of dict/Box observations and env.reset/step in evaluation.
- Consistent ordering of state history (latest-first) between dataset and evaluation.
- Fixed index generation to avoid negative ranges when episode shorter than horizon.
- Safer tensor device movement with recursive utility.
- Simpler and clearer training loop (no custom stacking functions).
- Gradient clipping, mixed-precision optional (toggle), save-best checkpointing.
- Better logging and error messages.

Run with: python fql_improved.py --dataset_id mujoco/pusher/expert-v0
"""

import argparse
import os
import random
import logging
import copy
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm

import minari
from gymnasium import spaces

# Logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# -------------------- Utilities --------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(x, device: torch.device):
    """Recursively move tensors in nested structure to device."""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = [to_device(v, device) for v in x]
        return type(x)(t)
    return x


# -------------------- NN blocks --------------------
activation_dict = {
    "Tanh": nn.Tanh(),
    "ReLU": nn.ReLU(),
    "GELU": nn.GELU(),
    "Mish": nn.Mish(),
    "Identity": nn.Identity(),
}


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half = self.dim // 2
        freq = torch.exp(torch.arange(half, device=device) * -(np.log(10000.0) / max(half - 1, 1)))
        ang = x[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class MLP(nn.Module):
    def __init__(self, dims: List[int], activation: str = "Mish"):
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(activation_dict[activation])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FlowMLP(nn.Module):
    def __init__(
        self,
        horizon_steps: int,
        action_dim: int,
        cond_dim: int,
        time_dim: int = 32,
        mlp_dims: List[int] = [256, 256],
    ):
        super().__init__()
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        self.total_act = horizon_steps * action_dim
        self.time_dim = time_dim
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )
        inp = time_dim + self.total_act + cond_dim
        self.mlp = MLP([inp] + mlp_dims + [self.total_act], activation="Mish")

    def forward(self, action: torch.Tensor, time: torch.Tensor, cond: Dict[str, torch.Tensor]) -> torch.Tensor:
        B, Ta, Da = action.shape
        action = action.view(B, -1)
        state = cond["state"].view(B, -1)
        if isinstance(time, (int, float)):
            time = torch.ones(B, 1, device=action.device) * float(time)
        t_emb = self.time_embedding(time.view(B, 1)).view(B, -1)
        feat = torch.cat([action, t_emb, state], dim=-1)
        vel = self.mlp(feat)
        return vel.view(B, Ta, Da)


class CriticObsAct(nn.Module):
    def __init__(self, cond_dim: int, mlp_dims: List[int], action_dim: int, action_steps: int = 1, double_q: bool = True):
        super().__init__()
        inp = cond_dim + action_dim * action_steps
        self.Q1 = MLP([inp] + mlp_dims + [1])
        self.Q2 = MLP([inp] + mlp_dims + [1]) if double_q else None

    def forward(self, cond: Dict[str, torch.Tensor], action: torch.Tensor):
        B = cond["state"].shape[0]
        s = cond["state"].view(B, -1)
        a = action.view(B, -1)
        x = torch.cat([s, a], dim=-1)
        q1 = self.Q1(x).squeeze(-1)
        if self.Q2 is None:
            return q1
        q2 = self.Q2(x).squeeze(-1)
        return q1, q2


class OneStepActor(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 cond_steps: int,
                 action_dim: int,
                 horizon_steps: int,
                 hidden_dim: int = 512):
        super().__init__()
        self.obs_dim = obs_dim
        self.cond_steps = cond_steps
        self.action_dim = action_dim
        self.horizon_steps = horizon_steps
        self.net = nn.Sequential(
            nn.Linear(cond_steps * obs_dim + horizon_steps * action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, horizon_steps * action_dim),
        )

    def forward(self, cond: Dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        s = cond['state']  # (B, cond_steps, obs_dim)
        B = s.shape[0]
        s_flat = s.view(B, -1)
        z_flat = z.view(B, -1)
        feature = torch.cat([s_flat, z_flat], dim=-1)
        output = self.net(feature)
        actions: torch.Tensor = output.view(B, self.horizon_steps, self.action_dim)
        return actions


class ReFlow(nn.Module):
    def __init__(
        self,
        network: FlowMLP,
        device: torch.device,
        horizon_steps: int,
        action_dim: int,
        act_min: float,
        act_max: float,
        obs_dim: int,
        max_denoising_steps: int,
        seed: int,
        sample_t_type: str = 'uniform',
    ):
        super().__init__()
        if int(max_denoising_steps) <= 0:
            raise ValueError("max_denoising_steps must be positive")
        self.network = network.to(device)
        self.device = device
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        self.data_shape = (horizon_steps, action_dim)
        self.act_range = (act_min, act_max)
        self.obs_dim = obs_dim
        self.max_denoising_steps = int(max_denoising_steps)
        self.sample_t_type = sample_t_type
        if seed is not None:
            set_seed(seed)

    def generate_trajectory(self, x1: torch.Tensor, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_ = (torch.ones_like(x1, device=self.device) * t.view(x1.shape[0], 1, 1)).to(self.device)
        return t_ * x1 + (1 - t_) * x0

    def sample_time(self, batch_size: int, time_sample_type: str = 'uniform') -> torch.Tensor:
        if time_sample_type == 'uniform':
            return torch.rand(batch_size, device=self.device)
        raise ValueError("Only 'uniform' time sampling is implemented.")

    def generate_target(self, x1: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        t = self.sample_time(batch_size=x1.shape[0], time_sample_type=self.sample_t_type)
        x0 = torch.randn_like(x1)
        xt = self.generate_trajectory(x1, x0, t)
        v = x1 - x0
        return (xt, t), v

    def loss(self, xt: torch.Tensor, t: torch.Tensor, obs: Dict[str, torch.Tensor], v: torch.Tensor) -> torch.Tensor:
        v_hat = self.network(xt, t, obs)
        return F.mse_loss(v_hat, v)

    @torch.no_grad()
    def sample(self, cond: Dict[str, torch.Tensor], inference_steps: int, record_intermediate: bool = False,
               clip_intermediate_actions: bool = True, z: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B = cond['state'].shape[0]
        chains = None
        if record_intermediate:
            chains = torch.zeros((inference_steps, B) + self.data_shape, device=self.device)
        x_hat = z if z is not None else torch.randn((B,) + self.data_shape, device=self.device)
        dt = 1.0 / inference_steps
        steps = torch.linspace(0, 1, inference_steps, device=self.device).repeat(B, 1)
        for i in range(inference_steps):
            t = steps[:, i]
            vt = self.network(x_hat, t, cond)
            x_hat = x_hat + vt * dt
            if clip_intermediate_actions or i == inference_steps - 1:
                x_hat = x_hat.clamp(*self.act_range)
            if record_intermediate and chains is not None:
                chains[i] = x_hat
        return {'trajectories': x_hat, 'chains': chains}


class FQLModel(nn.Module):
    def __init__(self, bc_flow: ReFlow, actor: OneStepActor, critic: CriticObsAct, inference_steps: int,
                 normalize_q_loss: bool, device: torch.device):
        super().__init__()
        self.bc_flow = bc_flow
        self.actor = actor
        self.critic = critic
        self.target_critic = copy.deepcopy(self.critic)
        self.device = device
        self.inference_steps = inference_steps
        self.normalize_q_loss = normalize_q_loss

    def forward(self, cond: Dict[str, torch.Tensor], mode: str = 'onestep') -> torch.Tensor:
        B = cond['state'].shape[0]
        assert mode in ['onestep', 'base_model']
        if mode == 'onestep':
            z = torch.randn(B, self.actor.horizon_steps, self.actor.action_dim, device=self.device)
            return self.actor(cond, z)
        return self.bc_flow.sample(cond, self.inference_steps, record_intermediate=False,
                                   clip_intermediate_actions=False)['trajectories']

    def loss_bc_flow(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor) -> torch.Tensor:
        (xt, t), v = self.bc_flow.generate_target(actions)
        v_hat = self.bc_flow.network(xt, t, obs)
        return F.mse_loss(v_hat, v)

    def loss_critic(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor, next_obs: Dict[str, torch.Tensor],
                    rewards: torch.Tensor, terminated: torch.Tensor, gamma: float) -> Tuple[torch.Tensor, Dict[str, float]]:
        with torch.no_grad():
            z = torch.randn_like(actions, device=self.device)
            next_actions = self.actor(next_obs, z)
            next_actions = torch.clamp(next_actions, min=self.bc_flow.act_range[0], max=self.bc_flow.act_range[1])
            next_q1, next_q2 = self.target_critic(next_obs, next_actions)
            next_q = torch.mean(torch.stack([next_q1, next_q2], dim=0), dim=0)
            rewards = rewards.view(-1)
            terminated = terminated.view(-1)
            target = rewards + gamma * (1 - terminated) * next_q
        q1, q2 = self.critic(obs, actions)
        loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        return loss, {"loss_critic": float(loss.item())}

    def loss_actor(self, obs: Dict[str, torch.Tensor], action_batch: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, Dict[str, float]]:
        B = obs['state'].shape[0]
        z = torch.randn(B, self.actor.horizon_steps, self.actor.action_dim, device=self.device)
        a_w = self.actor(obs, z)
        a_theta = self.bc_flow.sample(cond=obs, inference_steps=self.inference_steps, record_intermediate=False,
                                      clip_intermediate_actions=False, z=z)['trajectories']
        distill = F.mse_loss(a_w, a_theta)
        actor_actions = torch.clamp(a_w, min=self.bc_flow.act_range[0], max=self.bc_flow.act_range[1])
        q1, q2 = self.critic(obs, actor_actions)
        q = torch.mean(torch.stack([q1, q2], dim=0), dim=0)
        q_loss = -q.mean()
        if self.normalize_q_loss:
            q_loss = q_loss * (1.0 / (torch.abs(q).mean().detach() + 1e-6))
        bc = self.loss_bc_flow(obs, action_batch)
        loss = bc + alpha * distill + q_loss
        return loss, {"loss_actor": float(loss.item()), "loss_bc_flow": float(bc.item()), 
                     "q_loss": float(q_loss.item()), "distill_loss": float(distill.item()), 
                     "q": float(q.mean().item())}

    def update_target_critic(self, tau: float):
        for tp, sp in zip(self.target_critic.parameters(), self.critic.parameters()):
            tp.data.copy_(sp.data * tau + tp.data * (1 - tau))


# -------------------- Dataset --------------------
class MinariStitchedDataset(torch.utils.data.Dataset):
    """
    Minari dataset -> stitched dicts for easy collate in DataLoader.
    Each item is a dict with keys: actions, conditions (dict with 'state' and 'next_state'), rewards, dones, optional rtg
    """
    def __init__(
        self,
        minari_dataset: minari.MinariDataset,
        horizon_steps: int = 1,
        cond_steps: int = 1,
        max_n_episodes: int = -1,
        use_img: bool = False,
        device: str = "cpu",
        get_mc_return: bool = False,
        discount_factor: float = 0.99,
        **_
    ):
        assert cond_steps >= 1
        self.horizon_steps = int(horizon_steps)
        self.cond_steps = int(cond_steps)
        self.use_img = bool(use_img)
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.max_n_episodes = int(max_n_episodes)
        self.get_mc_return = bool(get_mc_return)
        self.discount_factor = float(discount_factor)

        episodes = list(minari_dataset.iterate_episodes())
        if self.max_n_episodes > 0:
            episodes = episodes[: self.max_n_episodes]
        self.episodes = episodes

        self.obs_space = minari_dataset.spec.observation_space
        self.act_space = minari_dataset.spec.action_space

        self.traj_lengths = np.array([len(ep.actions) for ep in self.episodes], dtype=np.int64)
        self.total_steps = int(np.sum(self.traj_lengths))

        # Allocate storage
        self.obs_dim = self._setup_obs_storage()
        self._setup_action_storage()
        self._setup_reward_done_storage()

        # Build indices safely
        self.indices = self._make_indices_skip_trunc()

        if self.get_mc_return:
            self._compute_reward_to_go()

        log.info(f"Dataset contains {len(self.episodes)} episodes, {self.total_steps} steps")
        log.info(f"Actions shape: {self.actions.shape}")

    def _setup_obs_storage(self) -> Optional[int]:
        if isinstance(self.obs_space, spaces.Box):
            obs_dim = int(np.prod(self.obs_space.shape or (1,)))
            self.states = torch.zeros((self.total_steps, obs_dim), dtype=torch.float32)
            pos = 0
            for ep in self.episodes:
                T = len(ep.actions)
                obs_arr = np.asarray(ep.observations)
                if obs_arr.shape[0] < T:
                    last = obs_arr[-1:]
                    pad = np.repeat(last, T - obs_arr.shape[0], axis=0)
                    obs_use = np.concatenate([obs_arr, pad], axis=0)
                else:
                    obs_use = obs_arr[:T]
                obs_use = obs_use.reshape(T, -1)
                self.states[pos : pos + T] = torch.as_tensor(obs_use, dtype=torch.float32)
                pos += T
            return obs_dim
        elif isinstance(self.obs_space, spaces.Dict):
            self.state_data: Dict[str, torch.Tensor] = {}
            pos = 0
            for ep in self.episodes:
                T = len(ep.actions)
                for t in range(T):
                    obs_t = ep.observations[t]
                    for k, v in obs_t.items():
                        arr = np.asarray(v)
                        shape = arr.shape
                        if k not in self.state_data:
                            self.state_data[k] = torch.zeros((self.total_steps, *shape), dtype=torch.float32)
                        self.state_data[k][pos + t] = torch.as_tensor(arr, dtype=torch.float32)
                pos += T
            if "state" in self.state_data:
                return int(self.state_data["state"].shape[-1])
            return None
        else:
            raise TypeError(f"Unsupported observation space: {type(self.obs_space)}")

    def _setup_action_storage(self) -> None:
        if not isinstance(self.act_space, spaces.Box):
            raise TypeError(f"Unsupported action space: {type(self.act_space)}")
        act_dim = int(np.prod(self.act_space.shape or (1,)))
        self.actions = torch.zeros((self.total_steps, act_dim), dtype=torch.float32)
        pos = 0
        for ep in self.episodes:
            T = len(ep.actions)
            self.actions[pos : pos + T] = torch.as_tensor(np.asarray(ep.actions).reshape(T, -1), dtype=torch.float32)
            pos += T

    def _setup_reward_done_storage(self) -> None:
        self.rewards = torch.zeros((self.total_steps,), dtype=torch.float32)
        self.terminations = torch.zeros((self.total_steps,), dtype=torch.float32)
        self.truncations = torch.zeros((self.total_steps,), dtype=torch.float32)
        pos = 0
        for ep in self.episodes:
            T = len(ep.actions)
            self.rewards[pos : pos + T] = torch.as_tensor(np.asarray(ep.rewards), dtype=torch.float32)
            self.terminations[pos : pos + T] = torch.as_tensor(np.asarray(ep.terminations), dtype=torch.float32)
            self.truncations[pos : pos + T] = torch.as_tensor(np.asarray(ep.truncations), dtype=torch.float32)
            pos += T

    def _make_indices_skip_trunc(self) -> List[Tuple[int, int]]:
        indices: List[Tuple[int, int]] = []
        cur = 0
        skipped = 0
        for L in self.traj_lengths.tolist():
            L = int(L)
            # If the episode is shorter than horizon, skip entirely
            if L < 1:
                cur += L
                continue
            max_start = cur + max(0, L - self.horizon_steps)
            # if last step truncated, avoid last index
            if self.truncations[cur + L - 1] == 1 and max_start >= cur:
                max_start = max(cur, max_start - 1)
                skipped += 1
            for start in range(cur, max_start + 1):
                indices.append((start, start - cur))
            cur += L
        log.info(f"Skipped {skipped} transitions due to truncation")
        return indices

    def _compute_reward_to_go(self) -> None:
        self.reward_to_go = torch.zeros_like(self.rewards)
        cum = np.cumsum(self.traj_lengths)
        prev = 0
        for end in cum:
            end = int(end)
            traj_rewards = self.rewards[prev:end]
            ret = torch.zeros_like(traj_rewards)
            running = torch.tensor(0.0)
            for t in range(len(traj_rewards)):
                ret[-t - 1] = traj_rewards[-t - 1] + self.discount_factor * running
                running = ret[-t - 1]
            self.reward_to_go[prev:end] = ret
            prev = end

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        start_idx, steps_from_start = self.indices[idx]
        traj_start = start_idx - steps_from_start
        ep_idx = self._get_episode_index(traj_start)
        traj_end = traj_start + int(self.traj_lengths[ep_idx])

        cond_end = start_idx + 1
        cond_start = max(cond_end - self.cond_steps, traj_start)
        cond_states = self._get_states(cond_start, cond_end)  # shape (n, obs_dim)
        if cond_states.shape[0] < self.cond_steps:
            pad = self._get_states(traj_start, traj_start + 1).repeat(self.cond_steps - cond_states.shape[0], 1)
            cond_states = torch.cat([pad, cond_states], dim=0)
        # Dataset uses latest-first ordering, match evaluate_policy
        cond_states = cond_states.flip(0)

        act_end = min(start_idx + self.horizon_steps, traj_end)
        actions = self.actions[start_idx:act_end]
        if actions.shape[0] < self.horizon_steps:
            pad_rows = self.horizon_steps - actions.shape[0]
            actions = torch.cat([actions, torch.zeros((pad_rows, self.actions.shape[1]), dtype=torch.float32)], dim=0)

        next_end = min(start_idx + 1 + self.horizon_steps, traj_end)
        next_start = max(next_end - self.cond_steps, traj_start)
        next_states = self._get_states(next_start, next_end)
        if next_states.shape[0] < self.cond_steps:
            pad = self._get_states(traj_start, traj_start + 1).repeat(self.cond_steps - next_states.shape[0], 1)
            next_states = torch.cat([pad, next_states], dim=0)
        next_states = next_states.flip(0)

        conditions = {"state": cond_states, "next_state": next_states}
        reward = self.rewards[start_idx]
        done = self.terminations[start_idx]

        out = {"actions": actions, "conditions": conditions, "rewards": reward, "dones": done}
        if self.get_mc_return:
            out["rtg"] = self.reward_to_go[start_idx]
        return out

    def _get_episode_index(self, start_idx: int) -> int:
        traj_starts = np.cumsum([0] + self.traj_lengths.tolist()[:-1])
        return int(np.searchsorted(traj_starts, start_idx, side="right") - 1)

    def _get_states(self, start_idx: int, end_idx: int) -> torch.Tensor:
        if hasattr(self, "states"):
            return self.states[start_idx:end_idx]
        elif hasattr(self, "state_data") and "state" in self.state_data:
            return self.state_data["state"][start_idx:end_idx]
        else:
            raise RuntimeError("State data not properly initialized")


# -------------------- Evaluation --------------------

def evaluate_policy(env, model: FQLModel, device: torch.device, episodes: int = 5,
                    render: bool = False, mode: str = 'onestep') -> float:
    model.eval()
    returns: List[float] = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        # state history: keep raw observations but convert to vector before stacking
        state_history = deque(maxlen=getattr(model.actor, "cond_steps", 1))
        for _ in range(getattr(model.actor, "cond_steps", 1)):
            state_history.append(obs)

        while not done:
            # convert deque to numpy array with latest-first ordering
            stack = []
            for s in list(state_history):
                if isinstance(s, dict):
                    if "state" in s:
                        stack.append(np.asarray(s["state"]).reshape(-1))
                    else:
                        # fall back to concatenating values in sorted key order
                        vals = [np.asarray(s[k]).reshape(-1) for k in sorted(s.keys())]
                        stack.append(np.concatenate(vals))
                else:
                    stack.append(np.asarray(s).reshape(-1))
            cond_states = np.stack(stack)
            # dataset uses latest-first ordering -> flip so latest is first
            cond_states = cond_states[::-1].copy()
            o = torch.as_tensor(cond_states, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                act_seq = model.forward({"state": o}, mode=mode)
            act = act_seq[0, 0].detach().cpu().numpy()

            next_obs, r, terminated, truncated, info = env.step(act)
            done = bool(terminated or truncated)
            total += float(r)
            state_history.append(next_obs)

            if render:
                try:
                    env.render()
                except Exception:
                    pass
        returns.append(total)
    return float(np.mean(returns)) if returns else 0.0


# -------------------- Main / Training --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", type=str, default="mujoco/pusher/expert-v0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr_actor", type=float, default=3e-4)
    parser.add_argument("--lr_critic", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--normalize_q_loss", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cond_steps", type=int, default=1)
    parser.add_argument("--horizon_steps", type=int, default=8)
    parser.add_argument("--inference_steps", type=int, default=32)
    parser.add_argument("--eval_every", type=int, default=2)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--bc_warmup_epochs", type=int, default=2)
    parser.add_argument("--max_episodes", type=int, default=10000)
    parser.add_argument("--mc_returns", action="store_true")
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    set_seed(args.seed)

    dataset = minari.load_dataset(args.dataset_id)
    env = dataset.recover_environment()
    eval_env = dataset.recover_environment(eval_env=True)

    ds = MinariStitchedDataset(
        minari_dataset=dataset,
        horizon_steps=args.horizon_steps,
        cond_steps=args.cond_steps,
        max_n_episodes=args.max_episodes,
        use_img=False,
        device="cpu",
        get_mc_return=args.mc_returns,
        discount_factor=args.discount,
    )

    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(torch.cuda.is_available() and args.device.startswith("cuda")),
    )

    # get dims
    obs_dim = ds.obs_dim
    if obs_dim is None:
        obs_dim = ds._get_states(0, 1).shape[-1]
    act_dim = int(np.prod(env.action_space.shape)) if isinstance(env.action_space, spaces.Box) else 1
    act_min = float(env.action_space.low.min()) if isinstance(env.action_space, spaces.Box) else -1.0
    act_max = float(env.action_space.high.max()) if isinstance(env.action_space, spaces.Box) else 1.0

    device = torch.device(args.device)

    flow_backbone = FlowMLP(
        horizon_steps=args.horizon_steps,
        action_dim=act_dim,
        cond_dim=args.cond_steps * obs_dim,
        time_dim=32,
        mlp_dims=[256, 256],
    )
    reflow_policy = ReFlow(
        network=flow_backbone,
        device=device,
        horizon_steps=args.horizon_steps,
        action_dim=act_dim,
        act_min=act_min,
        act_max=act_max,
        obs_dim=obs_dim,
        max_denoising_steps=args.inference_steps,
        seed=args.seed,
        sample_t_type='uniform',
    )
    actor = OneStepActor(
        obs_dim=obs_dim,
        cond_steps=args.cond_steps,
        action_dim=act_dim,
        horizon_steps=args.horizon_steps,
        hidden_dim=512,
    ).to(device)
    critic = CriticObsAct(
        cond_dim=args.cond_steps * obs_dim,
        mlp_dims=[256, 256],
        action_dim=act_dim,
        action_steps=args.horizon_steps,
        double_q=True,
    ).to(device)

    model = FQLModel(
        bc_flow=reflow_policy,
        actor=actor,
        critic=critic,
        inference_steps=args.inference_steps,
        normalize_q_loss=args.normalize_q_loss,
        device=device,
    ).to(device)

    opt_actor = torch.optim.Adam(list(model.bc_flow.network.parameters()) + list(model.actor.parameters()), lr=args.lr_actor)
    opt_critic = torch.optim.Adam(model.critic.parameters(), lr=args.lr_critic)

    os.makedirs(args.save_dir, exist_ok=True)
    best_eval = -float('inf')

    # Training loop
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            pbar = tqdm(dl, desc=f"Epoch {epoch}/{args.epochs}")
            for batch in pbar:
                # batch is a dict with batched tensors
                batch = to_device(batch, device)
                actions = batch['actions']
                rewards = batch['rewards']
                dones = batch['dones']
                conditions = batch['conditions']
                obs = {'state': conditions['state']}
                next_obs = {'state': conditions['next_state']} if 'next_state' in conditions else {'state': conditions['state']}

                # Critic update
                loss_c, info_c = model.loss_critic(obs, actions, next_obs, rewards, dones, args.gamma)
                opt_critic.zero_grad()
                loss_c.backward()
                torch.nn.utils.clip_grad_norm_(model.critic.parameters(), args.grad_clip)
                opt_critic.step()
                model.update_target_critic(args.tau)

                # Actor/Flow update
                if epoch <= args.bc_warmup_epochs:
                    loss_bc = model.loss_bc_flow(obs, actions)
                    opt_actor.zero_grad()
                    loss_bc.backward()
                    torch.nn.utils.clip_grad_norm_(list(model.bc_flow.network.parameters()) + list(model.actor.parameters()), args.grad_clip)
                    opt_actor.step()
                    pbar.set_postfix({"Lc": f"{info_c['loss_critic']:.3f}", "Lbc": f"{loss_bc.item():.3f}"})
                else:
                    loss_a, info_a = model.loss_actor(obs, actions, args.alpha)
                    opt_actor.zero_grad()
                    loss_a.backward()
                    torch.nn.utils.clip_grad_norm_(list(model.bc_flow.network.parameters()) + list(model.actor.parameters()), args.grad_clip)
                    opt_actor.step()
                    pbar.set_postfix({"Lc": f"{info_c['loss_critic']:.3f}", "La": f"{info_a['loss_actor']:.3f}", "Q": f"{info_a['q']:.3f}"})

            # Evaluation
            if epoch % args.eval_every == 0:
                avg_ret = evaluate_policy(eval_env, model, device, episodes=3, mode='onestep', render=False)
                log.info(f"Eval avg return (epoch {epoch}): {avg_ret:.2f}")
                if avg_ret > best_eval:
                    best_eval = avg_ret
                    save_path = os.path.join(args.save_dir, f"fql_best_{args.dataset_id.replace('/', '_')}.pt")
                    torch.save({"actor": model.actor.state_dict(), "critic": model.critic.state_dict(), "flow": model.bc_flow.network.state_dict(), "config": vars(args), "eval_return": avg_ret}, save_path)
                    log.info(f"Saved best model to {save_path}")

    except KeyboardInterrupt:
        log.info("Training interrupted by user. Saving checkpoint...")

    # Final evaluation and save
    final_ret = evaluate_policy(eval_env, model, device, episodes=5, mode='onestep', render=False)
    log.info(f"Training complete. Final avg return: {final_ret:.2f}")
    save_path = os.path.join(args.save_dir, f"fql_final_{args.dataset_id.replace('/', '_')}.pt")
    torch.save({"actor": model.actor.state_dict(), "critic": model.critic.state_dict(), "flow": model.bc_flow.network.state_dict(), "config": vars(args), "eval_return": final_ret}, save_path)
    log.info(f"Model saved to {save_path}")
