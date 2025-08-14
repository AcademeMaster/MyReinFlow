"""
FQL baseline training on Minari datasets.
- Builds ReFlow (behavior cloning flow), OneStepActor, and double Critic.
- Trains on transitions sampled from Minari episodes.
- Evaluates in a recovered eval env with optional render/video.

Note: This is a minimal, pragmatic baseline to get things running first.
"""
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple
import sys, os
import argparse
import os

import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from tqdm.auto import tqdm

import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import flatten_space, flatten

try:
    import wandb  # type: ignore
    WANDB_AVAILABLE = True
except Exception:
    wandb = None  # type: ignore
    WANDB_AVAILABLE = False

import minari

# Ensure project root is on sys.path when running as a script
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from itertools import chain

from reflow.flows.reflow import ReFlow
from reflow.networks.flow_mlp import FlowMLP
from reflow.models.critic import CriticObsAct
from reflow.algorithms.fql import OneStepActor, FQLModel


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@dataclass
class FQLConfig:
    dataset_name: str = 'mujoco/pusher/expert-v0'

    # horizon and context
    cond_steps: int = 1  # number of observation steps in the condition
    horizon_steps: int = 4  # number of action steps predicted

    # training
    batch_size: int = 256
    num_epochs: int = 20
    lr_flow: float = 3e-4
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    gamma: float = 0.99
    tau: float = 0.005
    alpha_distill: float = 1.0
    normalize_q_loss: bool = False
    inference_steps: int = 32  # Euler steps for rectified flow sampling
    seed: int = 42
    target_q_agg: str = 'min'  # 'min' | 'mean'

    # architecture
    hidden_dim_actor: int = 512
    mlp_dims_critic: List[int] = None  # type: ignore
    mlp_dims_flow: List[int] = None  # type: ignore
    residual_mlp: bool = False
    time_dim: int = 16

    # perf
    num_workers: int = 0
    use_amp: bool = True

    # logging
    project_name: str = 'fql-baseline'
    experiment_name: str = 'fql-minari'
    eval_episodes: int = 5
    eval_freq: int = 5
    render_mode: str = 'human'  # 'none' | 'human' | 'rgb_array'
    record_video: bool = False
    video_dir: str = 'videos'
    save_model_path: Optional[str] = None
    # training mode
    only_optimize_bc_flow: bool = False

    def __post_init__(self):
        if self.mlp_dims_critic is None:
            self.mlp_dims_critic = [256, 256]
        if self.mlp_dims_flow is None:
            self.mlp_dims_flow = [256, 256]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MinariTransitions(Dataset):
    """Flatten Minari dataset into transition samples for FQL.

    Each item returns a dict:
      - 'obs': FloatTensor [cond_steps, obs_dim]
      - 'action': FloatTensor [horizon_steps, act_dim]
      - 'next_obs': FloatTensor [cond_steps, obs_dim]
      - 'reward': FloatTensor []
      - 'terminated': FloatTensor []  (1.0 if done else 0.0)
    """

    def __init__(self, minari_dataset, observation_space, action_space, cond_steps: int = 1, horizon_steps: int = 1):
        self.ds = minari_dataset
        self.obs_space = observation_space
        self.action_space = action_space
        self.cond_steps = int(max(1, cond_steps))
        self.horizon_steps = int(max(1, horizon_steps))

        # Precompute indices (episode_id, t)
        self.indices: List[Tuple[int, int]] = []
        for ep_idx, ep in enumerate(self.ds):  # type: ignore
            T = len(ep.actions)
            # We need action at t and next obs at t+1, so t in [cond_steps-1, T-1) and t+1 exists
            for t in range(self.cond_steps - 1, T - 1):
                # horizon >1: ensure t+horizon-1 < T
                if t + self.horizon_steps - 1 < T:
                    self.indices.append((ep_idx, t))
        # cache dims
        flat_obs_space = flatten_space(self.obs_space)
        self.obs_dim = int(np.prod(getattr(flat_obs_space, 'shape', (1,))))
        self.act_dim = int(np.prod(self.action_space.shape)) if isinstance(self.action_space, spaces.Box) else int(self.action_space.n)

    def __len__(self):
        return len(self.indices)

    def _flatten_obs(self, obs) -> np.ndarray:
        try:
            flat = flatten(self.obs_space, obs)
        except Exception:
            flat = np.asarray(obs, dtype=np.float32)
        return np.asarray(flat, dtype=np.float32)

    def __getitem__(self, idx: int):
        ep_idx, t = self.indices[idx]
        ep = self.ds[ep_idx]  # type: ignore

        # obs sequence ending at t (inclusive)
        start_t = t - (self.cond_steps - 1)
        obs_seq = [self._flatten_obs(ep.observations[k]) for k in range(start_t, t + 1)]
        obs_seq = np.stack(obs_seq, axis=0).astype(np.float32)

        # action window t : t+horizon_steps-1
        act_seq = [np.asarray(ep.actions[k], dtype=np.float32) for k in range(t, t + self.horizon_steps)]
        act_seq = np.stack(act_seq, axis=0).astype(np.float32)

        # next obs seq ending at t+1
        next_t = t + 1
        next_start = next_t - (self.cond_steps - 1)
        next_obs_seq = [self._flatten_obs(ep.observations[k]) for k in range(next_start, next_t + 1)]
        next_obs_seq = np.stack(next_obs_seq, axis=0).astype(np.float32)

        reward = np.asarray(ep.rewards[t], dtype=np.float32)
        term = np.asarray(ep.terminations[t] or ep.truncations[t], dtype=np.float32)

        return {
            'obs': torch.from_numpy(obs_seq),            # [cond_steps, obs_dim]
            'action': torch.from_numpy(act_seq),         # [horizon_steps, act_dim]
            'next_obs': torch.from_numpy(next_obs_seq),  # [cond_steps, obs_dim]
            'reward': torch.tensor(reward),              # []
            'terminated': torch.tensor(term),            # []
        }


def make_eval_env_from_minari(ds, render_mode: str, record_video: bool, video_dir: str):
    render = None if render_mode == 'none' else render_mode
    env = None
    try:
        env = ds.recover_environment(eval_env=True, render_mode=render)  # type: ignore[arg-type]
    except TypeError:
        try:
            env = ds.recover_environment(eval_env=True)  # type: ignore
        except Exception:
            env = None
    if env is None:
        base = ds.recover_environment()
        spec = getattr(base, 'spec', None)
        env_id = getattr(spec, 'id', None)
        if not isinstance(env_id, str):
            raise RuntimeError('无法推断评估环境 ID')
        env = gym.make(env_id, render_mode=render)

    # optionally record video
    if record_video:
        if getattr(env, 'render_mode', None) != 'rgb_array':
            env.close()
            spec = getattr(env, 'spec', None)
            env_id = getattr(spec, 'id', None)
            if not isinstance(env_id, str):
                base = ds.recover_environment()
                spec = getattr(base, 'spec', None)
                env_id = getattr(spec, 'id', None)
            if not isinstance(env_id, str):
                raise RuntimeError('无法推断评估环境 ID 以启用视频录制')
            env = gym.make(env_id, render_mode='rgb_array')
        os.makedirs(video_dir, exist_ok=True)
        env = gym.wrappers.RecordVideo(env, video_folder=video_dir, episode_trigger=lambda i: True, name_prefix='fql_eval')
    return env


class FQLTrainer:
    def __init__(self, cfg: FQLConfig):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # data & env
        print(f"加载 Minari 数据集: {cfg.dataset_name}")
        self.dataset = minari.load_dataset(cfg.dataset_name)
        base_env = self.dataset.recover_environment()
        self.eval_env = make_eval_env_from_minari(self.dataset, cfg.render_mode, cfg.record_video, cfg.video_dir)

        self.observation_space = base_env.observation_space
        self.action_space = base_env.action_space
        self.flat_obs_space = flatten_space(self.observation_space)
        flat_shape = getattr(self.flat_obs_space, 'shape', None)
        self.obs_dim = int(np.prod(flat_shape if flat_shape else (1,)))
        if isinstance(self.action_space, spaces.Box):
            self.act_dim = int(np.prod(self.action_space.shape))
            self.is_discrete = False
            # scalar bounds for legacy components (flow clamp), plus per-dim arrays for eval
            self.act_low = float(np.min(self.action_space.low))
            self.act_high = float(np.max(self.action_space.high))
            self.act_low_arr = np.asarray(self.action_space.low, dtype=np.float32).reshape(-1)
            self.act_high_arr = np.asarray(self.action_space.high, dtype=np.float32).reshape(-1)
        else:
            raise ValueError('FQL baseline 目前仅支持连续动作空间(Box)。')

        # dataset to transitions
        self.train_ds = MinariTransitions(self.dataset, self.observation_space, self.action_space, cfg.cond_steps, cfg.horizon_steps)
        if cfg.num_workers > 0:
            self.loader = DataLoader(
                self.train_ds,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                pin_memory=(self.device.type=='cuda'),
                persistent_workers=True,
                prefetch_factor=2,
            )
        else:
            self.loader = DataLoader(
                self.train_ds,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=(self.device.type=='cuda'),
            )

        # models
        cond_dim = cfg.cond_steps * self.obs_dim
        flow_net = FlowMLP(
            horizon_steps=cfg.horizon_steps,
            action_dim=self.act_dim,
            cond_dim=cond_dim,
            time_dim=cfg.time_dim,
            mlp_dims=cfg.mlp_dims_flow,
            residual_style=cfg.residual_mlp,
        )
        self.bc_flow = ReFlow(
            network=flow_net,
            device=self.device,
            horizon_steps=cfg.horizon_steps,
            action_dim=self.act_dim,
            act_min=self.act_low,
            act_max=self.act_high,
            obs_dim=cond_dim,  # informational only
            max_denoising_steps=max(1, cfg.inference_steps),
            seed=cfg.seed,
            sample_t_type='uniform',
        ).to(self.device)

        self.actor = OneStepActor(
            obs_dim=self.obs_dim,
            cond_steps=cfg.cond_steps,
            action_dim=self.act_dim,
            horizon_steps=cfg.horizon_steps,
            hidden_dim=cfg.hidden_dim_actor,
        ).to(self.device)

        self.critic = CriticObsAct(
            cond_dim=cond_dim,
            mlp_dims=cfg.mlp_dims_critic,
            action_dim=self.act_dim,
            action_steps=cfg.horizon_steps,
            residual_style=cfg.residual_mlp,
            use_layernorm=False,
        ).to(self.device)

        self.fql = FQLModel(
            self.bc_flow,
            self.actor,
            self.critic,
            cfg.inference_steps,
            cfg.normalize_q_loss,
            cfg.target_q_agg,
            self.act_low_arr,
            self.act_high_arr,
            self.device,
        )
        # optimizers
        if cfg.only_optimize_bc_flow:
            self.opt_flow = torch.optim.Adam(
                self.bc_flow.network.parameters(), lr=cfg.lr_flow, weight_decay=cfg.weight_decay
            )
            self.opt_actor = None
        else:
            # merge actor + bc_flow like reference; do not create separate opt_flow
            self.opt_actor = torch.optim.Adam(
                chain(self.bc_flow.network.parameters(), self.actor.parameters()),
                lr=cfg.lr_actor,
                weight_decay=cfg.weight_decay,
            )
            self.opt_flow = None
        self.opt_critic = torch.optim.Adam(
            self.critic.parameters(), lr=cfg.lr_critic, weight_decay=cfg.weight_decay
        )

        self.scaler = GradScaler(enabled=(self.device.type == 'cuda' and cfg.use_amp))
        self.global_step = 0
        self.warmup_epochs = 1  # small warmup: train flow/actor before actor Q term kicks in
        # keep target critic in eval mode (weights are updated by EMA, not gradient)
        try:
            self.fql.target_critic.eval()
        except Exception:
            pass

    def _to_device(self, batch):
        # reshape to model inputs
        obs = batch['obs'].to(self.device)            # [B, To, Do]
        action = batch['action'].to(self.device)      # [B, Ta, Da]
        next_obs = batch['next_obs'].to(self.device)  # [B, To, Do]
        reward = batch['reward'].to(self.device).view(-1)
        terminated = batch['terminated'].to(self.device).view(-1)
        cond = { 'state': obs }
        next_cond = { 'state': next_obs }
        return cond, action, next_cond, reward, terminated

    def train(self):
        print('开始训练 FQL baseline...')
        for epoch in range(self.cfg.num_epochs):
            # simple schedule: gradually increase Q influence
            q_weight = 0.0 if epoch < 1 else (0.5 if epoch < 3 else 1.0)
            losses = []
            for batch in tqdm(self.loader, desc=f"Epoch {epoch+1}/{self.cfg.num_epochs}"):
                cond, action, next_cond, reward, terminated = self._to_device(batch)

                # Critic update
                self.opt_critic.zero_grad(set_to_none=True)
                with autocast('cuda', enabled=self.scaler.is_enabled()):
                    loss_c, info_c = self.fql.loss_critic(cond, action, next_cond, reward, terminated, self.cfg.gamma)
                if self.scaler.is_enabled():
                    self.scaler.scale(loss_c).backward()
                    # Unscale before clipping to ensure correct norms under AMP
                    self.scaler.unscale_(self.opt_critic)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                    self.scaler.step(self.opt_critic)
                else:
                    loss_c.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                    self.opt_critic.step()

                # Soft update target critic
                self.fql.update_target_critic(self.cfg.tau)  # type: ignore

                # Actor update (when using merged optimizer, includes flow BC term inside loss_actor)
                if not self.cfg.only_optimize_bc_flow:
                    assert self.opt_actor is not None, "opt_actor should exist when only_optimize_bc_flow=False"
                    self.opt_actor.zero_grad(set_to_none=True)
                    with autocast('cuda', enabled=self.scaler.is_enabled()):
                        # During early epochs, reduce Q term influence via q_weight for stability
                        alpha = self.cfg.alpha_distill
                        loss_a, info_a = self.fql.loss_actor(cond, action, alpha, q_weight=q_weight)
                    if self.scaler.is_enabled():
                        self.scaler.scale(loss_a).backward()
                        # Unscale before clipping to ensure correct norms under AMP
                        self.scaler.unscale_(self.opt_actor)
                        # clip both actor and flow params
                        torch.nn.utils.clip_grad_norm_(
                            list(self.actor.parameters()) + list(self.bc_flow.network.parameters()),
                            self.cfg.max_grad_norm,
                        )
                        self.scaler.step(self.opt_actor)
                    else:
                        loss_a.backward()
                        torch.nn.utils.clip_grad_norm_(
                            list(self.actor.parameters()) + list(self.bc_flow.network.parameters()),
                            self.cfg.max_grad_norm,
                        )
                        self.opt_actor.step()
                else:
                    loss_a = torch.tensor(0.0, device=self.device)
                    info_a = {
                        'loss_actor': 0.0,
                        'loss_bc_flow': 0.0,
                        'q_loss': 0.0,
                        'distill_loss': 0.0,
                        'q': 0.0,
                        'onestep_expert_bc_loss': 0.0,
                    }

                # Flow (BC) update only when not using merged optimizer
                if self.cfg.only_optimize_bc_flow and self.opt_flow is not None:
                    self.opt_flow.zero_grad(set_to_none=True)
                    with autocast('cuda', enabled=self.scaler.is_enabled()):
                        loss_bc = self.fql.loss_bc_flow(cond, action)
                    if self.scaler.is_enabled():
                        self.scaler.scale(loss_bc).backward()
                        # Unscale before clipping to ensure correct norms under AMP
                        self.scaler.unscale_(self.opt_flow)
                        torch.nn.utils.clip_grad_norm_(self.bc_flow.network.parameters(), self.cfg.max_grad_norm)
                        self.scaler.step(self.opt_flow)
                    else:
                        loss_bc.backward()
                        torch.nn.utils.clip_grad_norm_(self.bc_flow.network.parameters(), self.cfg.max_grad_norm)
                        self.opt_flow.step()

                # Update scaler after all optimizers have stepped
                if self.scaler.is_enabled():
                    self.scaler.update()

                losses.append(float(loss_c.item()))
                self.global_step += 1
                # optional lightweight logging per N steps
                if WANDB_AVAILABLE and wandb is not None and (self.global_step % 200 == 0):
                    log_payload = {'step': self.global_step, 'train/critic_loss_step': float(loss_c.item())}
                    if 'loss_actor' in info_a:
                        log_payload.update({
                            'train/actor_loss_step': float(info_a.get('loss_actor', 0.0)),
                            'train/q_loss_step': float(info_a.get('q_loss', 0.0)),
                            'train/distill_loss_step': float(info_a.get('distill_loss', 0.0)),
                            'train/loss_bc_flow_step': float(info_a.get('loss_bc_flow', 0.0)),
                        })
                    wandb.log(log_payload)

            avg_loss = float(np.mean(losses)) if losses else 0.0
            print(f"Epoch {epoch+1} critic_loss: {avg_loss:.6f}")
            if WANDB_AVAILABLE and wandb is not None:
                wandb.log({'epoch': epoch+1, 'train/critic_loss': avg_loss, 'train/q_weight': q_weight})

            if (epoch + 1) % self.cfg.eval_freq == 0:
                res = self.evaluate(self.cfg.eval_episodes)
                print(f"评估: return {res['mean_return']:.2f}±{res['std_return']:.2f}, len {res['mean_length']:.1f}")
                if WANDB_AVAILABLE and wandb is not None:
                    wandb.log({
                        'eval/mean_return': res['mean_return'],
                        'eval/std_return': res['std_return'],
                        'eval/mean_length': res['mean_length'],
                        'eval/std_length': res['std_length'],
                    })

    @torch.no_grad()
    def evaluate(self, n_episodes: int = 5) -> Dict[str, float]:
        self.actor.eval()
        returns: List[float] = []
        lengths: List[int] = []

        for _ in range(n_episodes):
            obs, _ = self.eval_env.reset()
            buf: List[np.ndarray] = []  # last cond_steps obs
            ep_ret = 0.0
            ep_len = 0
            done = False
            while not done:
                o = flatten(self.observation_space, obs)
                o = np.asarray(o, dtype=np.float32)
                buf.append(o)
                if len(buf) > self.cfg.cond_steps:
                    buf = buf[-self.cfg.cond_steps:]
                if len(buf) < self.cfg.cond_steps:
                    # pad with first obs
                    first = buf[0]
                    pad = [first for _ in range(self.cfg.cond_steps - len(buf))]
                    seq = pad + buf
                else:
                    seq = buf
                s = torch.from_numpy(np.stack(seq, axis=0)[None]).to(self.device)  # [1, To, Do]
                z = torch.randn(1, self.cfg.horizon_steps, self.act_dim, device=self.device)
                act_seq = self.actor({'state': s}, z)[0]  # [Ta, Da]
                act = act_seq[0].detach().cpu().numpy()
                # per-dimension clip in eval to respect asymmetric bounds
                act = np.clip(act, self.act_low_arr, self.act_high_arr)

                obs, reward, terminated, truncated, _ = self.eval_env.step(act)
                ep_ret += float(reward)
                ep_len += 1
                done = bool(terminated or truncated)
                try:
                    if getattr(self.eval_env, 'render_mode', None) == 'human':
                        self.eval_env.render()
                except Exception:
                    pass
            returns.append(ep_ret)
            lengths.append(ep_len)

        self.actor.train()
        return {
            'mean_return': float(np.mean(returns)) if returns else 0.0,
            'std_return': float(np.std(returns)) if returns else 0.0,
            'mean_length': float(np.mean(lengths)) if lengths else 0.0,
            'std_length': float(np.std(lengths)) if lengths else 0.0,
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'bc_flow': self.bc_flow.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'config': self.cfg.to_dict(),
        }, path)
        print(f"模型已保存到: {path}")


def main(args=None):
    if args is None:
        # 作为独立脚本运行时，解析命令行参数
        parser = argparse.ArgumentParser(description='FQL baseline on Minari dataset')
        parser.add_argument('--dataset', type=str, default='mujoco/pusher/expert-v0')
        parser.add_argument('--epochs', type=int, default=20)
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--cond_steps', type=int, default=1)
        parser.add_argument('--horizon_steps', type=int, default=4)
        parser.add_argument('--lr_flow', type=float, default=3e-4)
        parser.add_argument('--lr_actor', type=float, default=3e-4)
        parser.add_argument('--lr_critic', type=float, default=3e-4)
        parser.add_argument('--eval_freq', type=int, default=5)
        parser.add_argument('--eval_episodes', type=int, default=5)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--render', type=str, default='none', choices=['none','human','rgb_array'])
        parser.add_argument('--record_video_dir', type=str, default='')
        parser.add_argument('--no_wandb', action='store_true')
        parser.add_argument('--only_optimize_bc_flow', action='store_true')
        parser.add_argument('--target_q_agg', type=str, default='min', choices=['min','mean'])
        args = parser.parse_args()

    cfg = FQLConfig(
        dataset_name=args.dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        cond_steps=args.cond_steps,
        horizon_steps=args.horizon_steps,
        lr_flow=args.lr_flow,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        render_mode=args.render,
        record_video=bool(args.record_video_dir),
        video_dir=args.record_video_dir or 'videos',
        only_optimize_bc_flow=bool(args.only_optimize_bc_flow),
        target_q_agg=args.target_q_agg,
    )

    set_seed(cfg.seed)
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if use_wandb and wandb is not None:
        wandb.init(project=cfg.project_name, name=cfg.experiment_name, config=cfg.to_dict())

    trainer = FQLTrainer(cfg)
    trainer.train()

    save_path = cfg.save_model_path or f"experiments/fql_model_{cfg.dataset_name.replace('/', '_')}.pth"
    trainer.save(save_path)

    if use_wandb and wandb is not None:
        wandb.finish()


if __name__ == '__main__':
    main()
