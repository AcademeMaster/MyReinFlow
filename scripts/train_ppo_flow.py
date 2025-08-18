"""
PPO Flow training script on Gymnasium continuous control (default: Pendulum-v1).

This trainer uses:
- FlowMLP as base rectified-flow policy backbone
- PPOFlow wrapper to compute PPO losses on flow-based chains
- CriticObs as state-value function

It performs on-policy rollouts with GAE advantages and updates policy/critic.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional
import os, sys, time

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import gymnasium as gym

# Ensure project root on path
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from reflow.networks.flow_mlp import FlowMLP
from reflow.models.critic import CriticObs
from reflow.algorithms.ppo_flow import PPOFlow


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@dataclass
class PPOFlowConfig:
    # env
    env_id: str = "Pendulum-v1"
    seed: int = 42
    render_mode: str = "none"  # 'none' | 'human' | 'rgb_array'

    # rollout
    total_timesteps: int = 50_000
    rollout_steps: int = 2048
    num_envs: int = 1

    # ppo
    update_epochs: int = 10
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # model dims
    cond_steps: int = 1
    horizon_steps: int = 1
    time_dim: int = 16
    mlp_dims_flow: List[int] | None = None
    mlp_dims_critic: List[int] | None = None
    residual_mlp: bool = False

    # learning rate & clip
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    max_grad_norm: float = 0.5
    clip_coef: float = 0.2
    vf_clip_coef: float = 0.2

    # reflow sampling
    inference_steps: int = 32

    # exploration noise (NoisyFlowMLP)
    noise_scheduler_type: str = "linear"  # 'linear' | 'cosine' | 'constant'
    ft_denoising_steps: int = 8
    randn_clip_value: float = 3.0
    min_sampling_denoising_std: float = 0.02
    min_logprob_denoising_std: float = 0.02
    max_logprob_denoising_std: float = 0.3
    logprob_min: float = -5.0
    logprob_max: float = 5.0
    denoised_clip_value: float = 5.0
    time_dim_explore: int = 16
    learn_explore_time_embedding: bool = True
    use_time_independent_noise: bool = False
    noise_hidden_dims: List[int] | None = None
    explore_net_activation_type: str = "Mish"
    logprob_debug_sample: bool = False
    logprob_debug_recalculate: bool = False

    # misc
    project_name: str = "ppo-flow"
    experiment_name: str = "ppo-pendulum"
    save_model_path: Optional[str] = None
    actor_policy_path: Optional[str] = None

    def __post_init__(self):
        if self.mlp_dims_flow is None:
            self.mlp_dims_flow = [256, 256]
        if self.mlp_dims_critic is None:
            self.mlp_dims_critic = [256, 256]


class RolloutBuffer:
    def __init__(self):
        self.obs: List[Tensor] = []          # [B, To, Do]
        self.actions: List[Tensor] = []      # [B, Ta, Da]
        self.chains: List[Tensor] = []       # [B, K+1, Ta, Da]
        self.logprobs: List[Tensor] = []     # [B]
        self.values: List[Tensor] = []       # [B]
        self.rewards: List[Tensor] = []      # [B]
        self.dones: List[Tensor] = []        # [B]

    def add(self, obs: Tensor, action: Tensor, chain: Tensor, logprob: Tensor, value: Tensor, reward: Tensor, done: Tensor):
        self.obs.append(obs)
        self.actions.append(action)
        self.chains.append(chain)
        self.logprobs.append(logprob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def cat(self) -> Dict[str, Tensor]:
        # concatenate along time (first) then flatten envs if needed
        obs = torch.cat(self.obs, dim=0)
        actions = torch.cat(self.actions, dim=0)
        chains = torch.cat(self.chains, dim=0)
        logprobs = torch.cat(self.logprobs, dim=0)
        values = torch.cat(self.values, dim=0)
        rewards = torch.cat(self.rewards, dim=0)
        dones = torch.cat(self.dones, dim=0)
        return {
            'obs': obs,
            'actions': actions,
            'chains': chains,
            'logprobs': logprobs,
            'values': values,
            'rewards': rewards,
            'dones': dones,
        }


class PPOFlowTrainer:
    def __init__(self, cfg: PPOFlowConfig):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        set_seed(cfg.seed)

        # env
        render = None if cfg.render_mode == 'none' else cfg.render_mode
        self.env = gym.make(cfg.env_id, render_mode=render)
        assert isinstance(self.env.action_space, gym.spaces.Box), "PPOFlow 目前仅支持连续动作空间(Box)"
        obs_space = self.env.observation_space
        flat_obs_dim = int(np.prod(getattr(obs_space, 'shape', (1,))))
        act_dim = int(np.prod(self.env.action_space.shape))
        act_low = float(np.min(self.env.action_space.low))
        act_high = float(np.max(self.env.action_space.high))

        # models
        cond_dim = cfg.cond_steps * flat_obs_dim
        flow_net = FlowMLP(
            horizon_steps=cfg.horizon_steps,
            action_dim=act_dim,
            cond_dim=cond_dim,
            time_dim=cfg.time_dim,
            mlp_dims=cfg.mlp_dims_flow,
            residual_style=cfg.residual_mlp,
        )
        critic = CriticObs(
            cond_dim=cond_dim,
            mlp_dims=cfg.mlp_dims_critic,
            residual_style=cfg.residual_mlp,
            use_layernorm=False,
        )

        self.algo = PPOFlow(
            device=self.device,
            policy=flow_net,
            critic=critic,
            actor_policy_path=cfg.actor_policy_path,  # 可选：加载预训练策略（如BC）
            act_dim=act_dim,
            horizon_steps=cfg.horizon_steps,
            act_min=act_low,
            act_max=act_high,
            obs_dim=cond_dim,
            cond_steps=cfg.cond_steps,
            noise_scheduler_type=cfg.noise_scheduler_type,
            inference_steps=cfg.inference_steps,
            ft_denoising_steps=cfg.ft_denoising_steps,
            randn_clip_value=cfg.randn_clip_value,
            min_sampling_denoising_std=cfg.min_sampling_denoising_std,
            min_logprob_denoising_std=cfg.min_logprob_denoising_std,
            logprob_min=cfg.logprob_min,
            logprob_max=cfg.logprob_max,
            clip_ploss_coef=cfg.clip_coef,
            clip_ploss_coef_base=cfg.clip_coef,
            clip_ploss_coef_rate=1.0,
            clip_vloss_coef=cfg.vf_clip_coef,
            denoised_clip_value=cfg.denoised_clip_value,
            max_logprob_denoising_std=cfg.max_logprob_denoising_std,
            time_dim_explore=cfg.time_dim_explore,
            learn_explore_time_embedding=cfg.learn_explore_time_embedding,
            use_time_independent_noise=cfg.use_time_independent_noise,
            noise_hidden_dims=cfg.noise_hidden_dims or [128, 128],
            logprob_debug_sample=cfg.logprob_debug_sample,
            logprob_debug_recalculate=cfg.logprob_debug_recalculate,
            explore_net_activation_type=cfg.explore_net_activation_type,
        ).to(self.device)

        # optims
        self.opt_actor = torch.optim.Adam(
            self.algo.actor_ft.parameters(), lr=cfg.lr_actor
        )
        self.opt_critic = torch.optim.Adam(
            self.algo.critic.parameters(), lr=cfg.lr_critic
        )

        self.obs_buf: List[np.ndarray] = []  # last cond_steps obs
        self.flat_obs_dim = flat_obs_dim
        self.act_dim = act_dim
        self.act_range = [act_low, act_high]

    def _obs_to_cond(self, obs_np: np.ndarray) -> Dict[str, Tensor]:
        # maintain sliding window of cond_steps
        self.obs_buf.append(obs_np.astype(np.float32))
        if len(self.obs_buf) > self.cfg.cond_steps:
            self.obs_buf = self.obs_buf[-self.cfg.cond_steps:]
        if len(self.obs_buf) < self.cfg.cond_steps:
            first = self.obs_buf[0]
            pad = [first for _ in range(self.cfg.cond_steps - len(self.obs_buf))]
            seq = pad + self.obs_buf
        else:
            seq = self.obs_buf
        s = torch.from_numpy(np.stack(seq, axis=0)[None]).to(self.device)  # [1, To, Do]
        return { 'state': s }

    def rollout(self, rollout_steps: int) -> Dict[str, Tensor]:
        buf = RolloutBuffer()
        obs, _ = self.env.reset(seed=self.cfg.seed)
        self.obs_buf = []
        done = False
        for t in range(rollout_steps):
            cond = self._obs_to_cond(obs)
            with torch.no_grad():
                ga_ret = self.algo.get_actions(
                    cond, eval_mode=False, save_chains=True, ret_logprob=True,
                    account_for_initial_stochasticity=True
                )
                # Robust unpacking: supports (xt, x_chain, logprob) or (xt, logprob)
                if isinstance(ga_ret, tuple):
                    if len(ga_ret) == 3:
                        act_seq, x_chain, logprob = ga_ret
                    elif len(ga_ret) == 2:
                        act_seq, logprob = ga_ret
                        x_chain = torch.zeros((act_seq.shape[0], self.cfg.inference_steps + 1, self.cfg.horizon_steps, self.act_dim), device=self.device, dtype=act_seq.dtype)
                    else:
                        act_seq = ga_ret[0]
                        x_chain = torch.zeros((act_seq.shape[0], self.cfg.inference_steps + 1, self.cfg.horizon_steps, self.act_dim), device=self.device, dtype=act_seq.dtype)
                        logprob = torch.zeros((act_seq.shape[0],), device=self.device, dtype=torch.float32)
                else:
                    act_seq = ga_ret
                    x_chain = torch.zeros((act_seq.shape[0], self.cfg.inference_steps + 1, self.cfg.horizon_steps, self.act_dim), device=self.device, dtype=act_seq.dtype)
                    logprob = torch.zeros((act_seq.shape[0],), device=self.device, dtype=torch.float32)
                value = self.algo.critic(cond).view(-1)  # [1]
            act = act_seq[:, 0].detach().cpu().numpy().reshape(-1)  # first step action
            act = np.clip(act, self.act_range[0], self.act_range[1])
            next_obs, reward, terminated, truncated, _ = self.env.step(act)
            done = bool(terminated or truncated)

            # ensure tensors
            if x_chain is None:
                x_chain = torch.zeros((act_seq.shape[0], self.cfg.inference_steps + 1, self.cfg.horizon_steps, self.act_dim), device=self.device, dtype=act_seq.dtype)
            if logprob is None:
                logprob = torch.zeros((act_seq.shape[0],), device=self.device, dtype=torch.float32)
            buf.add(
                obs=cond['state'],
                action=act_seq,
                chain=x_chain,
                logprob=logprob.view(-1),
                value=value.view(-1),
                reward=torch.tensor([reward], device=self.device, dtype=torch.float32),
                done=torch.tensor([done], device=self.device, dtype=torch.float32),
            )

            obs = next_obs
            if done:
                obs, _ = self.env.reset()
                self.obs_buf = []
                done = False

        # compute advantages with GAE
        data = buf.cat()
        with torch.no_grad():
            last_cond = self._obs_to_cond(obs)
            last_value = self.algo.critic(last_cond).view(-1)  # bootstrap
        advantages, returns = self._compute_gae(
            rewards=data['rewards'],
            values=data['values'],
            dones=data['dones'],
            last_value=last_value,
            gamma=self.cfg.gamma,
            lam=self.cfg.gae_lambda,
        )

        batch = {
            'obs': data['obs'],
            'chains': data['chains'],
            'returns': returns.detach(),
            'oldvalues': data['values'].detach(),
            'advantages': advantages.detach(),
            'oldlogprobs': data['logprobs'].detach(),
        }
        return batch

    @staticmethod
    def _compute_gae(rewards: Tensor, values: Tensor, dones: Tensor, last_value: Tensor, gamma: float, lam: float) -> Tuple[Tensor, Tensor]:
        # rewards/values/dones are [T]
        T = rewards.shape[0]
        adv = torch.zeros_like(rewards)
        lastgaelam = 0.0
        for t in reversed(range(T)):
            next_nonterminal = 1.0 - (dones[t])
            next_value = last_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
            lastgaelam = delta + gamma * lam * next_nonterminal * lastgaelam
            adv[t] = lastgaelam
        ret = adv + values
        return adv, ret

    def update(self, batch: Dict[str, Tensor]):
        # make dict inputs
        cond_dict = { 'state': batch['obs'] }

        num_samples = batch['obs'].shape[0]
        idx = torch.randperm(num_samples, device=self.device)
        mb = self.cfg.minibatch_size
        epochs = self.cfg.update_epochs

        pg_loss = torch.tensor(0.0, device=self.device)
        v_loss = torch.tensor(0.0, device=self.device)
        entropy_loss = torch.tensor(0.0, device=self.device)
        clipfrac = 0.0
        approx_kl = 0.0
        ratio_mean = 1.0

        for _ in range(epochs):
            for start in range(0, num_samples, mb):
                end = min(start + mb, num_samples)
                ids = idx[start:end]
                b = {
                    'obs': { 'state': cond_dict['state'][ids] },
                    'chains': batch['chains'][ids],
                    'returns': batch['returns'][ids].view(-1),
                    'oldvalues': batch['oldvalues'][ids].view(-1),
                    'advantages': batch['advantages'][ids].view(-1),
                    'oldlogprobs': batch['oldlogprobs'][ids].view(-1),
                }

                # loss
                loss_tuple = self.algo.loss(
                    obs=b['obs'],
                    chains=b['chains'],
                    returns=b['returns'],
                    oldvalues=b['oldvalues'],
                    advantages=b['advantages'],
                    oldlogprobs=b['oldlogprobs'],
                    use_bc_loss=False,
                    normalize_denoising_horizon=False,
                    normalize_act_space_dimension=False,
                    verbose=False,
                    clip_intermediate_actions=True,
                    account_for_initial_stochasticity=True,
                )
                (
                    pg_loss,
                    entropy_loss,
                    v_loss,
                    bc_loss,
                    clipfrac,
                    approx_kl,
                    ratio_mean,
                    *_
                ) = loss_tuple

                # combine losses (actor + critic)
                actor_loss = pg_loss + 0.0 * entropy_loss + 0.0 * bc_loss
                critic_loss = v_loss

                # backward actor
                self.opt_actor.zero_grad(set_to_none=True)
                actor_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.algo.actor_ft.parameters(), self.cfg.max_grad_norm)
                self.opt_actor.step()

                # backward critic
                self.opt_critic.zero_grad(set_to_none=True)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.algo.critic.parameters(), self.cfg.max_grad_norm)
                self.opt_critic.step()

        return {
            'pg_loss': float(pg_loss.item()),
            'v_loss': float(v_loss.item()),
            'entropy': float((-entropy_loss).item()) if isinstance(entropy_loss, torch.Tensor) else 0.0,
            'clipfrac': float(clipfrac),
            'approx_kl': float(approx_kl),
            'ratio': float(ratio_mean),
        }

    def train(self):
        print("开始训练 PPO Flow ...")
        total_steps = 0
        start_time = time.time()
        while total_steps < self.cfg.total_timesteps:
            batch = self.rollout(self.cfg.rollout_steps)
            total_steps += int(self.cfg.rollout_steps)
            info = self.update(batch)
            elapsed = time.time() - start_time
            print(f"steps={total_steps} pg={info['pg_loss']:.3f} v={info['v_loss']:.3f} ent={info['entropy']:.3f} kl={info['approx_kl']:.4f} clipfrac={info['clipfrac']:.3f} t={elapsed:.1f}s")

        if self.cfg.save_model_path:
            os.makedirs(os.path.dirname(self.cfg.save_model_path) or '.', exist_ok=True)
            torch.save({
                'actor_ft': self.algo.actor_ft.state_dict(),
                'critic': self.algo.critic.state_dict(),
                'config': asdict(self.cfg),
            }, self.cfg.save_model_path)
            print(f"模型已保存到: {self.cfg.save_model_path}")


def main(args=None):
    import argparse
    parser = argparse.ArgumentParser(description='Train PPO Flow on Gymnasium env')
    parser.add_argument('--env', type=str, default='Pendulum-v1')
    parser.add_argument('--total_timesteps', type=int, default=50_000)
    parser.add_argument('--rollout_steps', type=int, default=2048)
    parser.add_argument('--update_epochs', type=int, default=10)
    parser.add_argument('--minibatch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--inference_steps', type=int, default=32)
    parser.add_argument('--horizon_steps', type=int, default=1)
    parser.add_argument('--cond_steps', type=int, default=1)
    parser.add_argument('--no_wandb', action='store_true')  # placeholder for symmetry
    parser.add_argument('--actor_policy_path', type=str, default=None, help='预训练Actor策略权重路径（可选）')
    # Respect forwarded args from CLI wrapper; when None, argparse will read sys.argv
    args = parser.parse_args(args)

    cfg = PPOFlowConfig(
        env_id=args.env,
        total_timesteps=args.total_timesteps,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        seed=args.seed,
        inference_steps=args.inference_steps,
        horizon_steps=args.horizon_steps,
        cond_steps=args.cond_steps,
    actor_policy_path=args.actor_policy_path,
    )

    trainer = PPOFlowTrainer(cfg)
    trainer.train()


if __name__ == '__main__':
    main()
