import argparse
import os
import pickle
import numpy as np
import torch
import logging
import wandb
import hydra
from collections import deque
from typing import Tuple
from torch import Tensor

log = logging.getLogger(__name__)

from itertools import chain
from tqdm import tqdm as tqdm


import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import copy


import minari

dataset = minari.load_dataset('mujoco/pusher/expert-v0')
env  = dataset.recover_environment()
eval_env = dataset.recover_environment(eval_env=True)

assert env.spec == eval_env.spec

"""
1-Rectified Flow Policy
"""

import logging
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from collections import namedtuple
from model.flow.mlp_flow import FlowMLP



class MLP(nn.Module):
    def __init__(
        self,
        dim_list,
        append_dim=0,
        append_layers=None,
        activation_type="Tanh",
        out_activation_type="Identity",
        use_layernorm=False,
        use_layernorm_final=False,
        dropout=0,
        use_drop_final=False,
        out_bias_init=None,
        verbose=False,
    ):
        super(MLP, self).__init__()

        # Ensure append_layers is always a list to avoid TypeError
        self.append_layers = append_layers if append_layers is not None else []

        # Construct module list
        self.moduleList = nn.ModuleList()
        num_layer = len(dim_list) - 1
        for idx in range(num_layer):
            i_dim = dim_list[idx]
            o_dim = dim_list[idx + 1]
            if append_dim > 0 and idx in self.append_layers:
                i_dim += append_dim
            linear_layer = nn.Linear(i_dim, o_dim)

            # Add module components
            layers = [("linear_1", linear_layer)]
            if use_layernorm and (idx < num_layer - 1 or use_layernorm_final):
                layers.append(("norm_1", nn.LayerNorm(o_dim)))
            if dropout > 0 and (idx < num_layer - 1 or use_drop_final):
                layers.append(("dropout_1", nn.Dropout(dropout)))

            # Add activation function
            act = (
                activation_dict[activation_type]
                if idx != num_layer - 1
                else activation_dict[out_activation_type]
            )
            layers.append(("act_1", act))

            # Re-construct module
            module = nn.Sequential(OrderedDict(layers))
            self.moduleList.append(module)
        if verbose:
            logging.info(self.moduleList)

        # Initialize the bias of the final linear layer if specified
        if out_bias_init is not None:
            final_linear = self.moduleList[-1][0]  # Linear layer is first in the last Sequential
            nn.init.constant_(final_linear.bias, out_bias_init)

    def forward(self, x, append=None):
        for layer_ind, m in enumerate(self.moduleList):
            if append is not None and layer_ind in self.append_layers:
                x = torch.cat((x, append), dim=-1)
            x = m(x)
        return x


class ResidualMLP(nn.Module):
    """
    Simple multi-layer perceptron network with residual connections for
    benchmarking the performance of different networks. The residual layers
    are based on the IBC paper implementation, which uses 2 residual layers
    with pre-activation with or without dropout and normalization.
    """

    def __init__(
        self,
        dim_list,
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        use_layernorm_final=False,
        dropout=0,
        out_bias_init=None,
    ):
        super(ResidualMLP, self).__init__()
        hidden_dim = dim_list[1]
        num_hidden_layers = len(dim_list) - 3
        assert num_hidden_layers % 2 == 0
        self.layers = nn.ModuleList([nn.Linear(dim_list[0], hidden_dim)])
        self.layers.extend(
            [
                TwoLayerPreActivationResNetLinear(
                    hidden_dim=hidden_dim,
                    activation_type=activation_type,
                    use_layernorm=use_layernorm,
                    dropout=dropout,
                )
                for _ in range(1, num_hidden_layers, 2)
            ]
        )
        self.layers.append(nn.Linear(hidden_dim, dim_list[-1]))
        if use_layernorm_final:
            self.layers.append(nn.LayerNorm(dim_list[-1]))
        self.layers.append(activation_dict[out_activation_type])

        # Initialize the bias of the final linear layer if specified
        if out_bias_init is not None:
            for layer in reversed(self.layers):
                if isinstance(layer, nn.Linear):
                    nn.init.constant_(layer.bias, out_bias_init)
                    break

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TwoLayerPreActivationResNetLinear(nn.Module):
    def __init__(
        self,
        hidden_dim,
        activation_type="Mish",
        use_layernorm=False,
        dropout=0,
    ):
        super().__init__()
        self.l1 = nn.Linear(hidden_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = activation_dict[activation_type]
        if use_layernorm:
            self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-06)
            self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-06)
        if dropout > 0:
            raise NotImplementedError("Dropout not implemented for residual MLP!")

    def forward(self, x):
        x_input = x
        if hasattr(self, "norm1"):
            x = self.norm1(x)
        x = self.l1(self.act(x))
        if hasattr(self, "norm2"):
            x = self.norm2(x)
        x = self.l2(self.act(x))
        return x + x_input

class FlowMLP(nn.Module):
    def __init__(
            self,
            horizon_steps,
            action_dim,
            cond_dim,
            time_dim=16,
            mlp_dims=[256, 256],
            cond_mlp_dims=None,
            activation_type="Mish",
            out_activation_type="Identity",
            use_layernorm=False,
            residual_style=False,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.act_dim_total = action_dim * horizon_steps
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        self.cond_dim = cond_dim
        self.mlp_dims = mlp_dims
        self.activation_type = activation_type
        self.out_activation_type = out_activation_type
        self.use_layernorm = use_layernorm
        self.residual_style = residual_style

        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )

        model = ResidualMLP if residual_style else MLP

        # obs encoder
        if cond_mlp_dims:
            self.cond_mlp = MLP(
                [cond_dim] + cond_mlp_dims,
                activation_type=activation_type,
                out_activation_type="Identity",
            )
            self.cond_enc_dim = cond_mlp_dims[-1]
        else:
            self.cond_enc_dim = cond_dim
        input_dim = time_dim + action_dim * horizon_steps + self.cond_enc_dim

        # velocity head
        self.mlp_mean = model(
            [input_dim] + mlp_dims + [self.act_dim_total],
            activation_type=activation_type,
            out_activation_type=out_activation_type,
            use_layernorm=use_layernorm,
        )

    def forward(
            self,
            action,
            time,
            cond,
            output_embedding=False,
            **kwargs,
    ):
        """
        **Args**:
            action: (B, Ta, Da)
            time: (B,) or int, diffusion step
            cond: dict with key state/rgb; more recent obs at the end
                    state: (B, To, Do)
        **Outpus**:
            velocity.
            vel: (B, Ta, Da) when output_embedding==False
            vel,time_emb, cond_emb: when output_embedding==False
        """
        B, Ta, Da = action.shape

        # flatten action chunk
        action = action.view(B, -1)

        # flatten obs history
        state = cond["state"].view(B, -1)

        # obs encoder
        cond_emb = self.cond_mlp(state) if hasattr(self, "cond_mlp") else state

        # time encoder
        if isinstance(time, int) or isinstance(time, float):
            time = torch.ones((B, 1), device=action.device) * time
        time_emb = self.time_embedding(time.view(B, 1)).view(B, self.time_dim)

        # velocity head
        vel_feature = torch.cat([action, time_emb, cond_emb], dim=-1)
        vel = self.mlp_mean(vel_feature)

        if output_embedding:
            return vel.view(B, Ta, Da), time_emb, cond_emb
        return vel.view(B, Ta, Da)

    def sample_action(self, cond: dict, inference_steps: int, clip_intermediate_actions: bool, act_range: List[float],
                      z: Tensor = None, save_chains: bool = False):
        """
        simply return action via integration (Euler's method). the initial noise could be specified.
        when `save_chains` is True, also return the denoising trajectory.
        """
        B = cond['state'].shape[0]
        device = cond['state'].device

        x_hat: Tensor = z if z is not None else torch.randn(B, self.horizon_steps, self.action_dim, device=device)
        if save_chains:
            x_chain = torch.zeros((B, inference_steps + 1, self.horizon_steps, self.action_dim), device=device)
        dt = (1 / inference_steps) * torch.ones_like(x_hat, device=device)
        steps = torch.linspace(0, 1, inference_steps, device=device).repeat(B, 1)
        for i in range(inference_steps):
            t = steps[:, i]
            vt = self.forward(x_hat, t, cond)
            x_hat += vt * dt
            if clip_intermediate_actions or i == inference_steps - 1:  # always clip the output action. appended by ReinFlow Authors on 04/25/2025
                x_hat = x_hat.clamp(*act_range)
            if save_chains:
                x_chain[:, i + 1] = x_hat
        if save_chains:
            return x_hat, x_chain
        return x_hat


log = logging.getLogger(__name__)
Sample = namedtuple("Sample", "trajectories chains")
class CriticObsAct(torch.nn.Module):
    """State-action double critic network. Q(s,a) or Q_i(s,a)"""

    def __init__(
        self,
        cond_dim,
        mlp_dims,
        action_dim,
        action_steps=1,
        activation_type="Mish",
        use_layernorm=False,
        residual_tyle=False,
        double_q=True,
        **kwargs,
    ):
        super().__init__()
        mlp_dims = [cond_dim + action_dim * action_steps] + mlp_dims + [1]
        if residual_tyle:
            model = ResidualMLP
        else:
            model = MLP
        self.Q1 = model(
            mlp_dims,
            activation_type=activation_type,
            out_activation_type="Identity",
            use_layernorm=use_layernorm,
        )
        if double_q:
            self.Q2 = model(
                mlp_dims,
                activation_type=activation_type,
                out_activation_type="Identity",
                use_layernorm=use_layernorm,
            )

    def forward(self, cond: dict, action):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
        action: (B, Ta, Da)
        """
        B = len(cond["state"])

        # flatten history
        state = cond["state"].view(B, -1)

        # flatten action
        action = action.view(B, -1)

        x = torch.cat((state, action), dim=-1)
        if hasattr(self, "Q2"):
            q1 = self.Q1(x)
            q2 = self.Q2(x)
            return q1.squeeze(1), q2.squeeze(1)
        else:
            q1 = self.Q1(x)
            return q1.squeeze(1)


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
            sample_t_type: str = 'uniform'
    ):
        """Initialize the ReFlow model with specified parameters.

        Args:
            network: FlowMLP network for velocity prediction.
            device: Device to run the model on (e.g., 'cuda' or 'cpu').
            horizon_steps: Number of steps in the trajectory horizon.
            action_dim: Dimension of the action space.
            act_min: Minimum action value for clipping.
            act_max: Maximum action value for clipping.
            obs_dim: Dimension of the observation space.
            max_denoising_steps: Maximum number of denoising steps for sampling.
            seed: Random seed for reproducibility.
            batch_size: Batch size for training and sampling.
            sample_t_type: Type of time sampling ('uniform', 'logitnormal', 'beta').

        Raises:
            ValueError: If max_denoising_steps is not a positive integer.
        """
        super().__init__()
        if int(max_denoising_steps) <= 0:
            raise ValueError('max_denoising_steps must be a positive integer')
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.network = network.to(device)
        self.device = device
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        self.data_shape = (self.horizon_steps, self.action_dim)
        self.act_range = (act_min, act_max)
        self.obs_dim = obs_dim
        self.max_denoising_steps = int(max_denoising_steps)
        self.sample_t_type = sample_t_type

    def generate_trajectory(self, x1: Tensor, x0: Tensor, t: Tensor) -> Tensor:
        """Generate rectified flow trajectory xt = t * x1 + (1 - t) * x0.

        Args:
            x1: Target data tensor of shape (batch_size, horizon_steps, action_dim).
            x0: Initial noise tensor of shape (batch_size, horizon_steps, action_dim).
            t: Time step tensor of shape (batch_size,).

        Returns:
            Tensor: Interpolated trajectory xt of shape (batch_size, horizon_steps, action_dim).
        """
        t_ = (torch.ones_like(x1, device=self.device) * t.view(x1.shape[0], 1, 1)).to(
            self.device)  # ReinFlow Authors revised on 04/23/2025
        xt = t_ * x1 + (1 - t_) * x0
        return xt

    def sample_time(self, batch_size: int, time_sample_type: str = 'uniform', **kwargs) -> Tensor:
        """Sample time steps from a specified distribution in [0, 1).

        Args:
            batch_size: Number of time samples to generate.
            time_sample_type: Type of distribution ('uniform', 'logitnormal', 'beta').
            **kwargs: Additional parameters for non-uniform distributions.

        Returns:
            Tensor: Time samples of shape (batch_size,).

        Raises:
            ValueError: If time_sample_type is not supported.
        """
        supported_time_sample_type = ['uniform', 'logitnormal', 'beta']
        if time_sample_type == 'uniform':
            return torch.rand(batch_size, device=self.device)
        elif time_sample_type == 'logitnormal':
            m = kwargs.get("m", 0)  # Default mean
            s = kwargs.get("s", 1)  # Default standard deviation
            normal_samples = torch.normal(mean=m, std=s, size=(batch_size,), device=self.device)
            logit_normal_samples = (1 / (1 + torch.exp(-normal_samples))).to(self.device)
            return logit_normal_samples
        elif time_sample_type == 'beta':
            alpha = kwargs.get("alpha", 1.5)  # Default alpha
            beta = kwargs.get("beta", 1.0)  # Default beta
            s = kwargs.get("s", 0.999)  # Default cutoff
            beta_distribution = torch.distributions.Beta(alpha, beta)
            beta_sample = beta_distribution.sample((batch_size,)).to(self.device)
            tau = s * (1 - beta_sample)
            return tau
        else:
            raise ValueError(
                f'Unknown time_sample_type = {time_sample_type}. Supported types: {supported_time_sample_type}')

    def generate_target(self, x1: Tensor) -> tuple:
        """Generate training targets for the velocity field.

        Args:
            x1: Real data tensor of shape (batch_size, horizon_steps, action_dim).

        Returns:
            tuple: Contains (xt, t, obs) and v where:
                - xt: Corrupted data tensor of shape (batch_size, horizon_steps, action_dim).
                - t: Time step tensor of shape (batch_size,).
                - v: Target velocity tensor of shape (batch_size, horizon_steps, action_dim).
        """
        t = self.sample_time(batch_size=x1.shape[0], time_sample_type=self.sample_t_type)
        x0 = torch.randn(x1.shape, dtype=torch.float32, device=self.device)
        xt = self.generate_trajectory(x1, x0, t)
        v = x1 - x0
        return (xt, t), v

    def loss(self, xt: Tensor, t: Tensor, obs: dict, v: Tensor) -> Tensor:
        """Compute the MSE loss between predicted and target velocities.

        Args:
            xt: Corrupted data tensor of shape (batch_size, horizon_steps, action_dim).
            t: Time step tensor of shape (batch_size,).
            obs: Dictionary containing 'state' tensor of shape (batch_size, cond_steps, obs_dim).
            v: Target velocity tensor of shape (batch_size, horizon_steps, action_dim).

        Returns:
            Tensor: Mean squared error loss.
        """
        v_hat = self.network(xt, t, obs)
        return F.mse_loss(input=v_hat, target=v)

    @torch.no_grad()
    def sample(
            self,
            cond: dict,
            inference_steps: int,
            record_intermediate: bool = False,
            clip_intermediate_actions: bool = True,
            z: torch.Tensor = None
    ) -> Sample:
        """Sample trajectories using the learned velocity field.

        Args:
            cond: Dictionary containing 'state' tensor of shape (batch_size, cond_steps, obs_dim).
            inference_steps: Number of denoising steps.
            record_intermediate: Whether to return intermediate predictions.
            clip_intermediate_actions: Whether to clip actions to act_range.

        Returns:
            Sample: Named tuple with 'trajectories' (and 'chains' if record_intermediate).
        """
        B = cond['state'].shape[0]
        if record_intermediate:
            x_hat_list = torch.zeros((inference_steps,) + self.data_shape, device=self.device)
        x_hat = z if z is not None else torch.randn((B,) + self.data_shape, device=self.device)
        dt = (1 / inference_steps) * torch.ones_like(x_hat, device=self.device)
        steps = torch.linspace(0, 1, inference_steps, device=self.device).repeat(B, 1)
        for i in range(inference_steps):
            t = steps[:, i]
            vt = self.network(x_hat, t, cond)
            x_hat += vt * dt
            if clip_intermediate_actions or i == inference_steps - 1:  # always clip the output action. appended by ReinFlow Authors on 04/25/2025
                x_hat = x_hat.clamp(*self.act_range)
            if record_intermediate:
                x_hat_list[i] = x_hat
        return Sample(trajectories=x_hat, chains=x_hat_list if record_intermediate else None)

from collections import namedtuple
import numpy as np
import torch
import logging
import pickle
import random
from tqdm import tqdm

log = logging.getLogger(__name__)

Batch = namedtuple("Batch", "actions conditions")
Transition = namedtuple("Transition", "actions conditions rewards dones")
TransitionWithReturn = namedtuple(
    "Transition", "actions conditions rewards dones reward_to_gos"
)


class StitchedSequenceDataset(torch.utils.data.Dataset):
    """
    Load stitched trajectories of states/actions/images, and 1-D array of traj_lengths, from npz or pkl file.

    Use the first max_n_episodes episodes (instead of random sampling)

    Example:
        states: [----------traj 1----------][---------traj 2----------] ... [---------traj N----------]
        Episode IDs (determined based on traj_lengths):  [----------   1  ----------][----------   2  ---------] ... [----------   N  ---------]

    Each sample is a namedtuple of (1) chunked actions and (2) a list (obs timesteps) of dictionary with keys states and images.

    """

    def __init__(
            self,
            dataset_path,
            horizon_steps=64,
            cond_steps=1,
            img_cond_steps=1,
            max_n_episodes=-1,
            use_img=False,
            device="cuda:0",
    ):
        assert (
                img_cond_steps <= cond_steps
        ), "consider using more cond_steps than img_cond_steps"

        print(f"max_n_episodes={max_n_episodes}")
        self.horizon_steps = horizon_steps
        self.cond_steps = cond_steps  # states (proprio, etc.)
        self.img_cond_steps = img_cond_steps
        self.device = device
        self.use_img = use_img
        self.max_n_episodes = max_n_episodes
        self.dataset_path = dataset_path

        # Load dataset to device specified
        if dataset_path.endswith(".npz"):
            dataset = np.load(dataset_path, allow_pickle=False)  # only np arrays
        elif dataset_path.endswith(".pkl"):
            with open(dataset_path, "rb") as f:
                dataset = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")

        if max_n_episodes == -1:
            max_n_episodes = len(dataset["traj_lengths"])
            log.info(f"max_n_episodes specified as -1, fall back to maximum value {max_n_episodes}")
        traj_lengths = dataset["traj_lengths"][:max_n_episodes]  # 1-D array
        total_num_steps = np.sum(traj_lengths)
        # Set up indices for sampling
        self.indices = self.make_indices(traj_lengths, horizon_steps)

        # Extract states and actions up to max_n_episodes
        self.states = (
            torch.from_numpy(dataset["states"][:total_num_steps]).float().to(device)
        )  # (total_num_steps, obs_dim)

        self.actions = (
            torch.from_numpy(dataset["actions"][:total_num_steps]).float().to(device)
        )  # (total_num_steps, action_dim)

        log.info(f"Successfully loaded dataset from {dataset_path}")
        n_eps = min(max_n_episodes, len(traj_lengths))
        if n_eps <= 0:
            raise ValueError(f"number of episodes less than 1, check where is wrong...")
        log.info(f"Number of episodes: {n_eps}")
        log.info(f"States shape/type: {self.states.shape, self.states.dtype}")
        log.info(f"Actions shape/type: {self.actions.shape, self.actions.dtype}")
        if self.use_img:
            self.images = torch.from_numpy(dataset["images"][:total_num_steps]).to(
                device
            )  # (total_num_steps, C, H, W)
            log.info(f"Images shape/type: {self.images.shape, self.images.dtype}")
        log.info(f"Finished creating {self.__class__.__name__} from {dataset_path}")

    def __getitem__(self, idx):
        """
        repeat states/images if using history observation at the beginning of the episode
        """
        start, num_before_start = self.indices[idx]
        end = start + self.horizon_steps
        # print(f"start={start}, num_before_start={num_before_start}, (start - num_before_start) : (start + 1)={(start - num_before_start)} : {(start + 1)}")
        # print(f"self.states length: {self.states.shape}")

        # print(f"self.states[673463 : 673758]")
        # try:
        #     print(self.states[673463])
        # except RuntimeError as e:
        #     print(f"RuntimeError: {e}")

        # print(self.states[673463])
        # exit()

        states = self.states[(start - num_before_start): (start + 1)]
        actions = self.actions[start:end]
        states = torch.stack(
            [
                states[max(num_before_start - t, 0)]
                for t in reversed(range(self.cond_steps))
            ]
        )  # more recent is at the end
        conditions = {"state": states}
        if self.use_img:
            images = self.images[(start - num_before_start): end]
            images = torch.stack(
                [
                    images[max(num_before_start - t, 0)]
                    for t in reversed(range(self.img_cond_steps))
                ]
            )
            conditions["rgb"] = images
        batch = Batch(actions, conditions)
        return batch

    def make_indices(self, traj_lengths, horizon_steps):
        """
        makes indices for sampling from dataset;
        each index maps to a datapoint, also save the number of steps before it within the same trajectory
        """
        indices = []
        cur_traj_index = 0
        for traj_length in traj_lengths:
            max_start = cur_traj_index + traj_length - horizon_steps
            indices += [
                (i, i - cur_traj_index) for i in range(cur_traj_index, max_start + 1)
            ]
            cur_traj_index += traj_length
        return indices

    def set_train_val_split(self, train_split):
        """
        Not doing validation right now
        """
        num_train = int(len(self.indices) * train_split)
        train_indices = random.sample(self.indices, num_train)
        val_indices = [i for i in range(len(self.indices)) if i not in train_indices]
        self.indices = train_indices
        return val_indices

    def __len__(self):
        return len(self.indices)


class StitchedSequenceQLearningDataset(StitchedSequenceDataset):
    """
    Extends StitchedSequenceDataset to include rewards and dones for Q learning

    **Returns:**
    batch = Transition(
                actions,
                conditions,
                rewards,
                dones,
            )

    Do not load the last step of **truncated** episodes since we do not have the correct next state for the final step of each episode.
    Truncation can be determined by terminal=False by the end of the episode.
    """

    def __init__(
            self,
            dataset_path,
            max_n_episodes=10000,
            discount_factor=1.0,
            device="cuda:0",
            get_mc_return=False,
            **kwargs,
    ):
        if dataset_path.endswith(".npz"):
            dataset = np.load(dataset_path, allow_pickle=False)
        elif dataset_path.endswith(".pkl"):
            with open(dataset_path, "rb") as f:
                dataset = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")
        traj_lengths = dataset["traj_lengths"][:max_n_episodes]
        total_num_steps = np.sum(traj_lengths)

        # discount factor
        self.discount_factor = discount_factor

        # rewards and dones(terminals)
        self.rewards = (
            torch.from_numpy(dataset["rewards"][:total_num_steps]).float().to(device)
        )
        log.info(f"Rewards shape/type: {self.rewards.shape, self.rewards.dtype}")
        self.dones = (
            torch.from_numpy(dataset["terminals"][:total_num_steps]).to(device).float()
        )
        log.info(f"Dones shape/type: {self.dones.shape, self.dones.dtype}")

        super().__init__(
            dataset_path=dataset_path,
            max_n_episodes=max_n_episodes,
            device=device,
            **kwargs,
        )
        log.info(f"Total number of transitions using: {len(self)}")

        # compute discounted reward-to-go for each trajectory
        self.get_mc_return = get_mc_return
        if get_mc_return:
            self.reward_to_go = torch.zeros_like(self.rewards)
            cumulative_traj_length = np.cumsum(traj_lengths)
            prev_traj_length = 0
            for i, traj_length in tqdm(
                    enumerate(cumulative_traj_length), desc="Computing reward-to-go"
            ):
                traj_rewards = self.rewards[prev_traj_length:traj_length]
                returns = torch.zeros_like(traj_rewards)
                prev_return = 0
                for t in range(len(traj_rewards)):
                    returns[-t - 1] = (
                            traj_rewards[-t - 1] + self.discount_factor * prev_return
                    )
                    prev_return = returns[-t - 1]
                self.reward_to_go[prev_traj_length:traj_length] = returns
                prev_traj_length = traj_length
            log.info(f"Computed reward-to-go for each trajectory.")

    def make_indices(self, traj_lengths, horizon_steps):
        """
        skip last step of truncated episodes
        """
        num_skip = 0
        indices = []
        cur_traj_index = 0
        for traj_length in traj_lengths:
            max_start = cur_traj_index + traj_length - horizon_steps
            if not self.dones[cur_traj_index + traj_length - 1]:  # truncation
                max_start -= 1
                num_skip += 1
            indices += [
                (i, i - cur_traj_index) for i in range(cur_traj_index, max_start + 1)
            ]
            cur_traj_index += traj_length
        log.info(f"Number of transitions skipped due to truncation: {num_skip}")
        return indices

    def __getitem__(self, idx):
        start, num_before_start = self.indices[idx]
        end = start + self.horizon_steps
        states = self.states[(start - num_before_start): (start + 1)]
        actions = self.actions[start:end]
        rewards = self.rewards[start: (start + 1)]
        dones = self.dones[start: (start + 1)]

        # Account for action horizon
        if idx < len(self.indices) - self.horizon_steps:
            next_states = self.states[
                          (start - num_before_start + self.horizon_steps): start
                                                                           + 1
                                                                           + self.horizon_steps
                          ]  # even if this uses the first state(s) of the next episode, done=True will prevent bootstrapping. We have already filtered out cases where done=False but end of episode (truncation).
        else:
            # prevents indexing error, but ignored since done=True
            next_states = torch.zeros_like(states)

        # stack obs history
        states = torch.stack(
            [
                states[max(num_before_start - t, 0)]
                for t in reversed(range(self.cond_steps))
            ]
        )  # more recent is at the end
        next_states = torch.stack(
            [
                next_states[max(num_before_start - t, 0)]
                for t in reversed(range(self.cond_steps))
            ]
        )  # more recent is at the end
        conditions = {"state": states, "next_state": next_states}
        if self.use_img:
            images = self.images[(start - num_before_start): end]
            images = torch.stack(
                [
                    images[max(num_before_start - t, 0)]
                    for t in reversed(range(self.img_cond_steps))
                ]
            )
            conditions["rgb"] = images
        if self.get_mc_return:
            reward_to_gos = self.reward_to_go[start: (start + 1)]
            batch = TransitionWithReturn(
                actions,
                conditions,
                rewards,
                dones,
                reward_to_gos,
            )
        else:
            batch = Transition(
                actions,
                conditions,
                rewards,
                dones,
            )
        return batch


class OneStepActor(nn.Module):
    """Distill a multistep flow model to single step stochastic policy"""

    def __init__(self,
                 obs_dim,
                 cond_steps,
                 action_dim,
                 horizon_steps,
                 hidden_dim=512):
        """Initialize the OneStepActor with a neural network to map observations and noise to actions.

        Args:
            obs_dim (int): Dimension of the observation space.
            cond_steps (int): Number of observation steps in the horizon.
            action_dim (int): Dimension of the action space.
            horizon_steps (int): Number of action steps in the horizon.
            hidden_dim (int, optional): Hidden layer size. Defaults to 512.
        """
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

    def forward(self, cond: Dict[str, Tensor], z: Tensor) -> Tensor:
        """Generate actions from observations and noise using the actor network.

        Args:
            cond (Dict[str, Tensor]): Dictionary containing the state tensor with key 'state', which is a tensor of shape (batch, cond_steps, obs_dim)
            z (Tensor): Noise tensor of shape (batch, horizon_steps, action_dim).

        Returns:
            Tensor: Actions of shape (batch, horizon_steps, action_dim).
        """
        s = cond['state']  # (batch, cond_step, obs_dim)
        B = s.shape[0]
        s_flat = s.view(B, -1)  # (batch, cond_step * obs_dim)
        z_flat = z.view(B, -1)  # (batch, horizon*action_dim)
        feature = torch.cat([s_flat, z_flat], dim=-1)
        output = self.net(feature)
        actions: Tensor = output.view(B, self.horizon_steps, self.action_dim)
        return actions


class FQLModel(nn.Module):
    def __init__(self,
                 bc_flow: ReFlow,
                 actor: OneStepActor,
                 critic: CriticObsAct,
                 inference_steps: int,
                 normalize_q_loss: bool,
                 device):
        """Initialize the FQL model with behavior cloning flow, actor, critic, and target critic.

        Args:
            bc_flow (ReFlow): Behavior cloning flow model.
            actor (OneStepActor): Actor network for action generation.
            critic (CriticObsAct): Critic network for Q-value estimation.
            inference_steps (int): Number of inference steps for the flow model.
            normalize_q_loss (bool): Whether to normalize the Q-value loss.
            device: Device to run the model on (e.g., 'cuda').
        """
        super().__init__()
        self.bc_flow: ReFlow = bc_flow
        self.actor: OneStepActor = actor
        self.critic: CriticObsAct = critic
        self.target_critic: CriticObsAct = copy.deepcopy(self.critic)
        self.device = device
        self.inference_steps = inference_steps  # for the base model.
        self.normalize_q_loss = normalize_q_loss

    def forward(self,
                cond: Dict[str, Tensor],
                mode: str = 'onestep'):
        """Generate actions for a batch of observations using the actor with random noise.

        Args:
            cond (Dict[str, Tensor]): Dictionary containing the state tensor with key 'state'.

        Returns:
            Tensor: Actions of shape (batch, horizon_steps, action_dim).
        """
        batch_size = cond['state'].shape[0]
        assert mode in ['onestep', 'base_model']
        if mode == 'onestep':
            z = torch.randn(batch_size, self.actor.horizon_steps, self.actor.action_dim, device=self.device)
            actions = self.actor(cond, z)
        elif mode == 'base_model':
            actions = self.bc_flow.sample(cond, self.inference_steps, record_intermediate=False,
                                          clip_intermediate_actions=False).trajectories
        return actions

    def loss_bc_flow(self, obs: Dict[str, Tensor], actions: Tensor):
        """Compute the behavior cloning flow loss by comparing predicted and target flow values. This is the same as doing pre-training.

        Args:
            obs (Dict[str, Tensor]): Dictionary containing the state tensor with key 'state'.
            actions (Tensor): Ground truth actions of shape (batch, horizon_steps, action_dim).

        Returns:
            Tensor: Mean squared error loss for the behavior cloning flow.
        """
        (xt, t), v = self.bc_flow.generate_target(actions)
        v_hat = self.bc_flow.network(xt, t, obs)
        loss_bc_flow = F.mse_loss(v_hat, v)
        return loss_bc_flow

    def loss_critic(self, obs, actions, next_obs, rewards, terminated, gamma) -> Tuple[Tensor, Dict]:
        """Compute the critic loss using mean squared error between predicted and target Q-values, with debug information.
        Use the one-step actor's action to compute q target, use dataset batch's action as q  prediction. Then adopt TD loss.
        Args:
            obs (Dict[str, Tensor]): Current state observations.
            actions (Tensor): Actions taken.
            next_obs (Dict[str, Tensor]): Next state observations.
            rewards (Tensor): Rewards received.
            terminated (Tensor): Termination flags.
            gamma (float): Discount factor.

        Returns:
            Tuple[Tensor, Dict]: Critic loss and dictionary of debug information.
        """
        with torch.no_grad():
            # get action from one-step actor.
            z = torch.randn_like(actions, device=self.device)
            next_actions = self.actor.forward(next_obs, z)
            next_actions.clamp_(-1, 1)
            next_q1, next_q2 = self.target_critic.forward(next_obs, next_actions)
            next_q = torch.mean(torch.stack([next_q1, next_q2], dim=0),
                                dim=0)  # this is typical of fql. possible bug in reproduction
            # target q value
            target = rewards + gamma * (1 - terminated) * next_q
        # get q prediction from dataset actions.
        q1, q2 = self.critic.forward(obs, actions)
        loss_critic = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        # debug info
        loss_critic_info = {
            'loss_critic': loss_critic.item(),
            'q1_mean': q1.mean().item(),
            'q1_max': q1.max().item(),
            'q1_min': q1.min().item(),
            'q2_mean': q2.mean().item(),
            'q2_max': q2.max().item(),
            'q2_min': q2.min().item(),
            'loss_critic_1': F.mse_loss(q1, target).item(),
            'loss_critic_2': F.mse_loss(q2, target).item(),  # Fixed from q1 to q2
        }
        return loss_critic, loss_critic_info

    def loss_actor(self, obs: Dict[str, Tensor], action_batch: Tensor, alpha: float) -> Tuple[
        Tensor, Dict[str, Tensor]]:
        """Compute the actor loss combining behavior cloning, Q-value, and distillation losses, with debug information.

        Args:
            obs (Dict[str, Tensor]): State observations.
            action_batch (Tensor): Ground truth actions for behavior cloning.
            alpha (float): Weight for the distillation loss.

        Returns:
            Tuple[Tensor, Dict]: Actor loss and dictionary of debug information.
        """
        # get distillation loss
        batch_size = obs['state'].shape[0]
        z = torch.randn(batch_size, self.actor.horizon_steps, self.actor.action_dim, device=self.device)
        a_ω = self.actor.forward(obs, z)
        a_θ = self.bc_flow.sample(cond=obs, inference_steps=self.inference_steps, record_intermediate=False,
                                  clip_intermediate_actions=False, z=z).trajectories  # Use same z
        distill_loss = F.mse_loss(a_ω, a_θ)

        # get q loss
        actor_actions = torch.clamp(a_ω, -1, 1)
        q1, q2 = self.critic.forward(obs, actor_actions)
        q = torch.mean(torch.stack([q1, q2], dim=0), dim=0)
        q_loss = -q.mean()
        if self.normalize_q_loss:
            lam = 1 / torch.abs(q).mean().detach()
            q_loss = lam * q_loss

        # get BC loss
        loss_bc_flow = self.loss_bc_flow(obs, action_batch)

        loss_actor = loss_bc_flow + alpha * distill_loss + q_loss

        # debug info
        onestep_expert_bc_loss = F.mse_loss(a_ω, action_batch)
        loss_actor_info = {
            'loss_actor': loss_actor.item(),
            'loss_bc_flow': loss_bc_flow.item(),
            'q_loss': q_loss.item(),
            'distill_loss': distill_loss.item(),
            'q': q.mean().item(),
            'onestep_expert_bc_loss': onestep_expert_bc_loss.item()
        }
        return loss_actor, loss_actor_info

    def update_target_critic(self, tau: float):
        """Update the target double critic network."""
        for target_param, source_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                source_param.data * tau + target_param.data * (1.0 - tau)
            )




import numpy as np


def td_values(states, rewards, terminateds, state_values, gamma=0.99, alpha=0.95, lam=0.95):
    """
    Compute TD(λ) estimates for a list of samples.
    This snippet is taken from agent/finetune/diffusion_baselines/train_awr_diffusion_agent.py

    Args:
        states: List of state observations (np.ndarray).
        rewards: List of rewards (np.ndarray).
        terminateds: List of termination flags (np.ndarray).
        state_values: Estimated state values (np.ndarray).
        gamma: Discount factor (float).
        alpha: TD learning rate (float).
        lam: Lambda for TD(λ) (float).

    Returns:
        np.ndarray: TD(λ) estimates.
    """
    sample_count = len(states)
    tds = np.zeros_like(state_values, dtype=np.float32)
    next_value = state_values[-1].copy()
    next_value[terminateds[-1]] = 0.0

    val = 0.0
    for i in range(sample_count - 1, -1, -1):
        if i < sample_count - 1:
            next_value = state_values[i + 1]
            next_value = next_value * (1 - terminateds[i])
        state_value = state_values[i]
        error = rewards[i] + gamma * next_value - state_value
        val = alpha * error + gamma * lam * (1 - terminateds[i]) * val
        tds[i] = val + state_value
    return tds


class TrainFQLAgent(TrainAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Load offline dataset
        self.dataset_offline = hydra.utils.instantiate(cfg.offline_dataset)

        # Discount factor applied every act_steps
        self.gamma = cfg.train.gamma

        # Optimizers for actor and critic
        self.only_optimize_bc_flow = cfg.only_optimize_bc_flow
        log.info(f"self.only_optimize_bc_flow={self.only_optimize_bc_flow}.")

        if self.only_optimize_bc_flow:
            self.bc_actor_optimizer = torch.optim.Adam(
                self.model.bc_flow.parameters(),
                lr=cfg.train.actor_lr,
            )
            self.onestep_actor_optimizer = torch.optim.Adam(
                self.model.actor.parameters(),
                lr=cfg.train.actor_lr,
            )
        else:
            self.actor_optimizer = torch.optim.Adam(
                chain(self.model.bc_flow.parameters(), self.model.actor.parameters()),
                lr=cfg.train.actor_lr,
            )

        self.critic_optimizer = torch.optim.Adam(
            self.model.critic.parameters(),
            lr=cfg.train.critic_lr,
        )

        # Target network update rate
        self.target_ema_rate = cfg.train.target_ema_rate

        # Reward scaling factor
        self.scale_reward_factor = cfg.train.scale_reward_factor

        # Update frequencies. for fql they use 1:1 ratio between actor and critic
        self.critic_update_freq = int(cfg.train.batch_size / cfg.train.critic_replay_ratio)  # default is 1
        self.actor_update_freq = int(cfg.train.batch_size / cfg.train.actor_replay_ratio)  # default is 1
        self.actor_update_number = cfg.train.actor_update_repeat  # default is 1

        # Buffer size for online data
        self.buffer_size = cfg.train.buffer_size

        # offline training steps:
        self.offline_steps = cfg.train.offline_steps
        self.online_steps = cfg.train.online_steps
        if self.n_train_itr != self.offline_steps + self.online_steps:
            raise ValueError(
                f"self.n_train_itr={self.n_train_itr}!=self.offline_steps({self.offline_steps})+self.online_steps({self.online_steps})")

        # Number of evaluation episodes
        self.n_steps_eval = cfg.train.n_steps_eval
        self.eval_base_model = cfg.train.eval_base_model

        # distillation loss weight (need to be tuned for each environment)
        self.alpha = cfg.train.alpha

        # Model and device setup
        self.model: FQLModel
        self.device = cfg.get('device', 'cuda:7')
        self.model.to(self.device)

    def agent_update(self, batch: Tuple[dict, Tensor, dict, Tensor, Tensor]):
        cond_b, actions_b, next_cond_b, reward_b, terminated_b = batch

        loss_actor = 0.0

        # Update critic
        loss_critic, loss_critic_info = self.model.loss_critic(
            *batch,
            self.gamma,
        )
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # Update target critic
        self.model.update_target_critic(self.target_ema_rate)

        # Update actor if frequency condition met. for fql they use 1:1 update ratio between actor and critic.
        if self.only_optimize_bc_flow:
            loss_bc_flow = self.model.loss_bc_flow(cond_b, actions_b)
            self.bc_actor_optimizer.zero_grad()
            loss_bc_flow.backward()
            self.bc_actor_optimizer.step()
            # place holder. this time loss_actor only records bc_flow loss.
            loss_actor = 0.00
            loss_actor_info = {
                'loss_actor': 0.00,
                'loss_bc_flow': loss_bc_flow.item(),
                'q_loss': 0.00,
                'distill_loss': 0.00,
                'q': 0.00,
                'onestep_expert_bc_loss': 0.0
            }
        else:
            loss_actor, loss_actor_info = self.model.loss_actor(cond_b, actions_b, self.alpha)
            self.actor_optimizer.zero_grad()
            loss_actor.backward()
            self.actor_optimizer.step()

        return loss_critic, loss_actor, loss_critic_info, loss_actor_info

    def run(self):
        # Initialize online FIFO replay buffers
        self.obs_buffer = deque(maxlen=self.buffer_size)
        self.next_obs_buffer = deque(maxlen=self.buffer_size)
        self.action_buffer = deque(maxlen=self.buffer_size)
        self.reward_buffer = deque(maxlen=self.buffer_size)
        self.terminated_buffer = deque(maxlen=self.buffer_size)
        loss_critic_info = {}
        loss_actor_info = {}

        # Load offline dataset into numpy arrays for efficient sampling
        dataloader_offline: StitchedSequenceQLearningDataset = torch.utils.data.DataLoader(
            self.dataset_offline,
            batch_size=len(self.dataset_offline),
            drop_last=False,
        )
        # Get dataset size and shapes
        dataset_size = len(self.dataset_offline)
        log.info(f"dataset_size={dataset_size}")
        obs_dim = self.dataset_offline[0][1]["state"].shape[-1]  # Shape of state
        cond_steps = self.dataset_offline[0][1]["state"].shape[-2]  # Horizon of state
        act_steps = self.dataset_offline[0][0].shape[0]  # Number of action steps
        act_dim = self.dataset_offline[0][0].shape[-1]  # Action dimension
        assert act_dim == self.cfg.action_dim
        assert act_steps == self.cfg.act_steps
        assert obs_dim == self.cfg.obs_dim
        assert cond_steps == self.cfg.cond_steps

        log.info(
            f"Caching dataset into numpy arrays with dataset_size={dataset_size}, obs_dim={obs_dim}, act_steps={act_steps}, act_dim={act_dim}")
        # Pre-allocate NumPy arrays
        obs_buffer_off = np.empty((dataset_size, cond_steps, obs_dim), dtype=np.float32)
        next_obs_buffer_off = np.empty((dataset_size, cond_steps, obs_dim), dtype=np.float32)
        action_buffer_off = np.empty((dataset_size, act_steps, act_dim), dtype=np.float32)
        reward_buffer_off = np.empty(dataset_size, dtype=np.float32)
        terminated_buffer_off = np.empty(dataset_size, dtype=np.float32)
        assert self.batch_size < len(obs_buffer_off)

        # Copy batches into pre-allocated arrays
        start_idx = 0
        for batch in dataloader_offline:
            actions, states_and_next, rewards, terminated = batch
            states = states_and_next["state"]  # Shape: (batch, obs_dim)
            next_states = states_and_next["next_state"]  # Shape: (batch, obs_dim)
            batch_size = states.shape[0]
            end_idx = start_idx + batch_size
            # Copy data directly into arrays
            obs_buffer_off[start_idx:end_idx] = states.cpu().numpy()  # Shape: (N_off, obs_dim)
            next_obs_buffer_off[start_idx:end_idx] = next_states.cpu().numpy()  # Shape: (N_off, obs_dim)
            action_buffer_off[start_idx:end_idx] = actions.cpu().numpy()  # Shape: (N_off, act_dim)
            reward_buffer_off[start_idx:end_idx] = rewards.cpu().numpy().flatten()  # Shape: (N_off,)
            terminated_buffer_off[start_idx:end_idx] = terminated.cpu().numpy().flatten()  # Shape: (N_off,)
            start_idx = end_idx
        log.info(
            f"Finished caching dataset into numpy arrays with dataset_size={dataset_size}, obs_dim={obs_dim}, act_steps={act_steps}, act_dim={act_dim}. Sampling starts.")

        # Training loop
        timer = Timer()
        run_results = []
        cnt_train_step = 0
        self.success_rate = 0.0
        self.avg_episode_reward = 0.0
        self.avg_best_reward = 0.0
        self.num_episode_finished = 0.0
        self.avg_traj_length = 0.0
        if self.eval_base_model:
            self.success_rate_base_model = 0.0
            self.avg_episode_reward_base_model = 0.0
            self.avg_best_reward_base_model = 0.0
            self.num_episode_finished_base_model = 0.0
            self.avg_traj_length_base_model = 0.0

        while self.itr < self.n_train_itr:
            if self.itr % 1000 == 0:
                if self.itr <= self.offline_steps:
                    print(
                        f"Finished training iteration {self.itr} of {self.n_train_itr}. Offline training (total offline itrs={self.offline_steps})")
                else:
                    print(f"Finished training iteration {self.itr} of {self.n_train_itr}. Off2On training")
            # Prepare video paths for rendering
            options_venv = [{} for _ in range(self.n_envs)]
            if self.itr % self.render_freq == 0 and self.render_video:
                for env_ind in range(self.n_render):
                    options_venv[env_ind]["video_path"] = os.path.join(
                        self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
                    )
            # Set train or eval mode
            eval_mode = (
                                self.itr % self.val_freq == 0
                                and not self.force_train
                        ) or self.itr == 0
            self.model.eval() if eval_mode else self.model.train()

            if eval_mode or self.itr == 0 or self.reset_at_iteration:
                self.prev_obs_venv = self.reset_env_all(options_venv=options_venv)

            if eval_mode:
                # Evaluation
                self.evaluate(self.prev_obs_venv, mode='onestep')
                if self.eval_base_model:
                    self.evaluate(self.prev_obs_venv, mode='base_model')
            else:
                if self.itr < self.offline_steps:
                    # Offline RL.
                    n_offline = self.batch_size
                    inds_off = np.random.choice(len(obs_buffer_off), n_offline, replace=True)
                    obs_b_off = torch.from_numpy(obs_buffer_off[inds_off]).float().to(self.device)
                    actions_b_off = torch.from_numpy(action_buffer_off[inds_off]).float().to(self.device)
                    next_obs_b_off = torch.from_numpy(next_obs_buffer_off[inds_off]).float().to(self.device)
                    rewards_b_off = torch.from_numpy(reward_buffer_off[inds_off]).float().to(self.device)
                    terminated_b_off = torch.from_numpy(terminated_buffer_off[inds_off]).float().to(self.device)
                    batch = (
                        {"state": obs_b_off},
                        actions_b_off,
                        {"state": next_obs_b_off},
                        rewards_b_off,
                        terminated_b_off
                    )
                    # update agent with this batch.
                    loss_critic, loss_actor, loss_critic_info, loss_actor_info = self.agent_update(batch)
                else:
                    # Online rollout
                    for step in range(self.n_steps):
                        with torch.no_grad():
                            cond = {
                                "state": torch.from_numpy(self.prev_obs_venv["state"]).float().to(self.device)
                            }
                            samples = self.model.forward(cond=cond,
                                                         mode='onestep').cpu().numpy()  # Shape: (n_env, horizon, act_dim)
                        action_venv = samples[:, :self.act_steps]  # Shape: (n_env, act_steps, act_dim)
                        obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(action_venv)
                        done_venv = terminated_venv | truncated_venv
                        # Store transitions in online buffer
                        for i in range(self.n_envs):
                            self.obs_buffer.append(self.prev_obs_venv["state"][i])  # Shape: (obs_dim,)
                            if "final_obs" in info_venv[i]:
                                self.next_obs_buffer.append(info_venv[i]["final_obs"]["state"])
                            else:
                                self.next_obs_buffer.append(obs_venv["state"][i])
                            self.action_buffer.append(action_venv[i])  # Shape: (act_steps, act_dim)
                        self.reward_buffer.extend((reward_venv * self.scale_reward_factor).tolist())
                        self.terminated_buffer.extend(terminated_venv.tolist())

                        self.prev_obs_venv = obs_venv
                        cnt_train_step += self.n_envs * self.act_steps
                    # Sample half from offline and half from online data
                    n_offline = self.batch_size // 2
                    n_online = self.batch_size - n_offline
                    if n_online < len(self.obs_buffer):
                        # Offline sampling
                        inds_off = np.random.choice(len(obs_buffer_off), n_offline, replace=True)
                        obs_b_off = torch.from_numpy(obs_buffer_off[inds_off]).float().to(self.device)
                        next_obs_b_off = torch.from_numpy(next_obs_buffer_off[inds_off]).float().to(self.device)
                        actions_b_off = torch.from_numpy(action_buffer_off[inds_off]).float().to(self.device)
                        rewards_b_off = torch.from_numpy(reward_buffer_off[inds_off]).float().to(self.device)
                        terminated_b_off = torch.from_numpy(terminated_buffer_off[inds_off]).float().to(self.device)
                        # Online sampling
                        inds_on = np.random.choice(len(self.obs_buffer), n_online, replace=False)
                        obs_b_on = torch.from_numpy(np.array([self.obs_buffer[i] for i in inds_on])).float().to(
                            self.device)
                        next_obs_b_on = torch.from_numpy(
                            np.array([self.next_obs_buffer[i] for i in inds_on])).float().to(self.device)
                        actions_b_on = torch.from_numpy(np.array([self.action_buffer[i] for i in inds_on])).float().to(
                            self.device)
                        rewards_b_on = torch.from_numpy(np.array([self.reward_buffer[i] for i in inds_on])).float().to(
                            self.device)
                        terminated_b_on = torch.from_numpy(
                            np.array([self.terminated_buffer[i] for i in inds_on])).float().to(self.device)
                        # Combine samples
                        obs_b = torch.cat([obs_b_off, obs_b_on], dim=0)  # Shape: (batch_size, obs_dim)
                        next_obs_b = torch.cat([next_obs_b_off, next_obs_b_on], dim=0)
                        actions_b = torch.cat([actions_b_off, actions_b_on],
                                              dim=0)  # Shape: (batch_size, act_steps, act_dim)
                        rewards_b = torch.cat([rewards_b_off, rewards_b_on], dim=0)  # Shape: (batch_size,)
                        terminated_b = torch.cat([terminated_b_off, terminated_b_on], dim=0)
                        batch = (
                            {"state": obs_b},
                            actions_b,
                            {"state": next_obs_b},
                            rewards_b,
                            terminated_b,
                        )
                        # update agent with this batch.
                        loss_critic, loss_actor, loss_critic_info, loss_actor_info = self.agent_update(batch)

            # Save model periodically
            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model()

            # Log metrics
            run_results.append({"itr": self.itr, "step": cnt_train_step})
            if self.itr % self.log_freq == 0:
                time = timer()
                if eval_mode:
                    log.info(
                        f"eval (one step model): success rate {self.success_rate:8.4f} | avg episode reward {self.avg_episode_reward:8.4f} | avg best reward {self.avg_best_reward:8.4f}"
                    )
                    if self.eval_base_model:
                        log.info(
                            f"eval (base model): success rate {self.success_rate_base_model:8.4f} | avg episode reward {self.avg_episode_reward_base_model:8.4f} | avg best reward {self.avg_best_reward_base_model:8.4f}"
                        )
                    eval_log_dict = {
                        "success rate - eval": self.success_rate,
                        "avg episode reward - eval": self.avg_episode_reward,
                        "avg best reward - eval": self.avg_best_reward,
                        "num episode - eval": self.num_episode_finished,
                        "avg traj length - eval": self.avg_traj_length,
                    }
                    if self.eval_base_model:
                        eval_log_dict.update({
                            "success rate (base model) - eval": self.success_rate_base_model,
                            "avg episode reward(base model) - eval": self.avg_episode_reward_base_model,
                            "avg best reward (base model)- eval": self.avg_best_reward_base_model,
                            "num episode (base model)- eval": self.num_episode_finished_base_model,
                            "avg traj length (base model) - eval": self.avg_traj_length_base_model,
                        })
                    run_results[-1].update(eval_log_dict)
                    if self.use_wandb:
                        wandb.log(
                            eval_log_dict,
                            step=self.itr,
                            commit=False,
                        )
                else:
                    # for SAC like algorithms witih only 1 environment and collects reward for only 1 step during training, this one-step reward is somewhat meaningless. So we just don't print it.
                    log.info(
                        f"{self.itr}: step {cnt_train_step:8d} | loss actor {loss_actor:8.4f} | loss critic {loss_critic:8.4f} | t {time:8.4f}"
                    )
                    train_log_dict = {
                        "total env step": cnt_train_step,
                        "loss - critic": loss_critic,
                        "num episode - train": self.num_episode_finished,
                    }
                    if loss_actor:
                        train_log_dict["loss - actor"] = loss_actor
                    train_log_dict.update(loss_critic_info)
                    train_log_dict.update(loss_actor_info)
                    if self.use_wandb:
                        wandb.log(train_log_dict, step=self.itr, commit=True)
                    run_results[-1].update(train_log_dict)
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            self.itr += 1

    def evaluate(self, prev_obs_venv: dict, mode: str = 'onestep'):
        """evaluate onestep policy or the multistep base policy.
        """
        assert mode in ['onestep', 'base_model']
        print(f"Evaluating...")
        # Reset environments during evaluation
        firsts_trajs = np.zeros((self.n_steps_eval + 1, self.n_envs))
        firsts_trajs[0] = 1
        # Online rollout starts
        reward_trajs = np.zeros((self.n_steps_eval, self.n_envs))
        for step in range(self.n_steps_eval):
            with torch.no_grad():
                cond = {
                    "state": torch.from_numpy(prev_obs_venv["state"]).float().to(self.device)
                }
                samples = self.model.forward(cond=cond, mode=mode).cpu().numpy()  # Shape: (n_env, horizon, act_dim)
            action_venv = samples[:, :self.act_steps]  # Shape: (n_env, act_steps, act_dim)
            # Step environment
            obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(action_venv)
            done_venv = terminated_venv | truncated_venv
            reward_trajs[step] = reward_venv
            firsts_trajs[step + 1] = done_venv
            prev_obs_venv = obs_venv
        # Compute episode rewards
        episodes_start_end = []
        for env_ind in range(self.n_envs):
            env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
            for i in range(len(env_steps) - 1):
                start, end = env_steps[i], env_steps[i + 1]
                if end - start > 1:
                    episodes_start_end.append((env_ind, start, end - 1))
        if episodes_start_end:
            reward_trajs_split = [reward_trajs[start:end + 1, env_ind] for env_ind, start, end in episodes_start_end]
            num_episode_finished = len(reward_trajs_split)
            episode_reward = np.array([np.sum(traj) for traj in reward_trajs_split])
            episode_best_reward = np.array([np.max(traj) / self.act_steps for traj in reward_trajs_split])
            avg_episode_reward = np.mean(episode_reward)
            avg_best_reward = np.mean(episode_best_reward)
            success_rate = np.mean(episode_best_reward >= self.best_reward_threshold_for_success)
            episode_lengths = np.array([end - start + 1 for _, start, end in episodes_start_end]) * self.act_steps
            avg_traj_length = np.mean(episode_lengths) if len(episode_lengths) > 0 else 0
        else:
            num_episode_finished = 0
            avg_episode_reward = avg_best_reward = success_rate = 0
            avg_traj_length = 0

        # record
        if mode == 'onestep':
            self.avg_episode_reward = avg_episode_reward
            self.avg_best_reward = avg_best_reward
            self.success_rate = success_rate
            self.num_episode_finished = num_episode_finished
            self.avg_traj_length = avg_traj_length
        elif mode == 'base_model':
            self.avg_episode_reward_base_model = avg_episode_reward
            self.avg_best_reward_base_model = avg_best_reward
            self.success_rate_base_model = success_rate
            self.num_episode_finished_base_model = num_episode_finished
            self.avg_traj_length_base_model = avg_traj_length
        else:
            raise ValueError(f"unsupported mode={mode}. A valid choice must be in ['onstep', 'base_model']")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="Pusher-v4")
