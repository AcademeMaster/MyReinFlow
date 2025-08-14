# MIT License

# Copyright (c) 2025 ReinFlow Authors

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



"""
1-Rectified Flow Policy with Mean Flow Support
"""


import logging
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from torch.autograd.functional import jvp
from collections import namedtuple
from reflow.networks.flow_mlp import FlowMLP

log = logging.getLogger(__name__)
Sample = namedtuple("Sample", "trajectories chains")



class MeanFlow(nn.Module):
    """Mean Flow implementation for one-step generation based on average velocity field.
    
    The key difference from regular flow is that MeanFlow learns an average velocity field
    that can perform transformation in fewer steps, potentially even one step.
    """
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
        """Initialize the MeanFlow model with specified parameters.

        Args:
            network: FlowMLP network for mean velocity prediction.
            device: Device to run the model on (e.g., 'cuda' or 'cpu').
            horizon_steps: Number of steps in the trajectory horizon.
            action_dim: Dimension of the action space.
            act_min: Minimum action value for clipping.
            act_max: Maximum action value for clipping.
            obs_dim: Dimension of the observation space.
            max_denoising_steps: Maximum number of denoising steps for sampling.
            seed: Random seed for reproducibility.
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

    def sample_time_pair(self, batch_size: int, time_sample_type: str = 'uniform', **kwargs) -> tuple[Tensor, Tensor]:
        """Sample time pairs (t, r) where t >= r for mean flow training.

        Args:
            batch_size: Number of time samples to generate.
            time_sample_type: Type of distribution ('uniform', 'logitnormal', 'beta').
            **kwargs: Additional parameters for non-uniform distributions.

        Returns:
            tuple: (t, r) tensors of shape (batch_size, 1) where t >= r.
        """
        if time_sample_type == 'uniform':
            # Sample from logit-normal distribution like in the demo
            if torch.rand(1).item() < 1.0:
                samples = torch.sigmoid(torch.normal(-0.4, 1.0, (batch_size, 2), device=self.device))
                # enforce t > r
                t = torch.max(samples[:, 0], samples[:, 1]).unsqueeze(1)
                r = torch.min(samples[:, 0], samples[:, 1]).unsqueeze(1)
            else:
                t = torch.sigmoid(torch.normal(-0.4, 1.0, (batch_size, 1), device=self.device))
                r = t.clone()
        elif time_sample_type == 'logitnormal':
            m = kwargs.get("m", -0.4)
            s = kwargs.get("s", 1.0)
            samples = torch.sigmoid(torch.normal(m, s, (batch_size, 2), device=self.device))
            t = torch.max(samples[:, 0], samples[:, 1]).unsqueeze(1)
            r = torch.min(samples[:, 0], samples[:, 1]).unsqueeze(1)
        else:
            # Fallback to uniform sampling
            samples = torch.rand(batch_size, 2, device=self.device)
            t = torch.max(samples[:, 0], samples[:, 1]).unsqueeze(1)
            r = torch.min(samples[:, 0], samples[:, 1]).unsqueeze(1)
            
        return t, r

    def generate_target_mean(self, x1: Tensor) -> tuple:
        """Generate training targets for the mean velocity field.

        Args:
            x1: Real data tensor of shape (batch_size, horizon_steps, action_dim).

        Returns:
            tuple: Contains (zt, t, r, obs) and u_tgt where:
                - zt: Interpolated data tensor of shape (batch_size, horizon_steps, action_dim).
                - t: Time step tensor of shape (batch_size, 1).
                - r: Reference time tensor of shape (batch_size, 1).
                - u_tgt: Target mean velocity tensor of shape (batch_size, horizon_steps, action_dim).
        """
        batch_size = x1.shape[0]
        t, r = self.sample_time_pair(batch_size, self.sample_t_type)
        
        # Sample noise
        x0 = torch.randn(x1.shape, dtype=torch.float32, device=self.device)
        
        # Generate interpolated point
        t_expanded = t.view(batch_size, 1, 1).expand_as(x1)
        zt = (1 - t_expanded) * x1 + t_expanded * x0
        
        # Compute instantaneous velocity
        v = x0 - x1
        
        return (zt, t, r), v

    def loss_mean(self, zt: Tensor, t: Tensor, r: Tensor, obs: dict, v: Tensor) -> Tensor:
        """Compute the mean flow loss using JVP to get velocity derivative.

        Args:
            zt: Interpolated data tensor of shape (batch_size, horizon_steps, action_dim).
            t: Time step tensor of shape (batch_size, 1).
            r: Reference time tensor of shape (batch_size, 1).
            obs: Dictionary containing 'state' tensor of shape (batch_size, cond_steps, obs_dim).
            v: Instantaneous velocity tensor of shape (batch_size, horizon_steps, action_dim).

        Returns:
            Tensor: Mean squared error loss.
        """
        # Define the function for JVP computation
        def velocity_func(z, r_param, t_param):
            # Reshape t_param to match network input format
            t_flat = t_param.squeeze(-1) if t_param.dim() > 1 else t_param
            return self.network(z, t_flat, obs)
        
        # Compute JVP: u and du/dt
        u, dudt = jvp(
            func=velocity_func,
            inputs=(zt, r, t),
            v=(v, torch.zeros_like(r), torch.ones_like(t)),
            create_graph=True
        )
        
        # Extract du/dt component (it's a tuple of gradients)
        if isinstance(dudt, tuple):
            dudt = dudt[2]  # The gradient w.r.t. t (third input)
        
        # Compute target mean velocity: u_tgt = v - (t - r) * du/dt
        time_diff = (t - r).view(t.shape[0], 1, 1).expand_as(v)
        u_tgt = v - time_diff * dudt
        u_tgt = u_tgt.detach()
        
        # Compute predicted velocity
        t_flat = t.squeeze(-1) if t.dim() > 1 else t
        predicted_velocity = self.network(zt, t_flat, obs)
        
        # Compute MSE loss
        loss = F.mse_loss(input=predicted_velocity, target=u_tgt)
        

        
        return loss

    @torch.no_grad()
    def sample_mean_flow(
        self,
        cond: dict,
        inference_steps: int = 1,
        record_intermediate: bool = False,
        clip_intermediate_actions: bool = True,
        z: torch.Tensor | None = None
    ) -> Sample:
        """Sample trajectories using the learned mean velocity field.
        
        The key advantage of mean flow is that it can generate good samples
        with very few steps (even 1 step).

        Args:
            cond: Dictionary containing 'state' tensor of shape (batch_size, cond_steps, obs_dim).
            inference_steps: Number of denoising steps (can be as low as 1).
            record_intermediate: Whether to return intermediate predictions.
            clip_intermediate_actions: Whether to clip actions to act_range.
            z: Optional initial noise tensor.

        Returns:
            Sample: Named tuple with 'trajectories' (and 'chains' if record_intermediate).
        """
        B = cond['state'].shape[0]
        x_hat_list = None
        if record_intermediate:
            x_hat_list = torch.zeros((inference_steps, B) + self.data_shape, device=self.device)
        
        # Initialize with noise
        x_hat = z if z is not None else torch.randn((B,) + self.data_shape, device=self.device)
        
        # For mean flow, we use backwards sampling
        dt = 1.0 / inference_steps
        
        for i in range(inference_steps, 0, -1):
            # Compute current time points
            r = torch.full((B,), (i-1) * dt, device=self.device)
            t = torch.full((B,), i * dt, device=self.device)
            
            # Get mean velocity
            velocity = self.network(x_hat, t, cond)
            
            # Update using mean velocity (backwards sampling)
            x_hat = x_hat - velocity * dt
            
            if clip_intermediate_actions or i == 1:  # always clip the final output
                x_hat = x_hat.clamp(*self.act_range)
                
            if record_intermediate and x_hat_list is not None:
                x_hat_list[inference_steps - i] = x_hat
        
        return Sample(trajectories=x_hat, chains=x_hat_list if record_intermediate else None)

    @torch.no_grad()
    def sample_one_step(
        self,
        cond: dict,
        z: torch.Tensor | None = None
    ) -> Tensor:
        """One-step generation using mean flow.
        
        This is the main advantage of mean flow - direct generation in one step.

        Args:
            cond: Dictionary containing 'state' tensor of shape (batch_size, cond_steps, obs_dim).
            z: Optional initial noise tensor.

        Returns:
            Tensor: Generated trajectories of shape (batch_size, horizon_steps, action_dim).
        """
        B = cond['state'].shape[0]
        
        # Initialize with noise
        x_hat = z if z is not None else torch.randn((B,) + self.data_shape, device=self.device)
        
        # One-step generation: from t=1 to t=0 (r=0)
        t = torch.ones(B, device=self.device)
        velocity = self.network(x_hat, t, cond)
        
        # Direct transformation
        x_hat = x_hat - velocity
        
        # Clip to action range
        x_hat = x_hat.clamp(*self.act_range)
        
        return x_hat