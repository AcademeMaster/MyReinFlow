"""
MyReinFlow - Rectified Flow for Reinforcement Learning

A modular implementation of rectified flow models with applications to 
reinforcement learning, including behavior cloning and policy optimization.

## Module Structure

- `flows`: Core flow model implementations (ReFlow, MeanFlow)
- `networks`: Neural network architectures (MLP, Transformer, etc.)  
- `models`: Probabilistic models (Gaussian, GMM, Critic)
- `algorithms`: RL algorithms using flow models (FQL, PPO)
- `utils`: Utility functions and helpers

## Quick Start

```python
from reflow.flows import ReFlow, MeanFlow
from reflow.networks import FlowMLP
from reflow.algorithms import FQLModel
from reflow.models import CriticObsAct

# Create a flow network
flow_net = FlowMLP(horizon_steps=4, action_dim=3, cond_dim=24)

# Create ReFlow model  
reflow = ReFlow(network=flow_net, device='cuda', ...)

# Use in FQL algorithm
fql = FQLModel(bc_flow=reflow, actor=..., critic=...)
```
"""

# Core flow models
from .flows import ReFlow, MeanFlow

# Main network architectures
from .networks import FlowMLP, MLP, ResidualMLP

# Key RL components
from .algorithms import FQLModel, OneStepActor, PPOFlow
from .models import CriticObsAct

__version__ = "0.2.0"

__all__ = [
    # Flow models
    "ReFlow",
    "MeanFlow", 
    
    # Networks
    "FlowMLP",
    "MLP", 
    "ResidualMLP",
    
    # RL algorithms
    "FQLModel",
    "OneStepActor", 
    "PPOFlow",
    
    # Models
    "CriticObsAct"
]
