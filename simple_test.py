import torch
import torch.nn as nn
import numpy as np
import math

print("Testing basic imports...")

# Test basic PyTorch functionality
device = torch.device("cpu")
x = torch.randn(4, 10)
print(f"PyTorch test successful: {x.shape}")

# Test the FlowMLP component from the main file
class SinusoidalPosEmb(nn.Module):
    """Sinusoidal position embedding for time encoding."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# Test SinusoidalPosEmb
time_embedding = SinusoidalPosEmb(16)
t = torch.randn(4, 1)
emb = time_embedding(t)
print(f"Time embedding test successful: {emb.shape}")

print("All basic tests passed!")
