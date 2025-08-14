import math
import torch
from torch import nn

class SinusoidalPosEmb(nn.Module):
    """
    Standard sinusoidal positional embedding as used in diffusion/transformer models.
    Inputs: tensor of shape (B, 1) or (B,) with values in [0,1] (time or step index normalized).
    Returns: embedding of shape (B, dim)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x[:, None]
        device = x.device
        half_dim = self.dim // 2
        emb_factor = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_factor)
        emb = x * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            # pad one dim if odd
            emb = torch.nn.functional.pad(emb, (0,1))
        return emb
