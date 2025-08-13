import numpy as np
import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """正弦位置嵌入模块，用于时间步编码"""
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0, "time embedding dim must be even"
        self.dim = dim

    def forward(self, t):
        # t: (B,) or (B,1)
        if t.dim() == 2 and t.shape[1] == 1:
            t = t.view(-1)
        t = t.view(-1).to(torch.float32)
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-torch.arange(half, device=device, dtype=torch.float32) * (np.log(10000.0) / half))
        args = t.unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb


class MLP(nn.Module):
    """基础多层感知机实现"""
    def __init__(self, dims, activation_type="ReLU", out_activation_type="Identity"):
        super().__init__()
        layers = []
        n = len(dims) - 1
        for i in range(n):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            is_last = (i == n - 1)
            act = out_activation_type if is_last else activation_type
            if hasattr(nn, act):
                layers.append(getattr(nn, act)())
            else:
                layers.append(nn.ReLU() if not is_last else nn.Identity())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)