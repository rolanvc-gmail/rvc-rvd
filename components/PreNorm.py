import torch
from torch import nn
import LayerNorm


class PreNorm(nn.Module):
    def __int__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)