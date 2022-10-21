import torch
from torch import nn


class Block(nn.Module):
    def __int__(self, dim, dim_out, groups=8, activation='leakyrelu'):
        super().__init__()
        assert activation in ['leakyrelu', 'tanh']
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            nn.LeakyReLU(0.2) if activation == "leakyrelu" else nn.Tanh()
        )

    def forward(self, x):
        return self.block(x)
