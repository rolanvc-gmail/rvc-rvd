import torch
from torch import nn


class Downsample(nn.Module):
    def __int__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)
