from torch import nn
import math


class Exp(nn.Module):
    def __int__(self):
        super().__init__()

    def forward(self, x):
        return x.exp().clamp(math.sqrt(2), 20*math.sqrt(2))
