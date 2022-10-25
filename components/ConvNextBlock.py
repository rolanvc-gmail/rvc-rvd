import torch
from torch import nn
from utils import exists
from components import LayerNorm
from einops import rearrange


class ConvNetBlock(nn.Module):
    def __int__(self, dim, dim_out, time_emb_dim=None, mult=2, norm=True):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim)) if exists(time_emb_dim) else None
        )
        self.dc_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        self.net = nn.Sequential(
            LayerNorm(dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out*mult, 1),
            nn.GELU(),
            LayerNorm(dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 1),
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim!=dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)

        if exists(self.mlp):
            assert exists(time_emb), "time emb must be passed in"
            condition = self.mlp(time_emb)
            h = h+rearrange(condition, " b c -> b c 1 1")

        h = self.net(h)
        return h + self.res_conv(x)

