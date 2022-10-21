import torch
from torch import nn
from einops import rearrange

class LinearAttention(nn.Module):

    def __int__(self, dim, heads=4, dim_head=16):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim*3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h x) x y -> b h c(x y)", h=self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim=-1)
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n_> b h e n", context, q)
        out = rearrange(out, "b h c (x y)-> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)




