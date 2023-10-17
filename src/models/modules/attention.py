import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class LinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32, dropout: float = 0.0, rescale: str = "qk"):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Sequential(nn.Dropout(dropout), nn.Conv2d(dim, hidden_dim * 3, 1, bias=False))
        assert rescale in ["qk", "qkv"]
        self.rescale = getattr(self, f"rescale_{rescale}")
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)
        # nn.Sequential(
        #     nn.Conv2d(hidden_dim, dim, 1),
        #     nn.Dropout(dropout)
        # )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)

        q, k, v = self.rescale(q, k, v, h=h, w=w)
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)

    def rescale_qk(self, q, k, v, h, w):
        q = q * self.scale
        k = k.softmax(dim=-1)
        return q, k, v

    def rescale_qkv(self, q, k, v, h, w):
        q = q.softmax(dim=-2)
        q = q * self.scale
        k = k.softmax(dim=-1)
        v = v / (h * w)
        return q, k, v


def l2norm(t):
    return F.normalize(t, dim=-1)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32, dropout: float = 0.0):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)

        q = q * self.scale

        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)
