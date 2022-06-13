from typing import Callable
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: Callable) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_features=dim, out_features=hidden_dim),
            nn.Dropout(p=dropout),
            nn.GELU(),
            nn.Linear(in_features=hidden_dim, out_features=dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)


class Attention(nn.Module):
    def __init__(
        self, dim: int, heads: int = 8, dim_heads: int = 64, dropout: float = 0.0
    ) -> None:
        super().__init__()
        inner_dim = dim_heads * heads
        project_out = not (heads == 1 and dim_heads == dim)

        self.heads = heads
        self.scale = dim_heads**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(
                nn.Linear(in_features=inner_dim, out_features=dim),
                nn.Dropout(p=dropout),
            )
            if project_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            Attention(
                                dim=dim,
                                heads=heads,
                                dim_heads=dim_heads,
                                dropout=dropout,
                            )
                        ),
                        PreNorm(
                            FeedForward(dim=dim, hidden_dim=mlp_dim, dropout=dropout)
                        ),
                    ]
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = x + attn(x)
            x = x + ff(x)
        return x
