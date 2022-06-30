from typing import Union
import torch
from torch import nn
from torch.nn import functional as F
import math


class FourierFeatures(nn.Module):
    """
    Project inputs to randomized fourier features.
    """

    def __init__(
        self,
        input_dim: int,
        fourier_embedding_dim: int = 256,
        fourier_embedding_scale: float = 1.0,
    ):
        super().__init__()
        assert (
            fourier_embedding_dim % 2 == 0
        ), "Fourier dim is not divisible by 2, can't be evenly distributed between sin and cos"
        d = fourier_embedding_dim // 2
        alpha = fourier_embedding_scale
        self.B = 2 * math.pi * alpha * torch.randn(input_dim, d)
        self.B.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        assumes ... x input_dim input
        """
        return torch.cat([torch.cos(x @ self.B), torch.sin(x @ self.B)], -1)

    def to(self, device: Union[str, torch.device]):
        self.B = self.B.to(device)
        self.B.requires_grad = False
        return self


class PositionalEmbedding(nn.Module):
    """
    Positional embedding class, which takes in either a euclidean position or quaternion
    direction information, and converts it to additive features that are passed in to our
    scene network.

    Optionally, we use Fourier features to project it to a higher sine-cosine dimension.
    """

    def __init__(
        self,
        coordinate_dim: int = 3,
        hidden_dim: int = 64,
        representation_dim: int = 256,
        fourier_features: bool = True,
        fourier_input_dim: int = 64,
        device: Union[str, torch.device] = "cuda",
    ):
        super().__init__()
        self.layer_1 = nn.Linear(coordinate_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.layer_2 = nn.Linear(
            hidden_dim, fourier_input_dim if fourier_features else representation_dim
        )
        self.mlp = nn.Sequential(self.layer_1, self.gelu, self.layer_2)
        self.fourier_proj = (
            FourierFeatures(
                input_dim=fourier_input_dim, fourier_embedding_dim=representation_dim
            )
            if fourier_features
            else nn.Identity()
        )
        self.mlp = self.mlp.to(device)
        self.fourier_proj = self.fourier_proj.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fourier_proj(self.mlp(x))
