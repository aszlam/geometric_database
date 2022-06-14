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
        d = fourier_embedding_dim
        alpha = fourier_embedding_scale
        self.B = 2 * math.pi * alpha * torch.randn(input_dim, d)
        self.B.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        assumes B x input_dim input
        """
        return torch.cat([torch.cos(x @ self.B), torch.sin(x @ self.B)], 1)

    def cuda(self, device=None):
        self.B = self.B.cuda(device=device)
        self.B.requires_grad = False
        return self

    def cpu(self):
        self.B = self.B.cpu()
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
    ):
        super().__init__()
        self.layer_1 = nn.Linear(coordinate_dim, hidden_dim)
        self.gelu = F.gelu()
        self.layer_2 = nn.Linear(
            hidden_dim, fourier_input_dim if fourier_features else representation_dim
        )
        self.fourier_proj = (
            FourierFeatures(
                input_dim=fourier_input_dim, fourier_embedding_dim=representation_dim
            )
            if fourier_features
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gelu(self.layer_1(x))
        return self.fourier_proj(self.layer_2(x))
