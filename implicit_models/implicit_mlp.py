import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.mlp import MLP
from models.scene_models.positional_encoding import FourierFeatures



class ImplicitSurfaceModel(nn.Module):
    def __init__(
        self,
        depth: int = 3,
        width: int = 512,
        batchnorm: bool = False,
        fourier_dim: int = 256,
    ):
        super().__init__()
        self.fourier_proj = FourierFeatures(
            input_dim=3,
            fourier_embedding_dim=fourier_dim,
            fourier_embedding_scale=1 / (2.0 ** 4),
        )
        self.trunk = MLP(
            input_dim=fourier_dim,
            output_dim=1,
            batchnorm=batchnorm,
            hidden_depth=depth,
            hidden_dim=width,
        )

    def to(self, device):
        self.fourier_proj = self.fourier_proj.to(device)
        self.trunk = self.trunk.to(device)
        return self

    def forward(self, x: torch.Tensor):
        return self.trunk(self.fourier_proj(x))


class ImplicitCLIPModel(nn.Module):
    def __init__(
        self,
        depth: int = 6,
        width: int = 2048,
        batchnorm: bool = False,
        use_camera_dir: bool = False,
        fourier_dim: int = 256,
        clip_dim: int = 512,
    ):
        super().__init__()
        input_dim = 3
        if use_camera_dir:
            input_dim += 4
        self.fourier_proj = FourierFeatures(
            input_dim=input_dim,
            fourier_embedding_dim=fourier_dim,
            fourier_embedding_scale=1 / (2.0 ** 4),
        )
        self.trunk_1 = MLP(
            input_dim=fourier_dim,
            output_dim=width,
            batchnorm=batchnorm,
            hidden_depth=depth // 2,
            hidden_dim=width,
        )
        self.trunk_2 = MLP(
            input_dim=width + fourier_dim,
            output_dim=2 * clip_dim,
            batchnorm=batchnorm,
            hidden_depth=depth // 2,
            hidden_dim=width,
        )
        # Magic value adviced by @imisra
        self.temperature = nn.Parameter(torch.log(torch.tensor(1.0 / 0.07)))

    def to(self, device):
        self.fourier_proj = self.fourier_proj.to(device)
        self.trunk_1 = self.trunk_1.to(device)
        self.trunk_2 = self.trunk_2.to(device)
        self.temperature.data = self.temperature.data.to(device)
        return self

    def forward(self, x: torch.Tensor):
        projected_x = self.fourier_proj(x)
        joint_output = self.trunk_2(
            torch.cat([self.trunk_1(projected_x), projected_x], dim=-1)
        )
        label_latent, image_latent = torch.chunk(joint_output, chunks=2, dim=-1)
        return label_latent, image_latent

    def compute_loss(
        self, predicted_latents, actual_latents, label_mask=None, weights=None
    ):
        temp = torch.exp(self.temperature)
        sim = torch.einsum("i d, j d -> i j", predicted_latents, actual_latents) * temp
        # Zero out the cells where the labels are same.
        if label_mask is not None:
            sim = sim * label_mask
            del label_mask
        labels = torch.arange(len(predicted_latents), device=predicted_latents.device)
        if weights is None:
            loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
        else:
            loss = (
                F.cross_entropy(sim, labels, reduction="none")
                + F.cross_entropy(sim.t(), labels, reduction="none")
            ) / 2
            loss = (loss * weights).mean()
        return loss
