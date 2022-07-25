from gridencoder import GridEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.mlp import MLP


class GridSurfaceModel(nn.Module):
    def __init__(
        self,
        mlp_depth: int = 2,
        mlp_width: int = 256,
        batchnorm: bool = False,
        log2_hashmap_size: int = 19,
        device: str = "cuda",
        bounds: float = 11.0,
    ):
        super().__init__()
        self._grid_model = GridEncoder(
            input_dim=3,
            num_levels=16,
            level_dim=2,
            per_level_scale=2,
            base_resolution=16,
            log2_hashmap_size=log2_hashmap_size,
            desired_resolution=None,
            gridtype="hash",
            align_corners=False,
        )
        # Now convert the output with an MLP
        self._post_grid = MLP(
            input_dim=16 * 2,
            hidden_dim=mlp_width,
            hidden_depth=mlp_depth,
            output_dim=1,
            batchnorm=batchnorm,
        )
        self._grid_model = self._grid_model.to(device)
        self._post_grid = self._post_grid.to(device)

    def forward(self, x: torch.Tensor, bounds: float = 11.0):
        grid_hash = self._grid_model(x, bounds)
        result = self._post_grid(grid_hash)
        return result


class GridCLIPModel(nn.Module):
    def __init__(
        self,
        mlp_depth: int = 2,
        mlp_width: int = 256,
        batchnorm: bool = False,
        log2_hashmap_size: int = 16,
        device: str = "cuda",
        clip_dim: int = 512,
        bounds: float = 11.0,
    ):
        super().__init__()
        self._grid_model = GridEncoder(
            input_dim=3,
            num_levels=16,
            level_dim=2,
            per_level_scale=2,
            base_resolution=16,
            log2_hashmap_size=log2_hashmap_size,
            desired_resolution=None,
            gridtype="hash",
            align_corners=False,
        )
        # Now convert the output with an MLP
        self._post_grid = MLP(
            input_dim=16 * 2,
            hidden_dim=mlp_width,
            hidden_depth=mlp_depth,
            output_dim=2 * clip_dim,
            batchnorm=batchnorm,
        )
        # Mini MLP for extra storage for image loss
        self._image_head = MLP(
            input_dim=clip_dim,
            hidden_dim=clip_dim,
            hidden_depth=1,
            output_dim=clip_dim,
            batchnorm=batchnorm,
        )
        # Magic value adviced by @imisra
        self.temperature = nn.Parameter(torch.log(torch.tensor(1.0 / 0.07)))

        self._grid_model = self._grid_model.to(device)
        self._post_grid = self._post_grid.to(device)
        self._image_head = self._image_head.to(device)
        self.temperature.data = self.temperature.data.to(device)

    def forward(self, x: torch.Tensor, bounds: float = 11.0):
        grid_hash = self._grid_model(x, bounds)
        result = self._post_grid(grid_hash)
        label_latent, image_latent = torch.chunk(result, chunks=2, dim=-1)
        image_latent = self._image_head(image_latent)
        return label_latent, image_latent

    def to(self, device):
        self._grid_model = self._grid_model.to(device)
        self._post_grid = self._post_grid.to(device)
        self._image_head = self._image_head.to(device)
        self.temperature.data = self.temperature.data.to(device)
        return self

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
