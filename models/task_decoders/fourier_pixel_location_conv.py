"""
Given the representation of the point (x, y, z), decode what the semantic tag of that position
should be.
"""
from typing import Dict, Tuple, Union
import einops
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn

from models.task_decoders.abstract_decoder import AbstractDecoder
from models.scene_models.positional_encoding import FourierFeatures


class FourierPixelLocationConvDecoder(AbstractDecoder):
    def __init__(
        self,
        representation_length: int,
        depth: int = 3,
        hidden_dim: int = 256,
        batchnorm: bool = True,
        image_size: int = 224,
        subset_grid_size: int = 32,  # The regularity with which points are sampled.
        device: Union[str, torch.device] = "cuda",
        lam: float = 10.0,
    ):
        super().__init__()
        self._rep_length = representation_length
        self._depth = depth
        self._hidden_dim = hidden_dim
        self._batchnorm = batchnorm
        self._device = device
        self._image_size = image_size
        self._subset_grid = subset_grid_size
        self._grid_size = self._image_size // self._subset_grid
        self.loss = nn.SmoothL1Loss()
        self.lam = lam

        self._fourier_feature_mapping = FourierFeatures(
            input_dim=3, fourier_embedding_dim=256
        )

    def register_embedding_map(self, embedding: nn.Embedding) -> None:
        super().register_embedding_map(embedding)
        # Create an MLP to map the representation to the same length as the embedding dict.
        self.trunk = nn.Sequential(
            Rearrange("b n (c h w) -> (b n) c h w", h=1, w=1),
            nn.ConvTranspose2d(
                in_channels=self._rep_length, out_channels=64, kernel_size=4
            ),
            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ConvTranspose2d(in_channels=128, out_channels=256, kernel_size=2),
        )
        self.trunk = self.trunk.to(self._device)
        self._fourier_feature_mapping = self._fourier_feature_mapping.to(self._device)

    def decode_representations(
        self, view_reps: torch.Tensor, scene_model_reps: torch.Tensor
    ) -> torch.Tensor:
        # We learn to extract the semantic tag of the position.
        decoded_xyz = self.trunk(view_reps)
        return decoded_xyz, scene_model_reps

    def compute_detailed_loss(
        self,
        decoded_view_representation: torch.Tensor,
        decoded_representation: torch.Tensor,
        ground_truth: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # First, generate the world XYZ coordinates from the pixels.
        _ = decoded_representation
        xyz_world = ground_truth[0]["xyz_position"]
        xyz_subset = xyz_world[..., :: self._subset_grid, :: self._subset_grid]
        fourier_xyz = self._fourier_feature_mapping(
            einops.rearrange(xyz_subset, "b c h w -> b h w c")
        )
        reshaped_fourier = einops.rearrange(fourier_xyz, "b h w d -> b d h w")
        position_loss = self.lam * self.loss(
            decoded_view_representation, reshaped_fourier
        )
        return position_loss, dict(
            position_loss=position_loss, distance=(position_loss * 2 / self.lam) ** 0.5
        )
