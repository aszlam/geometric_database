"""
Given the representation of the point (x, y, z), decode what the semantic tag of that position
should be.
"""
from typing import Dict, Tuple, Union
import einops
import torch
import torch.nn as nn

from models.task_decoders.abstract_decoder import AbstractDecoder
from utils.mlp import MLP


class PixelLocationDecoder(AbstractDecoder):
    def __init__(
        self,
        representation_length: int,
        depth: int = 3,
        hidden_dim: int = 256,
        batchnorm: bool = True,
        image_size: int = 224,
        subset_grid_size: int = 32,  # The regularity with which points are sampled.
        device: Union[str, torch.device] = "cuda",
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

    def register_embedding_map(self, embedding: nn.Embedding) -> None:
        super().register_embedding_map(embedding)
        # Create an MLP to map the representation to the same length as the embedding dict.
        self.trunk = MLP(
            input_dim=self._rep_length,
            hidden_dim=self._hidden_dim,
            output_dim=3 * (self._grid_size) ** 2,
            hidden_depth=self._depth,
            batchnorm=self._batchnorm,
        )
        self.trunk = self.trunk.to(self._device)

    def decode_representations(
        self, view_reps: torch.Tensor, scene_model_reps: torch.Tensor
    ) -> torch.Tensor:
        # We learn to extract the semantic tag of the position.
        decoded_xyz = self.trunk(view_reps.squeeze(0))
        decoded_full_xyz_array = einops.rearrange(
            decoded_xyz,
            "... (d h w) -> ... d h w",
            h=self._grid_size,
            w=self._grid_size,
            d=3,
        )
        return decoded_full_xyz_array, scene_model_reps

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
        position_loss = self.loss(decoded_view_representation, xyz_subset)
        return position_loss, dict(position_loss=position_loss)
