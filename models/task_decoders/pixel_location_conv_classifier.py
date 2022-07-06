"""
Given the representation of the point (x, y, z), decode what the semantic tag of that position
should be.
"""
from typing import Dict, Tuple, Union
import einops
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.task_decoders.abstract_decoder import AbstractDecoder
from utils.mlp import MLP


class PixelLocationConvSoftmaxDecoder(AbstractDecoder):
    def __init__(
        self,
        representation_length: int,
        depth: int = 3,
        hidden_dim: int = 256,
        batchnorm: bool = True,
        image_size: int = 224,
        subset_grid_size: int = 32,  # The regularity with which points are sampled.
        device: Union[str, torch.device] = "cuda",
        lam: float = 1.0,
        coord_range: float = 10.0,
        num_bins: int = 500,
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
        self.loss = nn.CrossEntropyLoss()
        self.lam = lam

        # Divide [-coord range, coord range] into num_bins bins.
        self.num_bins = num_bins
        self.coord_range = coord_range

    def register_embedding_map(self, embedding: nn.Embedding) -> None:
        super().register_embedding_map(embedding)
        # Create an MLP to map the representation to the same length as the embedding dict.
        self.trunk = nn.Sequential(
            Rearrange("b n (c h w) -> (b n) c h w", h=1, w=1),
            nn.ConvTranspose2d(
                in_channels=self._rep_length, out_channels=64, kernel_size=4
            ),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ConvTranspose2d(
                in_channels=64, out_channels=3 * self.num_bins, kernel_size=2
            ),
        )
        self.trunk = self.trunk.to(self._device)

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
        rearranged_xyz_subset = einops.rearrange(xyz_subset, "... d n m -> ... n m d")
        # Convert xyz_world to class labels.
        discretized_xyz_subset = torch.floor(
            self.num_bins
            * (rearranged_xyz_subset + self.coord_range)
            / (2.0 * self.coord_range)
        ).long()
        # Now reshape the view representation to the right shape.
        per_class_logits = einops.rearrange(
            decoded_view_representation, "... (bins d) n m -> ... n m bins d", d=3
        )
        # Flatten all but last dim
        flattened_labels = torch.flatten(discretized_xyz_subset, end_dim=-2)
        flattend_preds = torch.flatten(per_class_logits, end_dim=-3)
        position_loss = self.lam * self.loss(flattend_preds, flattened_labels)
        # Find out the actual distance.
        with torch.no_grad():
            chosen_class_label = torch.argmax(per_class_logits, dim=-2)
            # Convert label to actual pred.
            predicted_xyz = (
                chosen_class_label * 2 * self.coord_range / self.num_bins
            ) - self.coord_range
            # Now calculate the distance.
            avg_distance = F.mse_loss(predicted_xyz, rearranged_xyz_subset) ** 0.5
        # Calculate the predicted distance as well.
        return position_loss, dict(
            position_loss=position_loss,
            distance=avg_distance,
        )
