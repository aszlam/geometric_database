"""
Given the representation of the point (x, y, z), decode what the semantic tag of that position
should be.
"""
from typing import Dict, Tuple, Union
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.task_decoders.abstract_decoder import AbstractDecoder
from utils.mlp import MLP
from utils import quaternion_to_matrix


class PixelLocationCameraDecoder(AbstractDecoder):
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
        use_log_for_loss: float = True,
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
        # self.loss = nn.MSELoss()
        self.lam = lam
        self._loss_transform = torch.log if use_log_for_loss else lambda x: x

    def register_embedding_map(self, embedding: nn.Embedding) -> None:
        super().register_embedding_map(embedding)
        # Create an MLP to map the representation to the same length as the embedding dict.
        self.trunk = MLP(
            input_dim=self._rep_length,
            hidden_dim=self._hidden_dim,
            output_dim=3 + 4,  # Just regress camera parameters
            hidden_depth=self._depth,
            batchnorm=self._batchnorm,
        )
        self.trunk = self.trunk.to(self._device)

    def decode_representations(
        self, view_reps: torch.Tensor, scene_model_reps: torch.Tensor
    ) -> torch.Tensor:
        # We learn to extract the semantic tag of the position.
        mean_over_image = torch.mean(view_reps, dim=[-1, -2])
        decoded_xyz = self.trunk(mean_over_image)
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
        # position_loss = self.lam * self.loss(decoded_view_representation, xyz_subset)
        regression_loss = self.lam * self._loss_transform(
            self.loss(
                decoded_view_representation,
                torch.cat(
                    [
                        ground_truth[0]["camera_pos"],
                        ground_truth[0]["camera_direction"],
                    ],
                    dim=1,
                ),
            )
        )
        # Now calculate the distance by using some math.
        decoded_translation, decoded_rotation = (
            decoded_view_representation[..., :3],
            decoded_view_representation[..., 3:],
        )
        decoded_rotation_matrix = quaternion_to_matrix(
            F.normalize(decoded_rotation, dim=-1)
        )
        # Now figure out the transformation
        transformation_matrix = torch.cat(
            [decoded_rotation_matrix, decoded_translation.unsqueeze(-1)], dim=-1
        )
        # And to transform the local to global.
        local_coords_and_one = torch.cat(
            [
                einops.rearrange(
                    ground_truth[0]["local_xyz_position"][
                        ..., :: self._subset_grid, :: self._subset_grid, :
                    ],
                    "b w h d -> b (w h) d",
                ),
                torch.ones(
                    (
                        len(decoded_view_representation),
                        self._grid_size**2,
                        1,
                    ),
                    device=self._device,
                ),
            ],
            dim=-1,
        )
        local_coord_transpose = einops.rearrange(local_coords_and_one, "b n d -> b d n")
        projected_coords_transpose = torch.bmm(
            transformation_matrix, local_coord_transpose
        )
        projected_coords = einops.rearrange(
            projected_coords_transpose, "b d n -> b n d"
        )
        position_loss = self.loss(
            projected_coords, einops.rearrange(xyz_subset, "b d w h -> b (w h) d")
        )
        return regression_loss, dict(
            camera_loss=regression_loss,
            position_loss=position_loss,
            distance=(position_loss * 2) ** 0.5,
        )
