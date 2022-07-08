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


class PixelLocationConvLargeResidualDecoder(AbstractDecoder):
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
        seg_beta: float = 1.0,
        use_log: bool = True,
        use_residual: bool = True,
        use_linalg_solve: bool = True,
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
        self.representation_loss = nn.CrossEntropyLoss()
        self.lam = lam
        self.use_residual = use_residual
        self.use_log_loss = use_log
        self._seg_beta = seg_beta
        # Find the least squares solution to the transformation matrix
        self._leastsq = use_linalg_solve

    def register_embedding_map(self, embedding: nn.Embedding) -> None:
        super().register_embedding_map(embedding)
        # Create an MLP to map the representation to the same length as the embedding dict.
        self.trunk = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self._rep_length, out_channels=64, kernel_size=1
            ),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=1),
        )
        self.trunk = self.trunk.to(self._device)

        # Also create a recognition network XYZ -> labels
        self._segmentation_network = MLP(
            input_dim=self._rep_length,
            hidden_depth=3,
            hidden_dim=256,
            batchnorm=True,
            output_dim=self._embedding.weight.shape[0],
        )
        self._segmentation_network = self._segmentation_network.to(self._device)

    def decode_representations(
        self, view_reps: torch.Tensor, scene_model_reps: torch.Tensor
    ) -> torch.Tensor:
        # We learn to extract the semantic tag of the position.
        reps_only = view_reps.squeeze(0)
        if self.use_residual:
            mean_over_image = torch.mean(reps_only, dim=[-1, -2], keepdim=True)
            decoded_xyz = self.trunk(reps_only + mean_over_image)
        else:
            decoded_xyz = self.trunk(reps_only)
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

        # First, figure out the classification loss.
        segmentation_logits = self._segmentation_network(decoded_representation)
        # Now use CE to calculate the actual loss.
        segmentation_loss = self.representation_loss(
            segmentation_logits, ground_truth[1]["label"]
        )
        if self.use_log_loss:
            position_loss = self.lam * torch.log(
                self.loss(decoded_view_representation, xyz_subset)
            )
            distance = (2 * torch.exp(position_loss / self.lam)) ** 0.5

        else:
            position_loss = self.lam * (
                self.loss(decoded_view_representation, xyz_subset)
            )
            distance = (2 * (position_loss / self.lam)) ** 0.5

        with torch.no_grad():
            if self._leastsq:
                # Use least square solve to see what the error becomes.
                predicted_coords_and_one = torch.cat(
                    [
                        einops.rearrange(
                            decoded_view_representation, "b d w h -> b (w h) d"
                        ),
                        torch.ones(
                            (len(decoded_view_representation), self._grid_size**2, 1),
                            device=self._device,
                        ),
                    ],
                    dim=-1,
                )

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
                transformation_matrix = torch.linalg.lstsq(
                    local_coords_and_one, predicted_coords_and_one
                ).solution
                # Now, find the least square distance.
                projected_points_and_one = torch.bmm(
                    local_coords_and_one, transformation_matrix
                )
                projected_loss = F.smooth_l1_loss(
                    projected_points_and_one[..., :3],
                    einops.rearrange(xyz_subset, "b d w h -> b (w h) d"),
                )
                projected_distance = (2 * projected_loss) ** 0.5
            else:
                projected_loss = 0
                projected_distance = 0
        return position_loss + self._seg_beta * segmentation_loss, dict(
            position_loss=position_loss,
            distance=distance,
            segmentation_loss=segmentation_loss,
            projected_loss=projected_loss,
            projected_distance=projected_distance,
        )
