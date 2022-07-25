from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

import einops

from shencoder import SHEncoder
from gridencoder import GridEncoder
from utils.mlp import MLP


def to_rn_matrix(normalized_view_direction):
    # Now, make the rotation matrix.
    view_rotation_matrix = torch.empty((len(normalized_view_direction, 3, 3))).to(
        normalized_view_direction
    )
    # last row is [a, b, c]
    view_rotation_matrix[:, 2, :] = normalized_view_direction
    # last column is [-a, -b, c]
    view_rotation_matrix[:, 2, :2] = -normalized_view_direction[:, :2]
    one_plus_c = 1 + normalized_view_direction[:, 2]
    a_square = normalized_view_direction[:, 0] ** 2
    b_square = normalized_view_direction[:, 1] ** 2
    ab = normalized_view_direction[:, 0] * normalized_view_direction[:, 1]
    view_rotation_matrix[:, 0, 0] = (1 - a_square) / one_plus_c
    view_rotation_matrix[:, 1, 1] = (1 - b_square) / one_plus_c
    view_rotation_matrix[:, 0, 1] = -ab / one_plus_c
    view_rotation_matrix[:, 1, 0] = -ab / one_plus_c
    return view_rotation_matrix


def preprocess_position_and_directions(view_position, view_direction):
    normalized_view_directions = F.normalize(view_direction, p=2, dim=-1)
    normalized_view_positions = torch.bmm(
        to_rn_matrix(normalized_view_directions), view_position
    )
    return normalized_view_positions[:, :2], normalized_view_directions


class DirectionalSDFModel(nn.Module):
    def __init__(
        self,
        grid_encoder_log2_level: int = 16,
        grid_encoder_levels: int = 16,
        grid_encoder_level_size: int = 2,
        spherical_harmonics_level: int = 4,
        model_width: int = 256,
        model_depth: int = 2,
        batchnorm: bool = False,
    ):
        super().__init__()
        self._grid_encoder = GridEncoder(
            input_dim=2,
            log2_hashmap_size=grid_encoder_log2_level,
            num_levels=grid_encoder_levels,
            per_level_scale=grid_encoder_level_size,
        )
        self._grid_encoder_output = grid_encoder_levels * grid_encoder_level_size
        self._sh_encoder = SHEncoder(input_dim=3, degree=spherical_harmonics_level)
        self._sh_encoder_output = spherical_harmonics_level ** 2

        self._dsdf_model_trunk = MLP(
            input_dim=self._grid_encoder_output + self._sh_encoder_output,
            output_dim=1,
            hidden_dim=model_width,
            hidden_depth=model_depth,
            batchnorm=batchnorm,
        )
        self.squashing_fn = F.sigmoid
        self.squashing_inv = lambda x: torch.log(x / (1 - x))
        self._eps = 1.0 - 1e-6

    def forward(self, position: torch.Tensor, direction: torch.Tensor):
        normalized_pos, normalized_dir = preprocess_position_and_directions(
            position, direction
        )
        enc_pos = self._grid_encoder(normalized_pos)
        enc_dir = self._sh_encoder(normalized_dir)
        # Return both unnormalized and normalized distance.
        unnormalized_dist = self._dsdf_model_trunk(enc_pos, enc_dir)
        normalized_dist = self.squashing_inv(
            torch.clip(unnormalized_dist, max=self._eps)
        ) - torch.sum(position * normalized_dir, dim=-1)
        return normalized_dist, unnormalized_dist

    def calculate_loss(self, position: torch.Tensor, direction: torch.Tensor, unnormalized_dist: torch.Tensor, true_distance: torch.Tensor):
        normalized_direction = F.normalize(direction, p=2, dim=-1)
        loss = F.smooth_l1_loss(self.squashing_fn(true_distance + torch.sum(position * normalized_direction, dim=-1)), unnormalized_dist)
        return loss
