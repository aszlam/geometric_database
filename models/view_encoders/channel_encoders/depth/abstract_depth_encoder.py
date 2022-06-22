import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import torch_choice

from models.view_encoders.channel_encoders.abstract_channel_encoder import (
    AbstractChannelEncoder,
)


class AbstractDepthEncoder(AbstractChannelEncoder):
    def _cached_projection_matrix(self, image_size: int = 224):
        if hasattr(self, "_cached_projection") and self._cached_projection:
            if self._cached_image_size == image_size:
                return self._cached_xs, self._cached_ys
        xs, ys = np.meshgrid(
            np.linspace(-1, 1, image_size), np.linspace(1, -1, image_size)
        )
        xs = xs.reshape(1, image_size, image_size)
        ys = ys.reshape(1, image_size, image_size)
        self._cached_xs = torch.from_numpy(xs).to(self.device)
        self._cached_ys = torch.from_numpy(ys).to(self.device)
        self._cached_projection = True
        self._cached_image_size = image_size
        return self._cached_xs, self._cached_ys

    def preprocess_view(
        self,
        depth_view: torch.Tensor,
        rgb_data: torch.Tensor,
        pixel_batch_size: int = 4096,
    ) -> torch.Tensor:
        """
        Take a point cloud view and convert it to a pointnet array (XYZRGB).
        """
        # Assuming it is not channel-last
        image_size = depth_view.size(-1)
        cached_xs, cached_ys = self._cached_projection_matrix(image_size=image_size)

        z = depth_view.reshape(-1, image_size, image_size)
        # Shape batch_size x image_size x image_size x 3
        xyz_one = torch.stack((cached_xs * z, cached_ys * z, -z), dim=-1)
        xyz_flat = xyz_one.reshape(-1, image_size**2, 3)
        rgb_flat = rgb_data.reshape(-1, image_size**2, 3)

        # Subsample points to fit in memory
        xyz_max, xyz_min = xyz_flat.max(dim=1, keepdim=True), xyz_flat.min(
            dim=1, keepdim=True
        )
        # TODO revisit this, this center calculation is very brittle to outliers now.
        xyz_center = (xyz_max + xyz_min) / 2.0
        chosen_points = torch_choice(
            pop_size=image_size**2, num_samples=pixel_batch_size, device=self.device
        )
        chosen_xyz_normalized = xyz_flat[chosen_points] - xyz_flat.max(
            dim=1, keepdim=True
        )
        processed_view = torch.concat(
            (
                xyz_flat[chosen_points] - xyz_center,
                rgb_flat[chosen_points] / 255.0,
                chosen_xyz_normalized,
            ),
            dim=-1,
        )

        return processed_view
