from typing import Optional, Tuple
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
        max_coords: Optional[torch.Tensor] = None,
        min_coords: Optional[torch.Tensor] = None,
        mlp_depth: int = 2,
        mlp_width: int = 256,
        batchnorm: bool = False,
        num_levels: int = 16,
        level_dim: int = 4,
        log2_hashmap_size: int = 16,
        device: str = "cuda",
        image_rep_size: int = 512,
        text_rep_size: int = 512,
        segmentation_classes: int = 512,
        bounds: float = 10.0,
    ):
        super().__init__()

        self._grid_model = GridEncoder(
            input_dim=3,
            num_levels=num_levels,
            level_dim=level_dim,
            per_level_scale=2,
            base_resolution=16,
            log2_hashmap_size=log2_hashmap_size,
            desired_resolution=None,
            gridtype="hash",
            align_corners=False,
        )
        # Now convert the output with an MLP
        self._post_grid = MLP(
            input_dim=num_levels * level_dim,
            hidden_dim=mlp_width,
            hidden_depth=mlp_depth,
            output_dim=image_rep_size + text_rep_size + segmentation_classes,
            batchnorm=batchnorm,
        )
        # Mini MLP for extra storage for image loss
        self._image_head = nn.Identity()
        # MLP(
        #     input_dim=image_rep_size,
        #     hidden_dim=image_rep_size,
        #     hidden_depth=1,
        #     output_dim=image_rep_size,
        #     batchnorm=batchnorm,
        # )
        # Magic value adviced by @imisra
        self.temperature = nn.Parameter(torch.log(torch.tensor(1.0 / 0.07)))

        self._image_rep_size = image_rep_size
        self._text_rep_size = text_rep_size

        if not (max_coords is not None and min_coords is not None):
            self._max_bounds, self._min_bounds = (
                torch.ones(3) * bounds,
                torch.ones(3) * -bounds,
            )
        else:
            assert len(max_coords) == len(min_coords)
            self._max_bounds, self._min_bounds = max_coords, min_coords

        self._grid_model = self._grid_model.to(device)
        self._post_grid = self._post_grid.to(device)
        self._image_head = self._image_head.to(device)
        self.temperature.data = self.temperature.data.to(device)
        self._max_bounds = self._max_bounds.to(device)
        self._min_bounds = self._min_bounds.to(device)

    def compute_latents(self, x: torch.Tensor, bounds: Optional[float] = None):
        if bounds is None:
            max_bounds, min_bounds = self._max_bounds.to(x.device), self._min_bounds.to(
                x.device
            )
        else:
            max_bounds, min_bounds = (
                torch.ones(3, device=x.device) * bounds,
                torch.ones(3, device=x.device) * -bounds,
            )
        bounded_x = (x - min_bounds) / (max_bounds - min_bounds)
        grid_hash = self._grid_model(bounded_x, bound=1.0)
        result = self._post_grid(grid_hash)
        # label_latent, image_latent = torch.chunk(result, chunks=2, dim=-1)
        label_latent, image_latent, segmentation_logits = (
            result[..., : self._text_rep_size],
            result[
                ..., self._text_rep_size : self._text_rep_size + self._image_rep_size
            ],
            result[..., self._text_rep_size + self._image_rep_size :],
        )
        image_latent = self._image_head(image_latent)
        return label_latent, image_latent, segmentation_logits

    def forward(
        self,
        x: torch.Tensor,
        actual_label_latents: torch.Tensor,
        actual_image_latents: torch.Tensor,
        actual_label_indices: torch.Tensor,
        actual_image_indices: torch.Tensor,
        label_weights: torch.Tensor,
        image_weights: torch.Tensor,
        instance_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        # Given all the necessary ingredients, compute the elementwise loss.
        label_latent, image_latent, segmentation_logits = self.compute_latents(x)
        label_loss_batch = self.compute_full_loss(
            label_latent,
            actual_label_latents,
            actual_label_index=actual_label_indices,
            weights=label_weights,
        )
        image_loss_batch = self.compute_full_loss(
            image_latent,
            actual_image_latents,
            actual_label_index=actual_image_indices,
            weights=image_weights,
        )
        (
            instanse_loss_batch,
            instance_accuracy,
        ) = self.compute_instance_loss_and_accuracy(
            instance_logits=segmentation_logits, instance_labels=instance_labels
        )
        return (
            label_loss_batch.mean(),
            label_latent,
            image_loss_batch.mean(),
            image_latent,
            instanse_loss_batch,
            instance_accuracy,
        )

    def to(self, device):
        self._grid_model = self._grid_model.to(device)
        self._post_grid = self._post_grid.to(device)
        self._image_head = self._image_head.to(device)
        self._max_bounds = self._max_bounds.to(device)
        self._min_bounds = self._min_bounds.to(device)
        self.temperature.data = self.temperature.data.to(device)
        return self

    def compute_loss(
        self,
        predicted_latents: torch.Tensor,
        actual_latents: torch.Tensor,
        label_mask: torch.Tensor = None,
        weights: torch.Tensor = None,
    ):
        temp = torch.exp(self.temperature)
        sim = torch.einsum("i d, j d -> i j", predicted_latents, actual_latents) * temp
        # Zero out the cells where the labels are same.
        if label_mask is not None:
            sim = sim * label_mask
            del label_mask
        labels = torch.arange(len(predicted_latents), device=predicted_latents.device)
        loss = (
            F.cross_entropy(sim, labels, reduction="none")
            + F.cross_entropy(sim.t(), labels, reduction="none")
        ) / 2
        if weights is not None:
            loss = loss * weights
        return loss

    @torch.no_grad()
    def compute_label_mask(self, label_index: torch.Tensor):
        batch_size = label_index.size(0)
        label_mask = (label_index != label_index.t()).float() + torch.eye(
            batch_size, device=label_index.device
        )
        return label_mask

    def compute_full_loss(
        self,
        predicted_latents: torch.Tensor,
        actual_latents: torch.Tensor,
        actual_label_index: torch.Tensor,
        weights: torch.Tensor,
    ):
        label_mask = self.compute_label_mask(actual_label_index)
        return self.compute_loss(
            predicted_latents=predicted_latents,
            actual_latents=actual_latents,
            label_mask=label_mask,
            weights=weights,
        )

    def compute_instance_loss_and_accuracy(
        self,
        instance_logits: torch.Tensor,
        instance_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        instance_mask = instance_labels != -1
        if not torch.all(instance_labels == -1):
            inst_segmentation_loss = F.cross_entropy(
                instance_logits[instance_mask], instance_labels[instance_mask]
            )
            accuracy = (
                (
                    instance_logits[instance_mask].argmax(dim=-1)
                    == instance_labels[instance_mask]
                )
                .float()
                .mean()
            )
            accuracy = accuracy.detach()
        else:
            inst_segmentation_loss = torch.zeros(1, device=instance_logits.device)
            accuracy = torch.ones(1, device=instance_logits.device)
        return inst_segmentation_loss, accuracy
