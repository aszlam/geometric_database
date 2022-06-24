from typing import Dict, Iterable

import einops
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.view_encoders.abstract_view_encoder import AbstractViewEncoder
import utils


class MaskedTimmViewEncoder(AbstractViewEncoder):
    """
    View encoder where the architecture is initialized by a base timm library architecture.
    """

    def __init__(
        self,
        view_shape: Iterable[int],
        representation_length: int,
        timm_class: str,
        semantic_embedding_len: int,
        num_semantic_classes: int,
        device: str = "cuda",
    ):
        super().__init__(
            view_shape=view_shape, representation_length=representation_length
        )
        self.view_shape = tuple(view_shape)
        self.semantic_embedding_layer = nn.Embedding(
            num_embeddings=num_semantic_classes,
            embedding_dim=semantic_embedding_len,
            device=device,
        )
        self.num_semantic_classes = num_semantic_classes
        self.device = device
        self.representation_length = representation_length
        self.visual_model = timm.create_model(
            model_name=timm_class,
            pretrained=True,
            in_chans=(4 + 1 + semantic_embedding_len),
            num_classes=0,
        )
        self.visual_model.to(device)
        # Create adapters for resizing the input images to the necessary size, and rescaling
        # the output representation to the right representation length.
        self._setup_adapters_and_masks()

    def _setup_adapters_and_masks(self):
        BATCH_SIZE = 2
        view_shape = self.view_shape
        depth = torch.randn((BATCH_SIZE,) + view_shape)
        rgba = torch.randn((BATCH_SIZE, 4) + view_shape)
        semantic_segmentation = torch.randint_like(
            depth, high=self.num_semantic_classes
        ).long()
        self._depth_mask = nn.parameter.Parameter(data=depth)
        self._rgba_mask = nn.parameter.Parameter(data=rgba)
        self._semantic_mask = nn.parameter.Parameter(
            data=semantic_segmentation, requires_grad=False
        )
        self._mask_tokens = {
            "rgb": self._rgba_mask,
            "truth": self._semantic_mask,
            "depth": self._depth_mask,
        }
        sample_batch = {
            "rgb": rgba,
            "truth": semantic_segmentation,
            "depth": depth,
        }
        results = self.forward(
            {k: v.to(self.device) for k, v in sample_batch.items()},
            adapt_encoding=False,
        )
        self._visual_rep_len = results.shape[-1]
        self.embedding_adapter = (
            nn.Linear(self._visual_rep_len, self.representation_length)
            if self._visual_rep_len != self.representation_length
            else nn.Identity()
        )
        self.embedding_adapter.to(self.device)
        self._depth_mask.data = self._depth_mask.data.to(self.device)
        self._rgba_mask.data = self._rgba_mask.data.to(self.device)
        self._semantic_mask.data = self._semantic_mask.data.to(self.device)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        adapt_encoding: bool = True,
        masking_dict: Dict[str, torch.Tensor] = {},
    ) -> torch.Tensor:
        """
        Forward a view and encode that into a representation.
        """
        # First, make masked versions of the dictionaries.
        masked_dict = {k: v for k, v in x_dict.items()}
        for mask_key, mask in masking_dict.items():
            if mask_key in x_dict:
                masked_dict[mask_key] = utils.mask_batch_with_mask_token(
                    x_dict[mask_key],
                    mask_token=self._mask_tokens,
                    batch_mask_indices=mask,
                )

        embedded_semantics = self.semantic_embedding_layer(masked_dict["truth"])
        # Now stack the channels.
        rgba_image = einops.rearrange(masked_dict["rgb"], "... c h w -> ... h w c")
        model_input = torch.cat(
            [rgba_image, masked_dict["depth"].unsqueeze(-1), embedded_semantics], dim=-1
        )
        channel_first = einops.rearrange(model_input, "... h w c -> ... c h w")
        visual_rep = self.visual_model(channel_first)
        return visual_rep if not adapt_encoding else self.embedding_adapter(visual_rep)
