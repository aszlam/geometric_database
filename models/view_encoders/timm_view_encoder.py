from typing import Dict, Iterable

import einops
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.view_encoders.abstract_view_encoder import AbstractViewEncoder


class TimmViewEncoder(AbstractViewEncoder):
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

        self.visual_model_whole = timm.create_model(
            model_name=timm_class,
            pretrained=True,
            in_chans=(4 + 1 + 3 + semantic_embedding_len),
            num_classes=0,
        )
        self.visual_model_whole.to(device)
        # self.visual_model_features.to(device)
        # Create adapters for resizing the input images to the necessary size, and rescaling
        # the output representation to the right representation length.
        self._setup_adapters()

    def _setup_adapters(self):
        BATCH_SIZE = 2
        view_shape = self.view_shape
        depth = torch.randn((BATCH_SIZE,) + view_shape)
        rgba = torch.randn((BATCH_SIZE,) + view_shape + (4,))
        local_xyz = torch.randn((BATCH_SIZE,) + view_shape + (3,))
        semantic_segmentation = torch.randint_like(
            depth, high=self.num_semantic_classes
        ).long()
        sample_batch = {
            "rgb": rgba,
            "truth": semantic_segmentation,
            "depth": depth,
            "local_xyz_position": local_xyz,
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

    def forward(
        self, x_dict: Dict[str, torch.Tensor], adapt_encoding: bool = True
    ) -> torch.Tensor:
        """
        Forward a view and encode that into a representation.
        """
        model_input = torch.cat(
            [
                x_dict["rgb"],
                x_dict["depth"].unsqueeze(-1),
                x_dict["local_xyz_position"],
                self.semantic_embedding_layer(x_dict["truth"]),
            ],
            dim=-1,
        )
        channel_first = einops.rearrange(model_input, "... h w c -> ... c h w")
        visual_rep = self.visual_model_whole(channel_first)
        return visual_rep if not adapt_encoding else self.embedding_adapter(visual_rep)
