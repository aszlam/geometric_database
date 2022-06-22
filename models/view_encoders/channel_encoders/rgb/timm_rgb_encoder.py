from typing import Iterable, Union

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.view_encoders.channel_encoders.abstract_channel_encoder import (
    AbstractChannelEncoder,
)


class TimmViewEncoder(AbstractChannelEncoder):
    """
    View encoder where the architecture is initialized by a base timm library architecture.
    """

    def __init__(
        self,
        image_size: Iterable[int],
        representation_dim: int,
        model_class: str,
        num_channels: int,
        device: Union[str, torch.device] = "cuda",
    ):
        super().__init__(
            num_channels=num_channels,
            representation_dim=representation_dim,
            device=device,
        )
        self.view_shape = tuple(image_size)
        self.device = device
        self.representation_length = representation_dim
        self.visual_model = timm.create_model(
            model_name=model_class,
            pretrained=True,
            in_chans=(num_channels),
            num_classes=0,
        )
        self.visual_model.to(device)
        # Create adapters for resizing the input images to the necessary size, and rescaling
        # the output representation to the right representation length.
        self._setup_adapters()

    def _setup_adapters(self):
        BATCH_SIZE = 2
        view_shape = self.view_shape
        rgb = torch.randn((BATCH_SIZE, 3) + view_shape)

        results = self.visual_model.encode_view(
            rgb,
            adapt_encoding=False,
        )
        self._visual_rep_len = results.shape[-1]
        self.embedding_adapter = (
            nn.Linear(self._visual_rep_len, self.representation_length)
            if self._visual_rep_len != self.representation_length
            else nn.Identity()
        )
        self.embedding_adapter.to(self.device)

    def encode_view(
        self, view: torch.Tensor, adapt_encoding: bool = True
    ) -> torch.Tensor:
        """
        Forward a view and encode that into a representation.
        """
        # Now stack the channels.
        visual_rep = self.visual_model(view)
        return visual_rep if not adapt_encoding else self.embedding_adapter(visual_rep)
