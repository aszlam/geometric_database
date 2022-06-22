from typing import Tuple, Union

import torch
import torch.nn as nn
import torchvision.transforms.functional as tvf

import clip
from models.view_encoders.channel_encoders.abstract_channel_encoder import AbstractChannelEncoder


class CLIPEncoder(AbstractChannelEncoder):
    inherent_encoder_dim = 512
    available_models = clip.available_models()

    def __init__(
        self,
        representation_dim: int,
        image_size: Tuple[int, int],
        num_channels: int,
        model_class: str = "ViT-B/32",
        device: Union[str, torch.device] = "cuda",
    ):
        super().__init__(
            num_channels=num_channels,
            representation_dim=representation_dim,
            device=device,
        )
        self._representation_dim = representation_dim
        self._image_size = image_size
        self._num_channels = num_channels
        self.device = device

        # Just for CLIP, we need RGB or grayscale.
        assert self._num_channels in (1, 3), "Need RGB or grayscale images for CLIP"
        assert (
            model_class in CLIPEncoder.available_models
        ), f"Model {model_class} is not available, please choose a model from {CLIPEncoder.available_models}"

        self.model, self.preprocess = clip.load(model_class)
        self.model = self.model.to(device).eval()
        _input_resolution = self.model.visual.input_resolution
        if not all([_input_resolution == x for x in image_size]):
            self._resize = lambda x: tvf.resize(
                x, size=[_input_resolution, _input_resolution]
            )
        else:
            self._resize = nn.Identity()

        if representation_dim != self.inherent_encoder_dim:
            self.map_to_rep_dim = nn.Linear(
                in_features=self.inherent_encoder_dim, out_features=representation_dim
            )
        else:
            self.map_to_rep_dim = nn.Identity()

        self.map_to_rep_dim = self.map_to_rep_dim.to(device)

    def preprocess_view(self, view: torch.Tensor) -> torch.Tensor:
        return self._resize(view)

    def encode_view(self, view: torch.Tensor) -> torch.Tensor:
        return self.map_to_rep_dim(self.preprocess_view(view))
