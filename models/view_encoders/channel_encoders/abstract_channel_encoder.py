import abc
from typing import Union

import torch
import torch.nn as nn


class AbstractChannelEncoder(abc.ABC, nn.Module):
    def __init__(
        self,
        num_channels: int,
        representation_dim: int,
        device: Union[str, torch.device],
        *args,
        **kwargs
    ):
        super().__init__()
        self._num_channels = num_channels
        self._representation_size = representation_dim

    @abc.abstractmethod
    def encode_view(self, view: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Must be implemented in per-channel encoder.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode_view(x)

    def preprocess_view(self, view: torch.Tensor) -> torch.Tensor:
        """
        Some models such as point cloud encoders will need depth models to be projected into
        a point cloud format first (i.e. batch_size x num_points x 6, where 6 denotes XYZRGB)

        Defaults to just returning the view.
        """
        return view
