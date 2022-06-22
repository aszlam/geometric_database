from abc import ABC, abstractmethod
from typing import Tuple
from torch.nn import Module


class AbstractViewEncoder(ABC, Module):
    """
    Abstract base class for a view encoder.

    A view encoder is something that takes in a view from our agent, which is assumed to be a
    H x W x C tensor, as well as a POV, (x, y, z, Q) (Q quarternion), and returns a latent
    vector associated with the view.
    """

    @abstractmethod
    def __init__(
        self,
        view_shape: Tuple[int, ...],
        representation_length: int,
        *args,
        **kwargs,
    ):
        super().__init__()
        pass
