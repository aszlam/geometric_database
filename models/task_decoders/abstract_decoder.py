import abc
from typing import Any, Dict, Tuple
import torch
import torch.nn as nn


class AbstractDecoder(nn.Module, abc.ABC):
    """
    An abstraction for the decoder class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def register_embedding_map(self, embedding: nn.Embedding) -> None:
        """
        Register the semantic embedding map used in the model training.
        """
        self._embedding = embedding

    @abc.abstractmethod
    def decode_representations(self, scene_model_reps: torch.Tensor) -> torch.Tensor:
        """
        Decode the representation into a response for the relevant query.

        To be implemented in the subclasses.
        """
        raise NotImplementedError(
            "Representation decoding must be implemented per decoder."
        )

    @abc.abstractmethod
    def compute_detailed_loss(
        self,
        decoded_representation: torch.Tensor,
        ground_truth: Any,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the loss as well as the loss components for this particular decoder given
        ground truth implementation.

        To be implemented in the subclasses.
        """
        raise NotImplementedError(
            "Detailed loss computation must be implemented per decoder."
        )
