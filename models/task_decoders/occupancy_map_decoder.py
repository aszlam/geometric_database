"""
Given the representation of the point (x, y, z), decode what the semantic tag of that position
should be.
"""
from typing import Dict, Tuple
import torch
import torch.nn as nn

from models.task_decoders.abstract_decoder import AbstractDecoder
from utils.mlp import MLP


class SemanticOccupancyDecoder(AbstractDecoder):
    def __init__(
        self,
        representation_length: int,
        depth: int = 3,
        hidden_dim: int = 256,
        batchnorm: bool = True,
    ):
        super().__init__()
        self._rep_length = representation_length
        self._depth = depth
        self._hidden_dim = hidden_dim
        self._batchnorm = batchnorm

        self.loss = nn.CrossEntropyLoss()

    def register_embedding_map(self, embedding: nn.Embedding) -> None:
        super().register_embedding_map(embedding)
        # Create an MLP to map the representation to the same length as the embedding dict.
        self.trunk = MLP(
            input_dim=self._rep_length,
            hidden_dim=self._hidden_dim,
            output_dim=self._embedding.weight.shape[-1],
            hidden_depth=self._depth,
            batchnorm=self._batchnorm,
        )

    def decode_representations(self, scene_model_reps: torch.Tensor) -> torch.Tensor:
        # We learn to extract the semantic tag of the position.
        return self.trunk(scene_model_reps)

    def compute_detailed_loss(
        self, decoded_representation: torch.Tensor, ground_truth: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # For this, we could later try to learn a contrastive loss.
        # TODO mahi https://github.com/HobbitLong/SupContrast/blob/master/losses.py
        # For now, we will just compute a classification loss
        return self.loss(decoded_representation, ground_truth)
