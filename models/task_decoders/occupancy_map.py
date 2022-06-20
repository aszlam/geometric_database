"""
Given the representation of the point (x, y, z), decode what the semantic tag of that position
should be.
"""
from typing import Dict, Tuple, Union
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
        device: Union[str, torch.device] = "cuda",
    ):
        super().__init__()
        self._rep_length = representation_length
        self._depth = depth
        self._hidden_dim = hidden_dim
        self._batchnorm = batchnorm
        self._device = device

        # For this, we could later try to learn a contrastive loss.
        # TODO mahi https://github.com/HobbitLong/SupContrast/blob/master/losses.py
        # For now, we will just compute a classification loss
        self.loss = nn.CrossEntropyLoss()

    def register_embedding_map(self, embedding: nn.Embedding) -> None:
        super().register_embedding_map(embedding)
        # Create an MLP to map the representation to the same length as the embedding dict.
        self.trunk = MLP(
            input_dim=self._rep_length,
            hidden_dim=self._hidden_dim,
            # TODO switch out this line when we are mapping everything to its own embedding
            # vector instead of a one-hot vector for classification.
            # output_dim=self._embedding.weight.shape[-1],
            output_dim=self._embedding.weight.shape[0],
            hidden_depth=self._depth,
            batchnorm=self._batchnorm,
        )
        self.trunk = self.trunk.to(self._device)

    def decode_representations(self, scene_model_reps: torch.Tensor) -> torch.Tensor:
        # We learn to extract the semantic tag of the position.
        return self.trunk(scene_model_reps.squeeze(0))

    @staticmethod
    def _accuracy(output, target, topk=(1, 5)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = torch.flatten(correct[:k]).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def compute_detailed_loss(
        self, decoded_representation: torch.Tensor, ground_truth: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        occupancy_loss = self.loss(decoded_representation, ground_truth)
        topk_accuracy = self._accuracy(decoded_representation, ground_truth)
        return occupancy_loss, dict(
            occupancy_loss=occupancy_loss, top1=topk_accuracy[0], top5=topk_accuracy[1]
        )
