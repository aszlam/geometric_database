from typing import Dict, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.scene_models.transformers import Transformer
from models.scene_models.positional_encoding import PositionalEmbedding
from models.view_encoders.abstract_view_encoder import AbstractViewEncoder

from utils.mlp import MLP


class Feedforward(nn.Module):
    """
    Simple feedforward network that ignores the image input and learns the xyz information
    directly.
    """

    def __init__(
        self,
        representation_dim: int,
        scene_model: MLP,
        device: Union[str, torch.device],
        use_fourier_features: bool = False,
    ):
        super().__init__()
        self.scene_model = scene_model
        self.rep_dim = representation_dim
        # Send to device.
        self.use_fourier_features = use_fourier_features
        self.scene_model = self.scene_model.to(device)

    def register_encoders(
        self,
        positional_encoder_view: PositionalEmbedding,
        positional_encoder_query: PositionalEmbedding,
        quat_encoder: PositionalEmbedding,
        view_encoder: AbstractViewEncoder,
    ) -> None:
        # This naive model ignores the view entirely, and just tries to predict
        # semantic label given XYZ.
        if self.use_fourier_features:
            self.positional_encoder_query = positional_encoder_query
        else:
            self.positional_encoder_query = nn.Identity()

    def forward(
        self, view_dict: Dict[str, torch.Tensor], query_xyz: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a batch of views and a set of queries, this model will try to predict the
        semantic representation of the scene at position x y z.
        """
        output: torch.Tensor = self.scene_model(self.positional_encoder_query(query_xyz.float()))
        return None, output
