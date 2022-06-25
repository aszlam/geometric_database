from typing import Dict, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.scene_models.transformers import Transformer
from models.scene_models.positional_encoding import PositionalEmbedding
from models.view_encoders.abstract_view_encoder import AbstractViewEncoder
import utils


class SceneTransformer(nn.Module):
    """
    Scene transformer model that learns the prior about each scene.
    """

    def __init__(
        self,
        representation_dim: int,
        scene_model: Transformer,
        device: Union[str, torch.device],
        mask_prob: float = 0.5,
        pos_mask_prob: float = 0.9,
    ):
        super().__init__()
        self.scene_model = scene_model
        self.rep_dim = representation_dim
        # Set up tokens for views (updates) and queries so that our model can separate them
        self.view_token = nn.parameter.Parameter(data=torch.randn(self.rep_dim))
        self.query_token = nn.parameter.Parameter(data=torch.randn(self.rep_dim))

        self.xyz_position_mask = nn.parameter.Parameter(data=torch.randn(self.rep_dim))
        self.view_position_mask = nn.parameter.Parameter(data=torch.randn(self.rep_dim))
        self.view_direction_mask = nn.parameter.Parameter(
            data=torch.randn(self.rep_dim)
        )

        # Send to device.
        self.scene_model = self.scene_model.to(device)
        self.view_token.data = self.view_token.data.to(device)
        self.query_token.data = self.query_token.data.to(device)
        self.xyz_position_mask.data = self.query_token.data.to(device)
        self.view_position_mask.data = self.query_token.data.to(device)
        self.view_direction_mask.data = self.query_token.data.to(device)
        self.mask_prob = mask_prob
        self.pos_mask_prob = pos_mask_prob

    def register_encoders(
        self,
        positional_encoder_view: PositionalEmbedding,
        positional_encoder_query: PositionalEmbedding,
        quat_encoder: PositionalEmbedding,
        view_encoder: AbstractViewEncoder,
    ) -> None:
        # Register the common components across different scene encoders.
        self.view_xyz_encoder = positional_encoder_view
        self.query_xyz_encoder = positional_encoder_query
        self.view_quat_encoder = quat_encoder
        self.view_encoder = view_encoder

    def forward(
        self,
        view_dict: Dict[str, torch.Tensor],
        query_xyz: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a batch of views and a set of queries, this model will try to predict the
        semantic representation of the scene at position x y z.
        """
        # First, figure out how to mask each axis.
        to_mask = ["rgb", "truth", "depth"]
        pos_mask = ["camera_pos", "camera_direction"]
        masked_dict = {}
        for key in to_mask:
            masked_dict[key] = utils.generate_batch_mask(
                batch_size=len(view_dict[key]), masking_prob=self.mask_prob
            ).to(view_dict[key].device)
        for key in pos_mask:
            masked_dict[key] = utils.generate_batch_mask(
                batch_size=len(view_dict[key]), masking_prob=self.pos_mask_prob
            ).to(view_dict[key].device)
        encoded_views = self.view_encoder(view_dict, masked_dict)

        # Now, replace the view XYZ/quat with masks.
        encoded_view_xyz = self.view_xyz_encoder(view_dict["camera_pos"].float())
        encoded_view_quat = self.view_quat_encoder(
            view_dict["camera_direction"].float()
        )
        masked_encoded_view_xyz = utils.mask_batch_with_mask_token(
            encoded_view_xyz,
            mask_token=self.view_position_mask,
            batch_mask_indices=utils.generate_batch_mask(
                len(view_dict["camera_pos"]), masking_prob=self.mask_prob
            ),
        )
        masked_encoded_view_quat = utils.mask_batch_with_mask_token(
            encoded_view_quat,
            mask_token=self.view_direction_mask,
            batch_mask_indices=utils.generate_batch_mask(
                len(view_dict["camera_direction"]), masking_prob=self.mask_prob
            ),
        )
        encoded_query_xyz = self.query_xyz_encoder(query_xyz.float())

        num_views = len(encoded_view_xyz)
        num_queries = len(encoded_query_xyz)

        views = (
            self.view_token
            + encoded_views
            + masked_encoded_view_xyz
            + masked_encoded_view_quat
        )
        queries = self.query_token + encoded_query_xyz
        # Right now, in the transformer, there is a batch and there is a sequence.
        # We do not make a distinction right now, and just use the batch_size as 1
        # and sequence size as the full batch size. However, we can help the network learn
        # better priors by shuffling around stuff and creating multiple batches from the
        # same encoded sequences.
        # TODO Mahi: once we have gotten the encoded views and queries, randomly shuffle
        # and use half of them in each minibatch.
        scene_tf_input = torch.cat([views, queries], dim=0).unsqueeze(
            0
        )  # Join along batch axis since scene_tf operates on sets anyway.
        output: torch.Tensor = self.scene_model(scene_tf_input)
        assert output.size(1) == num_views + num_queries
        return output[:, :num_views, ...], output[:, num_views:, ...]
        # TODO Mahi decide if we want to add the positional encoding at every intermediate
        # transformer layers anyway
        # TODO Also figure out the attention map between different inputs. For example, we
        # probably don't want attention going into the pushed updates from the queries.
