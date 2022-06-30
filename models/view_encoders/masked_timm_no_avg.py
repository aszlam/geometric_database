from typing import Dict, Iterable

import einops
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.view_encoders.abstract_view_encoder import AbstractViewEncoder
from models.scene_models.positional_encoding import FourierFeatures
import utils


class MaskedTimmNoPoolEncoder(AbstractViewEncoder):
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
        use_fourier_features: bool = False,
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
        self._use_fourier_features = use_fourier_features
        if use_fourier_features:
            self._fourier_map = FourierFeatures(
                input_dim=3, fourier_embedding_dim=24, fourier_embedding_scale=1
            )
            self._fourier_map.to(device)
            xyz_embed_dim = 24
        else:
            xyz_embed_dim = 3
            self._fourier_map = nn.Identity()
        self.visual_model = timm.create_model(
            model_name=timm_class,
            pretrained=True,
            in_chans=(4 + 1 + xyz_embed_dim + semantic_embedding_len),
            num_classes=0,
        )
        # Now, convert to the no-avg-pool model.
        self.visual_model.to(device)
        self.final_avgpool = self.visual_model.global_pool
        self.visual_model.global_pool = nn.Identity()
        # Create adapters for resizing the input images to the necessary size, and rescaling
        # the output representation to the right representation length.
        self._setup_adapters_and_masks()

    def _setup_adapters_and_masks(self):
        BATCH_SIZE = 1
        view_shape = self.view_shape
        depth = torch.randn((BATCH_SIZE,) + view_shape)
        rgba = torch.randn((BATCH_SIZE,) + view_shape + (4,))
        semantic_segmentation = torch.randint_like(
            depth, high=self.num_semantic_classes
        ).long()
        local_xyz = torch.randn((BATCH_SIZE,) + view_shape + (3,))
        self._depth_mask = nn.parameter.Parameter(data=depth)
        self._rgba_mask = nn.parameter.Parameter(data=rgba)
        self._semantic_mask = nn.parameter.Parameter(
            data=semantic_segmentation, requires_grad=False
        )
        self._local_xyz_mask = nn.parameter.Parameter(data=local_xyz)
        self._mask_tokens = {
            "rgb": self._rgba_mask,
            "truth": self._semantic_mask,
            "depth": self._depth_mask,
            "local_xyz_position": self._local_xyz_mask,
        }
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
        self._visual_rep_len = results.shape[1]
        self.embedding_adapter = (
            nn.Conv2d(
                in_channels=self._visual_rep_len,
                out_channels=self.representation_length,
                kernel_size=1,
            )
            if self._visual_rep_len != self.representation_length
            else nn.Identity()
        )
        self.embedding_adapter.to(self.device)
        self._depth_mask.data = self._depth_mask.data.to(self.device)
        self._rgba_mask.data = self._rgba_mask.data.to(self.device)
        self._semantic_mask.data = self._semantic_mask.data.to(self.device)
        self._local_xyz_mask.data = self._local_xyz_mask.data.to(self.device)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        adapt_encoding: bool = True,
        masking_dict: Dict[str, torch.Tensor] = {},
    ) -> torch.Tensor:
        """
        Forward a view and encode that into a representation.
        """
        # First, make masked versions of the dictionaries.
        masked_dict = {k: v for k, v in x_dict.items()}
        for mask_key, mask in masking_dict.items():
            if mask_key in x_dict:
                masked_dict[mask_key] = utils.mask_batch_with_mask_token(
                    x_dict[mask_key],
                    mask_token=self._mask_tokens,
                    batch_mask_indices=mask,
                )

        # embedded_semantics = self.semantic_embedding_layer(masked_dict["truth"])
        # # Now stack the channels.
        # rgba_image = einops.rearrange(masked_dict["rgb"], "... c h w -> ... h w c")
        # model_input = torch.cat(
        #     [rgba_image, masked_dict["depth"].unsqueeze(-1), embedded_semantics], dim=-1
        # )
        # channel_first = einops.rearrange(model_input, "... h w c -> ... c h w")
        # visual_rep = self.visual_model(channel_first)
        # return visual_rep if not adapt_encoding else self.embedding_adapter(visual_rep)

        model_input = torch.cat(
            [
                masked_dict["rgb"],
                masked_dict["depth"].unsqueeze(-1),
                self._fourier_map(masked_dict["local_xyz_position"]),
                self.semantic_embedding_layer(masked_dict["truth"]),
            ],
            dim=-1,
        )
        channel_first = einops.rearrange(model_input, "... h w c -> ... c h w")
        visual_rep = self.visual_model(channel_first)
        return visual_rep if not adapt_encoding else self.embedding_adapter(visual_rep)
