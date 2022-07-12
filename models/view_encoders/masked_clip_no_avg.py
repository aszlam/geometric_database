from typing import Dict, Iterable

import einops
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.view_encoders.abstract_view_encoder import AbstractViewEncoder
from models.scene_models.positional_encoding import FourierFeatures
import utils


class ClipEncoder(AbstractViewEncoder):
    """
    View encoder where the architecture is initialized by a base timm library architecture.
    """

    def __init__(
        self,
        view_shape: Iterable[int],
        representation_length: int,
        clip_class: str,
        device: str = "cuda",
        num_semantic_classes: int = 256,
        semantic_embedding_len: int = 16,
    ):
        super().__init__(
            view_shape=view_shape, representation_length=representation_length
        )
        self.view_shape = tuple(view_shape)
        self.device = device
        self.representation_length = representation_length
        model, preprocess = clip.load(clip_class, device=device)
        self.visual_model = model.visual
        self._channels_to_use = [
            True,  # Always use RGB
            False,
            False,
            False,
        ]
        # Now, convert to the no-avg-pool model.
        self.visual_model.to(device)
        self.final_avgpool = self.visual_model.attnpool
        self.visual_model.attnpool = nn.Identity()
        self.num_semantic_classes = num_semantic_classes
        self.semantic_embedding_layer = nn.Embedding(
            num_embeddings=num_semantic_classes,
            embedding_dim=semantic_embedding_len,
            device=device,
        )
        self._fourier_map = nn.Identity()
        # Create adapters for resizing the input images to the necessary size, and rescaling
        # the output representation to the right representation length.
        self._setup_adapters_and_masks()

    def _setup_adapters_and_masks(self):
        BATCH_SIZE = 1
        view_shape = self.view_shape
        depth = torch.randn((BATCH_SIZE,) + view_shape)
        rgba = torch.randn((BATCH_SIZE,) + view_shape + (3,))
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

        # embedded_semantics = self.semantic_embedding_layer(masked_dict["truth"])
        # # Now stack the channels.
        # rgba_image = einops.rearrange(masked_dict["rgb"], "... c h w -> ... h w c")
        # model_input = torch.cat(
        #     [rgba_image, masked_dict["depth"].unsqueeze(-1), embedded_semantics], dim=-1
        # )
        # channel_first = einops.rearrange(model_input, "... h w c -> ... c h w")
        # visual_rep = self.visual_model(channel_first)
        # return visual_rep if not adapt_encoding else self.embedding_adapter(visual_rep)
        to_cat = [
            masked_dict["rgb"][..., :3],
            masked_dict["depth"].unsqueeze(-1),
            self._fourier_map(masked_dict["local_xyz_position"]),
            self.semantic_embedding_layer(masked_dict["truth"]),
        ]
        selected_items = [to_cat[i] for i in range(4) if self._channels_to_use[i]]
        model_input = torch.cat(selected_items, dim=-1)
        channel_first = einops.rearrange(model_input, "... h w c -> ... c h w")
        visual_rep: torch.Tensor = self.visual_model(channel_first)
        return visual_rep if not adapt_encoding else self.embedding_adapter(visual_rep)
