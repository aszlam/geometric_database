from typing import Dict
import clip
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from dataloaders.habitat_loaders import HabitatLocationDataset, HabitatViewDataset
from sentence_transformers import SentenceTransformer


class ClipLabelledLocation(Dataset):
    PROMPT = "A "
    EMPTY = "Background"

    def __init__(
        self,
        location_dataset: HabitatLocationDataset,
        clip_model_name: str = "ViT-B/32",
        sentence_encoding_model_name="all-mpnet-base-v2",
        device: str = "cuda",
        batch_size: int = 128,
    ):
        self.loc_dataset = location_dataset
        model, preprocess = clip.load(clip_model_name, device=device)
        sentence_model = SentenceTransformer(
            sentence_encoding_model_name, device=device
        )
        self._semantic_weight = (
            1.0 if self.loc_dataset._return_nonsegmented_images else 0.0
        )
        self._semantic_weight = torch.tensor(self._semantic_weight)
        self._id_to_clip_vector = {}
        self._view_to_clip_vector = {}
        self._setup_clip_vectors(
            model,
            sentence_model,
            self.loc_dataset.image_extractor.instance_id_to_name,
            batch_size,
            device,
        )

    def _setup_clip_vectors(
        self, clip_model, sentence_model, id_to_name: Dict, batch_size: int, device: str
    ):
        # Step 1: set up all the clip vectors for the tags.
        # Tokenize all the names.
        text_strings = []
        for name in id_to_name.values():
            text_strings.append(self.PROMPT + name.replace("-", " "))
        text_strings.append(self.EMPTY)
        # clip_tokens = clip.tokenize(text_strings)
        # Now, encode them into Clip vectors.
        # all_embedded_text = []
        # with torch.no_grad():
        #     for i in range(0, len(clip_tokens), batch_size):
        #         start, end = i, min(i + batch_size, len(clip_tokens))
        #         batch_data = clip_tokens[start:end].to(device)
        #         embedded_text = clip_model.encode_text(batch_data).float()
        #         embedded_text = F.normalize(embedded_text, p=2, dim=-1)
        #         all_embedded_text.append(embedded_text.cpu())
        # all_embedded_text = torch.cat(all_embedded_text)
        with torch.no_grad():
            all_embedded_text = sentence_model.encode(text_strings)
            all_embedded_text = torch.from_numpy(all_embedded_text).float()
        self._text_embed_size = all_embedded_text.size(-1)
        # Now map from text data to embeddings.
        for index, id in enumerate(id_to_name.keys()):
            self._id_to_clip_vector[id] = all_embedded_text[index]
        # The empty index
        self._id_to_clip_vector[-1] = all_embedded_text[-1]

        # Step 2: set up clip vector for every view images.
        habitat_view_ds = self.loc_dataset.habitat_view_dataset
        # set up dataloader
        all_clip_embeddings = []
        dataloader = DataLoader(
            habitat_view_ds, batch_size=batch_size, shuffle=False, pin_memory=False
        )
        with torch.no_grad():
            for data_dict in dataloader:
                rgb = einops.rearrange(data_dict["rgb"][..., :3], "b h w c -> b c h w")
                clip_embeddings = clip_model.encode_image(rgb.to(device)).float().cpu()
                all_clip_embeddings.append(clip_embeddings)
        all_clip_embeddings = torch.cat(all_clip_embeddings)
        self._image_embed_size = all_clip_embeddings.size(-1)
        self._view_to_clip_vector = {
            idx: vector for idx, vector in enumerate(all_clip_embeddings)
        }

    def __len__(self):
        return len(self.loc_dataset)

    def __getitem__(self, index):
        location_data = self.loc_dataset[index]
        result = {
            "clip_vector": self._id_to_clip_vector.get(
                location_data["label"].item(), self._id_to_clip_vector[-1]
            ),
            "clip_image_vector": self._view_to_clip_vector.get(
                location_data["img_idx"].item(), None  # self._id_to_clip_vector[-1]
            ),
            "semantic_weight": self._semantic_weight,
        }
        result.update(location_data)
        return result

    @property
    def image_representation_size(self):
        return self._image_embed_size

    @property
    def text_representation_size(self):
        return self._text_embed_size
