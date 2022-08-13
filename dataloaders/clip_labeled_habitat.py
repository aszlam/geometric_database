from typing import Dict, List
import clip
import einops
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize
from dataloaders.habitat_loaders import HabitatLocationDataset, HabitatViewDataset
from sentence_transformers import SentenceTransformer


class ClipLabelledLocation(Dataset):
    PROMPT = "A "
    EMPTY = "Other"
    FAR_DISTANCE = 2.0

    def __init__(
        self,
        view_dataset: HabitatViewDataset,
        location_dataset: HabitatLocationDataset,
        id_to_name: Dict[int, str],
        clip_model_name: str = "ViT-B/32",
        sentence_encoding_model_name="all-mpnet-base-v2",
        device: str = "cuda",
        batch_size: int = 128,
    ):
        self.loc_dataset = location_dataset
        model, _ = clip.load(clip_model_name, device=device)
        sentence_model = SentenceTransformer(
            sentence_encoding_model_name, device=device
        )
        self._semantic_weight = 1.0
        self._semantic_weight = torch.tensor(self._semantic_weight)
        self._id_to_clip_vector = {}
        self._view_to_clip_vector = {}
        self.resize = Resize(224)  # CLIP model has a fixed size input
        self._setup_clip_vectors(
            model,
            sentence_model,
            view_dataset,
            id_to_name,
            batch_size,
            device,
        )

    def _setup_clip_vectors(
        self,
        clip_model,
        sentence_model,
        habitat_view_ds: HabitatViewDataset,
        id_to_name: Dict,
        batch_size: int,
        device: str,
    ):
        # Step 1: set up all the clip vectors for the tags.
        # Tokenize all the names.
        text_strings = [self.EMPTY]
        for name in id_to_name.values():
            text_strings.append(self.PROMPT + name.replace("-", " ").replace("_", " "))

        with torch.no_grad():
            all_embedded_text = sentence_model.encode(text_strings)
            all_embedded_text = torch.from_numpy(all_embedded_text).float()
        self._text_embed_size = all_embedded_text.size(-1)
        # Now map from text data to embeddings.
        for id in id_to_name.keys():
            self._id_to_clip_vector[id] = all_embedded_text[id]
        # The empty index
        self._id_to_clip_vector[0] = all_embedded_text[0]

        # Step 2: set up clip vector for every view images.
        # set up dataloader
        all_clip_embeddings = []
        dataloader = DataLoader(
            habitat_view_ds, batch_size=batch_size, shuffle=False, pin_memory=False
        )
        with torch.no_grad():
            for data_dict in dataloader:
                rgb = einops.rearrange(data_dict["rgb"][..., :3], "b h w c -> b c h w")
                clip_embeddings = (
                    clip_model.encode_image(self.resize(rgb).to(device)).float().cpu()
                )
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
                location_data["label"].item(), self._id_to_clip_vector[0]
            ),
            "clip_image_vector": self._view_to_clip_vector.get(
                location_data["img_idx"].item(), None  # self._id_to_clip_vector[-1]
            ),
            "semantic_weight": self._semantic_weight,
            "distance": self.FAR_DISTANCE,  # Image labels are slightly misleading, so we set it high
        }
        result.update(location_data)
        return result

    @property
    def image_representation_size(self):
        return self._image_embed_size

    @property
    def text_representation_size(self):
        return self._text_embed_size

    @property
    def coordinate_range(self):
        return self.loc_dataset.max_coords, self.loc_dataset.min_coords


class ClassificationExtractor:
    PROMPT = "A "
    EMPTY_CLASS = "Other"
    LOGIT_TEMP = 100.0

    def __init__(
        self,
        clip_model_name: str,
        sentence_model_name: str,
        class_names: List[str],
        device: str = "cuda",
    ):
        clip_model, _ = clip.load(clip_model_name, device=device)
        sentence_model = SentenceTransformer(sentence_model_name, device=device)

        # Adding this class in the beginning since the labels are 1-indexed.
        text_strings = [self.EMPTY_CLASS]
        for name in class_names:
            text_strings.append(self.PROMPT + name.replace("-", " ").replace("_", " "))
        with torch.no_grad():
            all_embedded_text = sentence_model.encode(text_strings)
            all_embedded_text = torch.from_numpy(all_embedded_text).float().to(device)

        with torch.no_grad():
            text = clip.tokenize(text_strings).to(device)
            clip_encoded_text = clip_model.encode_text(text).float().to(device)

        del clip_model
        del sentence_model

        self.total_label_classes = len(text_strings)
        self._sentence_embed_size = all_embedded_text.size(-1)
        self._clip_embed_size = clip_encoded_text.size(-1)

        self._sentence_features = F.normalize(all_embedded_text, p=2, dim=-1)
        self._clip_text_features = F.normalize(clip_encoded_text, p=2, dim=-1)

    def calculate_classifications(
        self, model_text_features: torch.Tensor, model_image_features: torch.Tensor
    ):
        # Figure out the classification given the learned embedding of the objects.
        assert model_text_features.size(-1) == self._sentence_embed_size
        assert model_image_features.size(-1) == self._clip_embed_size

        # Now do the softmax over the classes.
        model_text_features = F.normalize(model_text_features, p=2, dim=-1)
        model_image_features = F.normalize(model_image_features, p=2, dim=-1)

        with torch.no_grad():
            text_logits = model_text_features @ self._sentence_features.T
            image_logits = model_image_features @ self._clip_text_features.T

        assert text_logits.size(-1) == self.total_label_classes
        assert image_logits.size(-1) == self.total_label_classes

        # Figure out sum of probabilities.
        return (
            F.softmax(self.LOGIT_TEMP * text_logits, dim=-1)
            + F.softmax(self.LOGIT_TEMP * image_logits, dim=-1)
        ) / 2.0
