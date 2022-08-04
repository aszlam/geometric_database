from typing import Dict
import clip
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from dataloaders.habitat_loaders import HabitatLocationDataset, HabitatViewDataset
from sentence_transformers import SentenceTransformer


import glob
import json
import os
from typing import Optional

import clip

# Some basic setup:
# Setup detectron2 logger
import detectron2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from detectron2.utils.logger import setup_logger
from PIL import Image
from pyntcloud import PyntCloud
from sentence_transformers import SentenceTransformer, util
from torch.utils.data import Dataset

setup_logger()

import json
import os
import random

# import some common libraries
import sys

import cv2
import numpy as np

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

# from google.colab.patches import cv2_imshow

# Detic libraries
sys.path.insert(0, "/private/home/notmahi/code/Detic/third_party/CenterNet2/")
sys.path.insert(0, "/private/home/notmahi/code/Detic/")
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test
from detic.modeling.text.text_encoder import build_text_encoder

cfg = get_cfg()
add_centernet_config(cfg)
add_detic_config(cfg)
cfg.merge_from_file(
    "/private/home/notmahi/code/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
)
cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand"
cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = (
    False  # For better visualization purpose. Set to False for all classes.
)
cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = (
    "/private/home/notmahi/code/Detic/datasets/metadata/lvis_v1_train_cat_info.json"
)
# cfg.MODEL.DEVICE='cpu' # uncomment this to use cpu-only mode.
predictor = DefaultPredictor(cfg)

# Setup the model's vocabulary using build-in datasets

BUILDIN_CLASSIFIER = {
    "lvis": "/private/home/notmahi/code/Detic/datasets/metadata/lvis_v1_clip_a+cname.npy",
    "objects365": "/private/home/notmahi/code/Detic/datasets/metadata/o365_clip_a+cnamefix.npy",
    "openimages": "/private/home/notmahi/code/Detic/datasets/metadata/oid_clip_a+cname.npy",
    "coco": "/private/home/notmahi/code/Detic/datasets/metadata/coco_clip_a+cname.npy",
}

BUILDIN_METADATA_PATH = {
    "lvis": "lvis_v1_val",
    "objects365": "objects365_v2_val",
    "openimages": "oid_val_expanded",
    "coco": "coco_2017_val",
}

vocabulary = "lvis"  # change to 'lvis', 'objects365', 'openimages', or 'coco'
metadata = MetadataCatalog.get(BUILDIN_METADATA_PATH[vocabulary])
classifier = BUILDIN_CLASSIFIER[vocabulary]
num_classes = len(metadata.thing_classes)
reset_cls_test(predictor.model, classifier, num_classes)


def get_clip_embeddings(vocabulary, prompt="a "):
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    texts = [prompt + x.replace("-", " ") for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb


class DeticDenseLabelledDataset(Dataset):
    def __init__(
        self,
        habitat_view_dataset: HabitatViewDataset,
        clip_model_name: str = "ViT-B/32",
        sentence_encoding_model_name="all-mpnet-base-v2",
        device: str = "cuda",
        batch_size: int = 128,
        detic_threshold: float = 0.3,
    ):
        self._dataset = habitat_view_dataset

        self.habitat_view_data = (
            habitat_view_dataset.dataset
            if isinstance(habitat_view_dataset, torch.utils.data.Subset)
            else habitat_view_dataset
        )
        self._image_width, self._image_height = self.habitat_view_data.image_size
        self._clip_model, _ = clip.load(clip_model_name, device=device)
        self._sentence_model = SentenceTransformer(sentence_encoding_model_name)

        self._batch_size = batch_size
        self._device = device
        self._detic_threshold = detic_threshold

        self._label_xyz = []
        self._label_rgb = []
        self._label_weight = []
        self._label_idx = []
        self._text_ids = []
        self._text_id_to_feature = {}
        self._image_features = []
        # Now, set up all the points and their labels.
        self._setup_detic_dense_labels()

    def _setup_detic_dense_labels(self):
        # First, setup detic with the combined classes.
        self._setup_detic_all_classes()
        # Now just iterate over the images and do Detic preprocessing.
        dataloader = DataLoader(
            self._dataset, batch_size=self._batch_size, shuffle=False, pin_memory=False
        )
        label_idx = 0
        with torch.no_grad():
            for data_dict in tqdm.tqdm(dataloader):
                rgb = einops.rearrange(data_dict["rgb"][..., :3], "b h w c -> b c h w")
                xyz = data_dict["xyz_position"]
                for image, coordinates in zip(rgb, xyz):
                    # Now calculate the Detic classification for this.
                    result = predictor.model(
                        [
                            {
                                "image": image * 255,
                                "height": self._image_height,
                                "width": self._image_width,
                            }
                        ]
                    )[0]
                    # Now extract the results from the image and store them
                    instance = result["instances"]
                    for pred_class, pred_mask, pred_score, feature in zip(
                        instance.pred_classes.cpu(),
                        instance.pred_masks.cpu(),
                        instance.scores.cpu(),
                        instance.features.cpu(),
                    ):
                        # Go over each instance and add it to the DB.
                        reshaped_coordinates = einops.rearrange(
                            coordinates, "c h w -> h w c"
                        )
                        reshaped_rgb = einops.rearrange(image, "c h w -> h w c")
                        total_points = len(reshaped_coordinates[pred_mask])
                        self._label_xyz.append(reshaped_coordinates[pred_mask])
                        self._label_rgb.append(reshaped_rgb[pred_mask])
                        self._text_ids.append(torch.ones(total_points) * pred_class)
                        self._label_weight.append(torch.ones(total_points) * pred_score)
                        self._image_features.append(
                            einops.repeat(feature, "d -> b d", b=total_points)
                        )
                        self._label_idx.append(torch.ones(total_points) * label_idx)
                        label_idx += 1

        # Now, get all the sentence encoding for all the labels.
        text_strings = [
            x.replace("-", " ").replace("_", " ") for x in self._all_classes
        ]
        with torch.no_grad():
            all_embedded_text = self._sentence_model.encode(text_strings)
            all_embedded_text = torch.from_numpy(all_embedded_text).float()

        for i, feature in enumerate(all_embedded_text):
            self._text_id_to_feature[i] = feature

        # Now, we map from label to text using this model.
        self._label_xyz = torch.cat(self._label_xyz)
        self._label_rgb = torch.cat(self._label_rgb)
        self._label_weight = torch.cat(self._label_weight)
        self._image_features = torch.cat(self._image_features)
        self._text_ids = torch.cat(self._text_ids)
        self._label_idx = torch.cat(self._label_idx)
        self._distance = torch.zeros_like(
            self._text_ids
        ).float()  # Image weight is always 1.
        self._instance = (
            torch.ones_like(self._text_ids) * -1
        ).long()  # We don't have instance ID from this dataset.

        print(len(self._label_xyz))

    def __getitem__(self, idx):
        # Create a dictionary with all relevant results.
        return {
            "xyz": self._label_xyz[idx],
            "rgb": self._label_rgb[idx],
            "label": self._text_ids[idx],
            "instance": self._instance[idx],
            "img_idx": self._label_idx[idx],
            "distance": self._distance[idx],
            "clip_vector": self._text_id_to_feature.get(self._text_ids[idx].item()),
            "clip_image_vector": self._image_features[idx],
            "semantic_weight": self._label_weight[idx],
        }

    def __len__(self):
        return len(self._label_xyz)

    def _setup_detic_all_classes(self):
        vocabulary = "custom"
        self._all_classes = metadata.thing_classes + list(
            self.habitat_view_data._id_to_name.values()
        )
        new_metadata = MetadataCatalog.get("__unused")
        new_metadata.thing_classes = self._all_classes
        classifier = get_clip_embeddings(new_metadata.thing_classes)
        num_classes = len(new_metadata.thing_classes)
        reset_cls_test(predictor.model, classifier, num_classes)
        # Reset visualization threshold
        output_score_threshold = self._detic_threshold
        for cascade_stages in range(len(predictor.model.roi_heads.box_predictor)):
            predictor.model.roi_heads.box_predictor[
                cascade_stages
            ].test_score_thresh = output_score_threshold
