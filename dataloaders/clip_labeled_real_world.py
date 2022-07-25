from turtle import back
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from pyntcloud import PyntCloud
import clip
from sentence_transformers import SentenceTransformer, util
import json
import numpy as np
import os
import glob
from PIL import Image
from torch.utils.data import Dataset
from typing import Optional

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import sys
import numpy as np
import os, json, cv2, random

# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# Detic libraries
sys.path.insert(0, "/private/home/notmahi/code/Detic/third_party/CenterNet2/")
sys.path.insert(0, "/private/home/notmahi/code/Detic/")
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test


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


def rotate_image(image):
    if isinstance(image, np.ndarray):
        return np.rot90(image, k=1, axes=(0, 1))
    else:
        return torch.rot90(image, 1, [0, 1])


class RealWorldSemanticDataset:
    CLIP_PROMPT = "A "
    FLOOR_CUTOFF = 0.05
    BG_LABEL_WEIGHT = 0.25

    def __init__(
        self,
        base_paths,
        id_to_class=metadata.thing_classes,
        device="cuda",
        clip_model_name="ViT-L/14",
        sentence_encoding_model_name="all-mpnet-base-v2",
    ):
        if isinstance(base_paths, str):
            base_paths = [base_paths]
        self._base_paths = base_paths

        self._surface_xyzs = []
        self._labelled_xyzs = []
        self._rgbs = []
        self._semantic_class_id = []
        self._image_id = []
        self._image_id_for_label = []
        self._rgb_image = []
        self._semantic_weight = []
        self._depths = []
        self._label_depths = []
        self._class_labels = []
        self._class_clip_vectors = []

        self._id_to_class_name = id_to_class
        # Add ad-hoc labels.
        self._id_to_class_name[-1] = "background"
        self._id_to_class_name[-2] = "floor"
        self._image_idx_to_camera = {}
        self._all_possible_ids = set([-1, -2])

        self._clip_model_name = clip_model_name
        self._sentence_encoding_model_name = sentence_encoding_model_name
        self._device = device

        self._build_dataset()

    def _build_dataset(self):
        # Build the entire dataset by crawling the directories, using pointcloud XYZs
        # as well as the DETIC semantic segmentation library to get the segmentations
        for base_path in self._base_paths:
            subdir = "modular_learned/trajectory/"
            full_dir = os.path.join(base_path, subdir)
            # Now go over each directory and join them in the dataset.
            with open(
                os.path.join(base_path, "modular_learned", "aggregate_logs.json")
            ) as f:
                step_logs = json.load(f)
            image_idx = 0
            for path in tqdm.tqdm(os.listdir(full_dir)):
                current_path = os.path.join(full_dir, path, "frames")
                step_id = int(path[len("step") :])
                # First, get the depth frame
                depth = rotate_image(np.load(os.path.join(current_path, "depth.npy")))
                depth_valid = depth > 0  # Rest of the points do not map to pointcloud]
                self._depths.append(depth[depth_valid])

                # And get the pointcloud
                pcd = np.load(os.path.join(current_path, "pcd.npy"))
                self._surface_xyzs.append(pcd)

                image_id = np.ones_like(depth[depth_valid]) * image_idx
                self._image_id.append(image_id)
                self._image_idx_to_camera[image_idx] = torch.tensor(
                    step_logs["poses"][step_id - 1]
                )

                rgb = Image.open(os.path.join(current_path, "rgb.png"))
                rgb_array = rotate_image(np.asarray(rgb))
                self._rgbs.append(rgb_array[depth_valid])
                self._rgb_image.append(rgb)

                with torch.no_grad():
                    # First, run detic on top of it.
                    im = cv2.imread(os.path.join(current_path, "rgb.png"))
                    outputs = predictor(im)

                torch_pcd = torch.from_numpy(pcd)
                torch_depth_valid = torch.from_numpy(depth[depth_valid])
                # Create background mask
                background_mask = torch.ones_like(torch_depth_valid)
                # Now figure out each label and label mask.
                instance = outputs["instances"]
                for pred_class, pred_mask, pred_score in zip(
                    instance.pred_classes.cpu(),
                    instance.pred_masks.cpu(),
                    instance.scores.cpu(),
                ):
                    rotated_pred_mask = rotate_image(pred_mask)
                    mask_flattened = rotated_pred_mask[depth_valid]
                    labeled_xyzs = torch_pcd[mask_flattened]
                    masked_depth = torch_depth_valid[mask_flattened]
                    background_mask[mask_flattened] = 0.0
                    label = torch.ones_like(masked_depth) * pred_class
                    score = torch.ones_like(masked_depth) * pred_score
                    image_id_for_label = torch.ones_like(masked_depth) * image_idx

                    self._labelled_xyzs.append(labeled_xyzs)
                    self._semantic_class_id.append(label)
                    self._semantic_weight.append(score)
                    self._image_id_for_label.append(image_id_for_label)
                    self._label_depths.append(masked_depth)

                # Now add the background points to the dataset
                background_mask = background_mask == 1
                labeled_xyzs = torch_pcd[background_mask]
                masked_depth = torch_depth_valid[background_mask]
                label = (
                    torch.ones_like(masked_depth) * -1
                )  # Pred class is -1 for background
                score = (
                    torch.ones_like(masked_depth) * self.BG_LABEL_WEIGHT
                )  # Pred class is -1 for background
                image_id_for_label = torch.ones_like(masked_depth) * image_idx

                self._labelled_xyzs.append(labeled_xyzs)
                self._semantic_class_id.append(label)
                self._semantic_weight.append(score)
                self._image_id_for_label.append(image_id_for_label)
                self._label_depths.append(masked_depth)

                # Finally, add the floor/navigable space.
                floor_mask = torch.logical_and(
                    background_mask, torch_pcd[:, 2] <= self.FLOOR_CUTOFF
                )

                labeled_xyzs = torch_pcd[floor_mask]
                masked_depth = torch_depth_valid[floor_mask]
                label = (
                    torch.ones_like(masked_depth) * -2
                )  # Pred class is -1 for background
                score = (
                    torch.ones_like(masked_depth) * self.BG_LABEL_WEIGHT
                )  # Pred class is -1 for background
                image_id_for_label = torch.ones_like(masked_depth) * image_idx

                self._labelled_xyzs.append(labeled_xyzs)
                self._semantic_class_id.append(label)
                self._semantic_weight.append(score)
                self._image_id_for_label.append(image_id_for_label)
                self._label_depths.append(masked_depth)

                # Add the possible ids from this image
                self._all_possible_ids = self._all_possible_ids | set(
                    instance.pred_classes.cpu().tolist()
                )
                image_idx += 1

        self._depths = np.concatenate(self._depths)
        self._rgbs = np.concatenate(self._rgbs) / 255.0
        self._surface_xyzs = np.concatenate(self._surface_xyzs)
        self._image_id = np.concatenate(self._image_id)

        self._labelled_xyzs = torch.cat(self._labelled_xyzs)
        self._semantic_class_id = torch.cat(self._semantic_class_id)
        self._semantic_weight = torch.cat(self._semantic_weight)
        self._image_id_for_label = torch.cat(self._image_id_for_label)
        self._label_depths = torch.cat(self._label_depths)

        self._build_clip_rep()

    def _build_clip_rep(
        self,
        clip_model_name: Optional[str] = None,
        sentence_encoding_model_name: Optional[str] = None,
        batch_size: int = 128,
    ):
        text_strings = []
        if clip_model_name is None:
            clip_model_name = self._clip_model_name
        if sentence_encoding_model_name is None:
            sentence_encoding_model_name = self._sentence_encoding_model_name
        clip_model, preprocess = clip.load(clip_model_name, device=self._device)
        all_used_id_to_name = {}
        all_used_id_to_vector = {}
        for class_id in self._all_possible_ids:
            name = self._id_to_class_name[class_id]
            all_used_id_to_name[class_id] = name
            text_strings.append(self.CLIP_PROMPT + name.replace("_", " "))
        # text_strings.append(self.EMPTY)
        clip_tokens = clip.tokenize(text_strings)
        # Now, encode them into Clip vectors.
        all_embedded_text = []
        with torch.no_grad():
            for i in range(0, len(clip_tokens), batch_size):
                start, end = i, min(i + batch_size, len(clip_tokens))
                batch_data = clip_tokens[start:end].to(self._device)
                embedded_text = clip_model.encode_text(batch_data).float()
                embedded_text = F.normalize(embedded_text, p=2, dim=-1)
                all_embedded_text.append(embedded_text.cpu())
        all_embedded_text = torch.cat(all_embedded_text)
        for class_id, embed in zip(all_used_id_to_name.keys(), all_embedded_text):
            all_used_id_to_vector[class_id] = embed

        self._label_id_to_vector = all_used_id_to_vector
        # Now figure out the RGB image embeddings.
        all_rgb_images = torch.stack([preprocess(x) for x in self._rgb_image]).to(
            self._device
        )
        all_embedded_image = []
        with torch.no_grad():
            for i in range(0, len(all_rgb_images), batch_size):
                start, end = i, min(i + batch_size, len(all_rgb_images))
                batch_data = all_rgb_images[start:end]
                embedded_image = clip_model.encode_image(batch_data).float()
                embedded_image = F.normalize(embedded_image, p=2, dim=-1)
                all_embedded_image.append(embedded_image.cpu())
        all_embedded_image = torch.cat(all_embedded_image)
        self._image_idx_to_vector = {
            idx: embed for (idx, embed) in enumerate(all_embedded_image)
        }

    def export_to_rgb_pcd(self):
        # Export the RGB points to pointcloud format.
        df = pd.DataFrame(
            data=np.concatenate([self._surface_xyzs, self._rgbs], axis=-1),
            columns=["x", "y", "z", "red", "green", "blue"],
        )
        return PyntCloud(df)


class RealWorldSurfaceDataset(Dataset):
    def __init__(
        self, parent_dataset: RealWorldSemanticDataset, sampling_rate: float = 0.01
    ):
        # Initialize the surface dataset from the parent dataset.
        self._full_surface_xyzs = parent_dataset._surface_xyzs
        self._image_idx_to_camera = parent_dataset._image_idx_to_camera
        self._full_image_id = parent_dataset._image_id

        assert len(self._full_surface_xyzs) == len(self._full_image_id)

        self.resample(sampling_rate)

    def resample(self, sampling_rate: float):
        count = len(self._full_surface_xyzs)
        indices = torch.rand(count) < sampling_rate
        self._image_id = self._full_image_id[indices]
        self._surface_xyzs = self._full_surface_xyzs[indices]

    def __len__(self):
        return len(self._surface_xyzs)

    def __getitem__(self, idx: int):
        result = {
            "xyz_position": self._surface_xyzs[idx],
            "camera_pos": self._image_idx_to_camera.get(self._image_id[idx]),
        }
        return result


class RealWorldClipDataset(Dataset):
    def __init__(
        self, parent_dataset: RealWorldSemanticDataset, sampling_rate: float = 0.01
    ):
        self._full_labelled_xyzs = parent_dataset._labelled_xyzs
        self._label_id_to_vector = parent_dataset._label_id_to_vector
        self._full_semantic_class_id = parent_dataset._semantic_class_id
        self._full_semantic_weight = parent_dataset._semantic_weight
        self._image_idx_to_vector = parent_dataset._image_idx_to_vector
        self._full_image_id_for_label = parent_dataset._image_id_for_label
        self._full_label_depths = parent_dataset._label_depths

        assert len(self._full_labelled_xyzs) == len(self._full_semantic_class_id)
        assert len(self._full_image_id_for_label) == len(self._full_labelled_xyzs)
        assert len(self._full_image_id_for_label) == len(self._full_label_depths)
        assert len(self._full_image_id_for_label) == len(self._full_semantic_weight)

        self.resample(sampling_rate)

    def resample(self, sampling_rate: float):
        count = len(self._full_labelled_xyzs)
        indices = torch.rand(count) < sampling_rate
        self._labelled_xyzs = self._full_labelled_xyzs[indices]
        self._semantic_class_id = self._full_semantic_class_id[indices]
        self._image_id_for_label = self._full_image_id_for_label[indices]
        self._label_depths = self._full_label_depths[indices]
        self._semantic_weight = self._full_semantic_weight[indices]

    def __len__(self):
        return len(self._labelled_xyzs)

    def __getitem__(self, idx):
        # Create a dictionary with all relevant results.
        return {
            "xyz": self._labelled_xyzs[idx],
            "clip_vector": self._label_id_to_vector.get(
                self._semantic_class_id[idx].item()
            ),
            "clip_image_vector": self._image_idx_to_vector.get(
                self._image_id_for_label[idx].item()
            ),
            "semantic_weight": self._semantic_weight[idx],
            "label": self._semantic_class_id[idx],
            "img_idx": self._image_id_for_label[idx],
            "distance": self._label_depths[idx],
        }


def get_voxel_normalized_sampler_and_occupied_voxels(xyz_locations, voxel_size=0.01):
    # Take in the XYZ locations, and return the voxel normalized weights.
    voxel_counts = {}
    for data_point in tqdm.tqdm(xyz_locations):
        point = data_point.get("xyz")
        if point is None:
            point = data_point.get("xyz_position")
        if isinstance(point, torch.Tensor):
            point = point.numpy()
        smoothed_point = (point / voxel_size).astype(int)
        voxel_counts[tuple(smoothed_point)] = (
            voxel_counts.get(tuple(smoothed_point), 0) + 1
        )
    weight = []
    for point in tqdm.tqdm(xyz_locations):
        point = data_point.get("xyz")
        if point is None:
            point = data_point.get("xyz_position")
        if isinstance(point, torch.Tensor):
            point = point.numpy()
        smoothed_point = (point / voxel_size).astype(int)
        weight.append(1 / voxel_counts[tuple(smoothed_point)])

    return weight, len(voxel_counts)
