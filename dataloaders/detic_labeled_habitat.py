import logging
import clip
import einops
import os
import torch
import tqdm

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from dataloaders.habitat_loaders import HabitatViewDataset
from dataloaders.scannet_200_classes import CLASS_LABELS_200
from sentence_transformers import SentenceTransformer

# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset

setup_logger()

# import some common libraries
import sys

# import some common detectron2 utilities
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor


DETIC_PATH = os.environ.get("DETIC_PATH", Path.home() / "code/Detic")
LSEG_PATH = os.environ.get(
    "LSEG_PATH", Path.home() / "code/geometric_database/lang-seg/"
)
# from google.colab.patches import cv2_imshow
sys.path.insert(0, f"{LSEG_PATH}/")
from encoding.models.sseg import BaseNet
from additional_utils.models import LSeg_MultiEvalModule
from modules.lseg_module import LSegModule
import torchvision.transforms as transforms

# Detic libraries
sys.path.insert(0, f"{DETIC_PATH}/third_party/CenterNet2/")
sys.path.insert(0, f"{DETIC_PATH}/")
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test
from detic.modeling.text.text_encoder import build_text_encoder

cfg = get_cfg()
add_centernet_config(cfg)
add_detic_config(cfg)
cfg.merge_from_file(
    f"{DETIC_PATH}/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
)
cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand"
cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = (
    False  # For better visualization purpose. Set to False for all classes.
)
cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = (
    f"{DETIC_PATH}/datasets/metadata/lvis_v1_train_cat_info.json"
)
# cfg.MODEL.DEVICE='cpu' # uncomment this to use cpu-only mode.

# Setup the model's vocabulary using build-in datasets

BUILDIN_CLASSIFIER = {
    "lvis": f"{DETIC_PATH}/datasets/metadata/lvis_v1_clip_a+cname.npy",
    "objects365": f"{DETIC_PATH}/datasets/metadata/o365_clip_a+cnamefix.npy",
    "openimages": f"{DETIC_PATH}/datasets/metadata/oid_clip_a+cname.npy",
    "coco": f"{DETIC_PATH}/datasets/metadata/coco_clip_a+cname.npy",
}

BUILDIN_METADATA_PATH = {
    "lvis": "lvis_v1_val",
    "objects365": "objects365_v2_val",
    "openimages": "oid_val_expanded",
    "coco": "coco_2017_val",
}

vocabulary = "lvis"  # change to 'lvis', 'objects365', 'openimages', or 'coco'
metadata = MetadataCatalog.get(BUILDIN_METADATA_PATH[vocabulary])


def get_clip_embeddings(vocabulary, prompt="a "):
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    texts = [prompt + x.replace("-", " ") for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb


class DeticDenseLabelledDataset(Dataset):
    LSEG_LABEL_WEIGHT = 0.1
    LSEG_IMAGE_DISTANCE = 10.0

    def __init__(
        self,
        habitat_view_dataset: HabitatViewDataset,
        clip_model_name: str = "ViT-B/32",
        sentence_encoding_model_name="all-mpnet-base-v2",
        device: str = "cuda",
        batch_size: int = 1,
        detic_threshold: float = 0.3,
        num_images_to_label: int = 300,
        subsample_prob: float = 0.2,
    ):
        dataset = habitat_view_dataset

        habitat_view_data = (
            habitat_view_dataset.dataset
            if isinstance(habitat_view_dataset, torch.utils.data.Subset)
            else habitat_view_dataset
        )
        self._image_width, self._image_height = habitat_view_data.image_size
        clip_model, _ = clip.load(clip_model_name, device=device)
        sentence_model = SentenceTransformer(sentence_encoding_model_name)

        self._batch_size = batch_size
        self._device = device
        self._detic_threshold = detic_threshold
        self._subsample_prob = subsample_prob

        self._label_xyz = []
        self._label_rgb = []
        self._label_weight = []
        self._label_idx = []
        self._text_ids = []
        self._text_id_to_feature = {}
        self._image_features = []
        self._distance = []
        # Now, set up all the points and their labels.
        images_to_label = self.get_best_sem_segmented_images(
            dataset, num_images_to_label
        )
        # First, setup detic with the combined classes.
        self._setup_detic_all_classes(habitat_view_data)
        self._setup_detic_dense_labels(
            dataset, images_to_label, clip_model, sentence_model
        )

    def get_best_sem_segmented_images(self, dataset, num_images_to_label=300):
        # Using depth as a proxy for object diversity in a scene.
        num_objects_and_images = [
            (dataset[idx]["depth"].max() - dataset[idx]["depth"].min(), idx)
            for idx in range(len(dataset))
        ]
        sorted_num_object_and_img = sorted(
            num_objects_and_images, key=lambda x: x[0], reverse=True
        )
        return [x[1] for x in sorted_num_object_and_img[:num_images_to_label]]

    @torch.no_grad()
    def _setup_detic_dense_labels(
        self, dataset, images_to_label, clip_model, sentence_model
    ):
        # Now just iterate over the images and do Detic preprocessing.
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=False)
        label_idx = 0
        for idx, data_dict in tqdm.tqdm(enumerate(dataloader), total=len(dataset)):
            if idx not in images_to_label:
                continue
            rgb = einops.rearrange(data_dict["rgb"][..., :3], "b h w c -> b c h w")
            xyz = data_dict["xyz_position"]
            for image, coordinates in zip(rgb, xyz):
                # Now calculate the Detic classification for this.
                with torch.no_grad():
                    result = self._predictor.model(
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
                reshaped_coordinates = einops.rearrange(coordinates, "c h w -> h w c")
                reshaped_rgb = einops.rearrange(image, "c h w -> h w c")
                for pred_class, pred_mask, pred_score, feature in zip(
                    instance.pred_classes.cpu(),
                    instance.pred_masks.cpu(),
                    instance.scores.cpu(),
                    instance.features.cpu(),
                ):
                    # Go over each instance and add it to the DB.
                    total_points = len(reshaped_coordinates[pred_mask])
                    self._label_xyz.append(reshaped_coordinates[pred_mask])
                    self._label_rgb.append(reshaped_rgb[pred_mask])
                    self._text_ids.append(torch.ones(total_points) * pred_class)
                    self._label_weight.append(torch.ones(total_points) * pred_score)
                    self._image_features.append(
                        einops.repeat(feature, "d -> b d", b=total_points)
                    )
                    self._label_idx.append(torch.ones(total_points) * label_idx)
                    self._distance.append(torch.zeros(total_points))
                    label_idx += 1

        # First delete leftover Detic predictors
        del self._predictor
        # Now, get to LSeg
        self._setup_lseg()
        for idx, data_dict in tqdm.tqdm(enumerate(dataloader), total=len(dataset)):
            if idx not in images_to_label:
                continue
            rgb = einops.rearrange(data_dict["rgb"][..., :3], "b h w c -> b c h w")
            xyz = data_dict["xyz_position"]
            for image, coordinates in zip(rgb, xyz):
                # Now figure out the LSeg lables.
                with torch.no_grad():
                    unsqueezed_image = image.unsqueeze(0).cuda()
                    resized_image = self.resize(image).unsqueeze(0).cuda()
                    tfm_image = self.transform(unsqueezed_image)
                    outputs = self.evaluator.parallel_forward(
                        tfm_image, self._all_lseg_classes
                    )
                    image_feature = clip_model.encode_image(resized_image).squeeze(0)
                    image_feature = image_feature.cpu()
                    predicts = [torch.max(output, 1)[1].cpu() for output in outputs]
                predict = predicts[0]

                reshaped_coordinates = einops.rearrange(coordinates, "c h w -> h w c")
                reshaped_rgb = einops.rearrange(image, "c h w -> h w c")

                for label in range(self._num_true_lseg_classes):
                    pred_mask = predict.squeeze(0) == label
                    total_points = len(reshaped_coordinates[pred_mask])
                    if total_points:
                        class_text_id = self._lseg_class_labels[
                            self._all_lseg_classes[label]
                        ]
                        self._label_xyz.append(reshaped_coordinates[pred_mask])
                        self._label_rgb.append(reshaped_rgb[pred_mask])
                        # Ideally, this should give all classes their true class label.
                        self._text_ids.append(torch.ones(total_points) * class_text_id)
                        # Uniform label confidence of LSEG_LABEL_WEIGHT
                        self._label_weight.append(
                            torch.ones(total_points) * self.LSEG_LABEL_WEIGHT
                        )
                        self._image_features.append(
                            einops.repeat(image_feature, "d -> b d", b=total_points)
                        )
                        self._label_idx.append(torch.ones(total_points) * label_idx)
                        self._distance.append(
                            torch.ones(total_points) * self.LSEG_IMAGE_DISTANCE
                        )
                # Since they all get the same image, here label idx is increased once
                # at the very end.
                label_idx += 1

        # Now, delete the module and the evaluator
        del self.evaluator
        del self.module
        del self.transform

        # Now, get all the sentence encoding for all the labels.
        text_strings = [
            x.replace("-", " ").replace("_", " ") for x in self._all_classes
        ]
        text_strings += self._lseg_classes
        with torch.no_grad():
            all_embedded_text = sentence_model.encode(text_strings)
            all_embedded_text = torch.from_numpy(all_embedded_text).float()

        for i, feature in enumerate(all_embedded_text):
            self._text_id_to_feature[i] = feature

        # Now, we map from label to text using this model.
        self._label_xyz = torch.cat(self._label_xyz).float()
        self._label_rgb = torch.cat(self._label_rgb).float()
        self._label_weight = torch.cat(self._label_weight).float()
        self._image_features = torch.cat(self._image_features).float()
        self._text_ids = torch.cat(self._text_ids).long()
        self._label_idx = torch.cat(self._label_idx).long()
        self._distance = torch.cat(self._distance).float()
        self._instance = (
            torch.ones_like(self._text_ids) * -1
        ).long()  # We don't have instance ID from this dataset.

        self._resample()

        print(len(self._label_xyz))

    def _resample(self):
        resampled_indices = torch.rand(len(self._label_xyz)) < self._subsample_prob
        logging.info(
            f"Resampling dataset down from {len(self._label_xyz)} points to {resampled_indices.long().sum().item()} points."
        )
        self._label_xyz = self._label_xyz[resampled_indices]
        self._label_rgb = self._label_rgb[resampled_indices]
        self._label_weight = self._label_weight[resampled_indices]
        self._image_features = self._image_features[resampled_indices]
        self._text_ids = self._text_ids[resampled_indices]
        self._label_idx = self._label_idx[resampled_indices]
        self._distance = self._distance[resampled_indices]
        self._instance = self._instance[resampled_indices]

    def __getitem__(self, idx):
        # Create a dictionary with all relevant results.
        return {
            "xyz": self._label_xyz[idx].float(),
            "rgb": self._label_rgb[idx].float(),
            "label": self._text_ids[idx].long(),
            "instance": self._instance[idx].long(),
            "img_idx": self._label_idx[idx].long(),
            "distance": self._distance[idx].float(),
            "clip_vector": self._text_id_to_feature.get(
                self._text_ids[idx].item()
            ).float(),
            "clip_image_vector": self._image_features[idx].float(),
            "semantic_weight": self._label_weight[idx].float(),
        }

    def __len__(self):
        return len(self._label_xyz)

    def _setup_detic_all_classes(self, habitat_view_data: HabitatViewDataset):
        # Unifying all the class labels.
        predictor = DefaultPredictor(cfg)
        prebuilt_class_names = list(habitat_view_data._id_to_name.values())
        prebuilt_class_set = set(prebuilt_class_names)
        filtered_new_classes = [
            x for x in CLASS_LABELS_200 if x not in prebuilt_class_set
        ]
        self._all_classes = ["Other"] + prebuilt_class_names + filtered_new_classes
        self._all_classes = [
            x.replace("-", " ").replace("_", " ").lower() for x in self._all_classes
        ]
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
        self._predictor = predictor

    def find_in_class(self, classname):
        try:
            return self._all_classes.index(classname)
        except ValueError:
            ret_value = len(self._all_classes) + self._unfound_offset
            self._unfound_offset += 1
            return ret_value

    def _setup_lseg(self):
        self._lseg_classes = self._all_classes
        self._num_true_lseg_classes = len(self._lseg_classes)
        self._all_lseg_classes = self._all_classes  # + ["Other"]

        self._unfound_offset = 0
        # Figure out the class labels.
        self._lseg_class_labels = {
            classname: self.find_in_class(classname)
            for classname in self._all_classes
        }
        # We will try to classify all the classes, but will use LSeg labels for classes that
        # are not identified by Detic.

        self.module = LSegModule.load_from_checkpoint(
            checkpoint_path=f"{LSEG_PATH}/checkpoints/demo_e200.ckpt",
            data_path="",
            dataset="ade20k",
            backbone="clip_vitl16_384",
            aux=False,
            num_features=256,
            aux_weight=0,
            se_loss=False,
            se_weight=0,
            base_lr=0,
            batch_size=1,
            max_epochs=0,
            ignore_index=255,
            dropout=0.0,
            scale_inv=False,
            augment=False,
            no_batchnorm=False,
            widehead=True,
            widehead_hr=False,
            map_locatin=self._device,
            arch_option=0,
            block_depth=0,
            activation="lrelu",
        )

        # model
        if isinstance(self.module.net, BaseNet):
            model = self.module.net
        else:
            model = self.module

        model = model.eval()
        model = model.to(self._device)
        self.scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]

        model.mean = [0.5, 0.5, 0.5]
        model.std = [0.5, 0.5, 0.5]

        self.transform = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.resize = transforms.Resize(224)
        self.resize_coords = transforms.Resize(
            224, interpolation=transforms.InterpolationMode.NEAREST
        )

        self.evaluator = LSeg_MultiEvalModule(model, scales=self.scales, flip=True).to(
            self._device
        )
        self.evaluator = self.evaluator.eval()
