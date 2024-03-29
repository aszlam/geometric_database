import glob
import logging
import os
import random
from typing import Dict, Optional

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import tqdm
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler, random_split

import wandb
from dataloaders.clip_labeled_habitat import (
    ClassificationExtractor,
    ClipLabelledLocation,
)
from dataloaders.clip_labeled_real_world import (
    RealWorldClipDataset,
    RealWorldSemanticDataset,
    RealWorldSurfaceDataset,
)
from dataloaders.detic_labeled_habitat import DeticDenseLabelledDataset
from dataloaders.habitat_loaders import HabitatLocationDataset, HabitatViewDataset
from implicit_models.grid_hash_model import GridCLIPModel
from implicit_models.implicit_mlp import ImplicitCLIPModel
from implicit_models.implicit_dataparallel import ImplicitDataparallel

SCENE = "apartment_0"
# Replace with the path to your scene file
SCENE_FILEPATH = [
    f"{Path.home()}/data/replica_dataset/{SCENE}/habitat/mesh_semantic.ply",
]

REAL_SCENE_DIRECTORY = glob.glob(
    f"{Path.home()}/data/stretch_fairmont/trajectories/july10_fremont/bed2/"
)
BATCH_SIZE = 256

SAVE_DIRECTORY = f"habitat_{SCENE}_larger_model_detic_lseg_labels"

# Create model using a simple MLP and a Fourier projection.
# This model should really tell you the probability of something being a surface point or not.
DEVICE = "cuda"
IMAGE_BATCH_SIZE = 32 * 7 * 7
IMAGE_SIZE = 224
POINT_BATCH_SIZE = 256 * 7 * 7
IMAGE_TO_LABEL_CLIP_LOSS_SCALE = 1.0
LABEL_TO_IMAGE_LOSS_SCALE = 1.0
EXP_DECAY_COEFF = 0.5
EPOCHS = 500
SUBSAMPLE_PROB = 0.2
EVAL_EVERY = 5
SURFACE_LOSS_LAMBDA = 10.0
INSTANCE_LOSS_SCALE = 5.0
GRID_SIZE = 8
NUM_INSTANCE_SEGMENTED_IMAGES = 5
NUM_SEM_SEGMENTED_IMAGES = 50
NUM_WEB_SEGMENTED_IMAGES = 300
GT_SEMANTIC_WEIGHT = 10

MODEL_TYPE = "hash"  # MLP or hash
CACHE = True
TOP_K = 3

# Set up the desired metrics.
METRICS = {
    "accuracy": torchmetrics.Accuracy,
    "precision": torchmetrics.Precision,
    "recall": torchmetrics.Recall,
}

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def train(
    clip_train_loader,
    labelling_model,
    optim,
    classifier: ClassificationExtractor,
    device=DEVICE,
    exp_decay_coeff=EXP_DECAY_COEFF,
    image_to_label_loss_ratio=IMAGE_TO_LABEL_CLIP_LOSS_SCALE,
    label_to_image_loss_ratio=LABEL_TO_IMAGE_LOSS_SCALE,
    instance_loss_scale=INSTANCE_LOSS_SCALE,
    disable_tqdm=False,
    metric_calculators: Dict[str, Dict[str, torchmetrics.Metric]] = {},
):
    total_loss = 0
    label_loss = 0
    image_loss = 0
    classification_loss = 0
    total_inst_segmentation_loss = 0
    total_samples = 0
    total_classification_loss = 0
    labelling_model.train()
    total = len(clip_train_loader)
    for clip_data_dict in tqdm.tqdm(
        clip_train_loader, total=total, disable=disable_tqdm
    ):
        optim.zero_grad()

        # Now calculate loss from the labelling side
        xyzs = clip_data_dict["xyz"].to(device)
        clip_labels = clip_data_dict["clip_vector"].to(device)
        clip_image_labels = clip_data_dict["clip_image_vector"].to(device)
        image_weights = torch.exp(-exp_decay_coeff * clip_data_dict["distance"]).to(
            device
        )
        label_weights = clip_data_dict["semantic_weight"].to(device)
        (
            predicted_label_latents,
            predicted_image_latents,
            segmentation_logits,
        ) = labelling_model(xyzs)

        # Now create the label mask for the contrastive losses.
        # The idea is that we want to push and pull the representations to the right
        # CLIP representations, however, we don't want to include the same label points
        # in representations to push away from.
        image_label_index = clip_data_dict["img_idx"].to(device).reshape(-1, 1)
        language_label_index = clip_data_dict["label"].to(device).reshape(-1, 1)
        instances = clip_data_dict["instance"].to(device).reshape(-1)
        batch_size = len(image_label_index)
        image_label_mask = (
            image_label_index != image_label_index.t()
        ).float() + torch.eye(batch_size, device=device)
        language_label_mask = (
            language_label_index != language_label_index.t()
        ).float() + torch.eye(batch_size, device=device)

        # For logging purposes, keep track of negative samples per point.
        image_label_mask.requires_grad = False
        language_label_mask.requires_grad = False
        # Use the predicted labels, the ground truth labels, and the masks to
        # compute the contrastive loss.
        contrastive_loss_labels = labelling_model.compute_loss(
            predicted_label_latents,
            clip_labels,
            label_mask=language_label_mask,
            weights=label_weights,
        )
        contrastive_loss_images = labelling_model.compute_loss(
            predicted_image_latents,
            clip_image_labels,
            label_mask=image_label_mask,
            weights=image_weights,
        )
        del (
            image_label_mask,
            image_label_index,
            language_label_mask,
        )
        instance_mask = instances != -1
        if not torch.all(instances == -1):
            inst_segmentation_loss = F.cross_entropy(
                segmentation_logits[instance_mask], instances[instance_mask]
            )
            if metric_calculators.get("instance"):
                with torch.no_grad():
                    for _, calculators in metric_calculators["instance"].items():
                        # Update the calculators.
                        _ = calculators(
                            segmentation_logits[instance_mask], instances[instance_mask]
                        )
        else:
            inst_segmentation_loss = torch.zeros_like(contrastive_loss_images)

        # Now figure out semantic segmentation.
        with torch.no_grad():
            class_probs = classifier.calculate_classifications(
                model_text_features=predicted_label_latents,
                model_image_features=predicted_image_latents,
            )
            # Now figure out semantic accuracy and loss.
            semseg_mask = torch.logical_and(
                language_label_index != -1,
                language_label_index < classifier.total_label_classes,
            ).squeeze(-1)
            if not torch.any(semseg_mask):
                classification_loss = torch.zeros_like(contrastive_loss_images)
            else:
                # Figure out the right classes.
                masked_class_prob = class_probs[semseg_mask]
                masked_labels = language_label_index[semseg_mask].squeeze(-1).long()
                classification_loss = F.cross_entropy(
                    torch.log(masked_class_prob),
                    masked_labels,
                )
                if metric_calculators.get("semantic"):
                    for _, calculators in metric_calculators["semantic"].items():
                        # Update the calculators.
                        _ = calculators(masked_class_prob, masked_labels)

        contrastive_loss = (
            image_to_label_loss_ratio * contrastive_loss_images
            + label_to_image_loss_ratio * contrastive_loss_labels
            + instance_loss_scale * inst_segmentation_loss
        )
        final_loss = contrastive_loss
        final_loss.backward()
        optim.step()
        # Clip the temperature term for stability
        labelling_model.temperature.data = torch.clamp(
            labelling_model.temperature.data, max=np.log(100.0)
        )
        label_loss += contrastive_loss_labels.detach().cpu().item()
        image_loss += contrastive_loss_images.detach().cpu().item()
        total_inst_segmentation_loss += inst_segmentation_loss.detach().cpu().item()
        total_classification_loss += classification_loss.detach().cpu().item()
        total_loss += final_loss.detach().cpu().item()
        total_samples += 1

    to_log = {
        "train_avg/contrastive_loss_labels": label_loss / total_samples,
        "train_avg/contrastive_loss_images": image_loss / total_samples,
        "train_avg/instance_loss": total_inst_segmentation_loss / total_samples,
        "train_avg/semseg_loss": total_classification_loss / total_samples,
        "train_avg/loss_sum": total_loss / total_samples,
        "train_avg/labelling_temp": torch.exp(labelling_model.temperature.data.detach())
        .cpu()
        .item(),
    }
    for metric_dict in metric_calculators.values():
        for metric_name, metric in metric_dict.items():
            try:
                to_log[f"train_avg/{metric_name}"] = (
                    metric.compute().detach().cpu().item()
                )
            except RuntimeError as e:
                to_log[f"train_avg/{metric_name}"] = 0.0
            metric.reset()

    wandb.log(to_log)
    logging.info(to_log)


def test(
    clip_test_loader,
    labelling_model,
    optim,
    epoch: int,
    classifier: ClassificationExtractor,
    device=DEVICE,
    exp_decay_coeff=EXP_DECAY_COEFF,
    image_to_label_loss_ratio=IMAGE_TO_LABEL_CLIP_LOSS_SCALE,
    label_to_image_loss_ratio=LABEL_TO_IMAGE_LOSS_SCALE,
    instance_loss_scale=INSTANCE_LOSS_SCALE,
    save_directory=SAVE_DIRECTORY,
    saving_dataparallel=False,
    metric_calculators: Dict[str, Dict[str, torchmetrics.Metric]] = {},
):
    labelling_model.eval()
    with torch.no_grad():
        total_loss = 0
        total_samples = 0
        label_loss = 0
        image_loss = 0
        total_acc = 0
        classification_loss = 0
        total_classification_loss = 0
        total_classification_accuracy = 0
        total_inst_segmentation_loss = 0
        for clip_data_dict in clip_test_loader:
            xyzs = clip_data_dict["xyz"].to(device)
            clip_labels = clip_data_dict["clip_vector"].to(device)
            clip_image_labels = clip_data_dict["clip_image_vector"].to(device)
            weights = torch.exp(-exp_decay_coeff * clip_data_dict["distance"]).to(
                device
            )
            label_weights = clip_data_dict["semantic_weight"].to(device)
            image_label_index = clip_data_dict["img_idx"].to(device).reshape(-1, 1)
            language_label_index = clip_data_dict["label"].to(device).reshape(-1, 1)
            instances = clip_data_dict["instance"].to(device).reshape(-1)
            (
                predicted_label_latents,
                predicted_image_latents,
                segmentation_logits,
            ) = labelling_model(xyzs)

            batch_size = len(image_label_index)
            image_label_mask = (
                image_label_index != image_label_index.t()
            ).float() + torch.eye(batch_size, device=device)
            language_label_mask = (
                language_label_index != language_label_index.t()
            ).float() + torch.eye(batch_size, device=device)

            image_label_mask.requires_grad = False
            language_label_mask.requires_grad = False
            # Use the predicted labels, the ground truth labels, and the masks to
            # compute the contrastive loss.
            contrastive_loss_labels = labelling_model.compute_loss(
                predicted_label_latents,
                clip_labels,
                label_mask=language_label_mask,
                weights=label_weights,
            )
            contrastive_loss_images = labelling_model.compute_loss(
                predicted_image_latents,
                clip_image_labels,
                label_mask=image_label_mask,
                weights=weights,
            )
            instance_mask = instances != -1
            if not torch.all(instances == -1):
                inst_segmentation_loss = F.cross_entropy(
                    segmentation_logits[instance_mask], instances[instance_mask]
                )
                if metric_calculators.get("instance"):
                    for _, calculators in metric_calculators["instance"].items():
                        # Update the calculators.
                        _ = calculators(
                            segmentation_logits[instance_mask], instances[instance_mask]
                        )
            else:
                inst_segmentation_loss = torch.zeros_like(contrastive_loss_images)

            del (
                image_label_mask,
                image_label_index,
                language_label_mask,
            )

            class_probs = classifier.calculate_classifications(
                model_text_features=predicted_label_latents,
                model_image_features=predicted_image_latents,
            )
            # Now figure out semantic accuracy and loss.
            semseg_mask = torch.logical_and(
                language_label_index != -1,
                language_label_index < classifier.total_label_classes,
            ).squeeze(-1)
            if not torch.any(semseg_mask):
                classification_loss = torch.zeros_like(contrastive_loss_images)
            else:
                # Figure out the right classes.
                masked_class_prob = class_probs[semseg_mask]
                masked_labels = language_label_index[semseg_mask].squeeze(-1).long()
                if metric_calculators.get("semantic"):
                    for _, calculators in metric_calculators["semantic"].items():
                        # Update the calculators.
                        _ = calculators(masked_class_prob, masked_labels)
                classification_loss = F.cross_entropy(
                    torch.log(masked_class_prob),
                    masked_labels,
                )

            contrastive_loss = (
                image_to_label_loss_ratio * contrastive_loss_images
                + label_to_image_loss_ratio * contrastive_loss_labels
                + instance_loss_scale * inst_segmentation_loss
            )
            final_loss = contrastive_loss

            label_loss += contrastive_loss_labels.cpu().item()
            image_loss += contrastive_loss_images.cpu().item()
            total_loss += final_loss.cpu().item()
            total_inst_segmentation_loss += inst_segmentation_loss.cpu().item()
            total_classification_loss += classification_loss.detach().cpu().item()
            total_samples += 1

    to_log = {
        "test_avg/contrastive_loss_label": label_loss / total_samples,
        "test_avg/contrastive_loss_image": image_loss / total_samples,
        "test_avg/instance_loss": total_inst_segmentation_loss / total_samples,
        "test_avg/semseg_loss": total_classification_loss / total_samples,
        "test_avg/loss_sum": total_loss / total_samples,
        "epoch": epoch,
    }
    for metric_dict in metric_calculators.values():
        for metric_name, metric in metric_dict.items():
            try:
                to_log[f"test_avg/{metric_name}"] = (
                    metric.compute().detach().cpu().item()
                )
            except RuntimeError as e:
                to_log[f"test_avg/{metric_name}"] = 0.0
            metric.reset()

    wandb.log(to_log)
    logging.info(
        f"Epoch {epoch}: Instance acc: {total_acc / total_samples:0.3f}, segmentation acc: {total_classification_accuracy / total_samples:0.3f}"
    )
    logging.info(str(to_log))
    if saving_dataparallel:
        to_save = labelling_model.module
    else:
        to_save = labelling_model
    torch.save(
        to_save,
        f"outputs/implicit_models/{save_directory}/implicit_scene_label_model_{epoch}.pt",
    )
    # Save the optimizer as well.
    torch.save(
        optim.state_dict(),
        f"outputs/implicit_models/{save_directory}/implicit_scene_label_model_optimizer_{epoch}.pt",
    )
    torch.save(
        to_save,
        f"outputs/implicit_models/{save_directory}/implicit_scene_label_model_latest.pt",
    )
    return -to_log.get("test_avg/semantic_accuracy_micro", 0) - to_log.get(
        "test_avg/instance_accuracy_micro", 0
    )


def get_habitat_dataset(
    base_scene=SCENE,
    habitat_scenes=SCENE_FILEPATH,
    base_data_path="",
    grid_size=GRID_SIZE,
    image_size=IMAGE_SIZE,
    use_cache=CACHE,
    point_subsample_prob=SUBSAMPLE_PROB,
    gt_segmentation_baseline=False,
    num_inst_segmented_images=NUM_INSTANCE_SEGMENTED_IMAGES,
    num_sem_segmented_images=NUM_SEM_SEGMENTED_IMAGES,
    num_web_segmented_images=NUM_WEB_SEGMENTED_IMAGES,
    gt_semantic_weight: float = GT_SEMANTIC_WEIGHT,
    use_lseg: bool = True,
    use_extra_classes: bool = True,
    use_gt_classes: bool = True,
    eval_only_on_seen_inst: bool = False,
    exclude_diffuse_classes: bool = False,
    class_remapping: Optional[Dict[str, str]] = None,
    exclude_gt_images: bool = True,
):
    gt_segmentation_baseline = gt_segmentation_baseline or (
        num_web_segmented_images == 0
    )
    if num_inst_segmented_images == 0 and num_sem_segmented_images == 0:
        web_only = True
        num_inst_segmented_images = 1
        # num_sem_segmented_images = 1
        eval_only_on_seen_inst = False
    else:
        web_only = False
    dataset = HabitatViewDataset(
        habitat_scenes=habitat_scenes,
        pose_extractor_grid_size=grid_size,
        image_size=(image_size, image_size),
        height_levels=0,
        base_data_path=base_data_path
    )
    id_to_name = dataset._id_to_name
    train_split_size = len(dataset) // 2
    view_train_dataset, view_test_dataset = random_split(
        dataset,
        lengths=[train_split_size, len(dataset) - train_split_size],
    )
    # Now we will have to create more dataloaders for the CLIP dataset.
    if use_cache:
        cache_fp = f".cache/habitat_gt_dataset_{base_scene}_{grid_size}_{image_size}_{num_inst_segmented_images}_{num_sem_segmented_images}.pt"
        if os.path.exists(cache_fp):
            clip_train_dataset = torch.load(cache_fp)
        else:
            location_train_dataset = HabitatLocationDataset(
                habitat_view_ds=view_train_dataset,
                subsample_prob=point_subsample_prob,
                return_nonsegmented_images=gt_segmentation_baseline,
                num_inst_segmented_images=num_inst_segmented_images,
                num_sem_segmented_images=num_sem_segmented_images,
                semantically_segment_instance_labeled=False,
                # In the GT segmentation baseline, we get all image segmentation data.
            )
            # Convert to clip datasets
            clip_train_dataset = ClipLabelledLocation(
                view_train_dataset,
                location_train_dataset,
                id_to_name=id_to_name,
                semantic_weight=gt_semantic_weight,
            )
            torch.save(clip_train_dataset, cache_fp)
    clip_train_dataset._semantic_weight = torch.tensor(gt_semantic_weight)

    if use_cache:
        only_seen_str = "_on_seen" if eval_only_on_seen_inst else ""
        class_remapping_str = "_remapped_classes" if class_remapping else ""
        exclude_diffuse_class_str = "_no_diffuse" if exclude_diffuse_classes else ""
        cache_fp = f".cache/habitat_gt_dataset_test{only_seen_str}{exclude_diffuse_class_str}{class_remapping_str}_{base_scene}_{grid_size}_{image_size}.pt"
        if os.path.exists(cache_fp):
            clip_test_dataset = torch.load(cache_fp)
        else:
            location_test_dataset = HabitatLocationDataset(
                habitat_view_ds=view_test_dataset,
                subsample_prob=point_subsample_prob,
                selective_instance_segmentation=False,
                selective_semantic_segmentation=False,
                use_only_valid_instance_ids=eval_only_on_seen_inst,
                valid_instance_ids=clip_train_dataset.loc_dataset.valid_instance_ids,
                exclude_diffuse_classes=exclude_diffuse_classes,
                class_remapping=class_remapping,
                # Return segmentation for all images in test.
            )
            clip_test_dataset = ClipLabelledLocation(
                view_test_dataset,
                location_test_dataset,
                id_to_name=id_to_name
                if not class_remapping
                else location_test_dataset._id_to_name,
                semantic_weight=gt_semantic_weight,
            )
            torch.save(clip_test_dataset, cache_fp)

    lseg_str = "lseg" if use_lseg else "no_lseg"
    extra_classes_str = "" if use_extra_classes else "_no_sn_200"
    use_gt_classes_str = "" if use_gt_classes else "_no_gt_classes"
    no_gt_images_str = (
        ""
        if not exclude_gt_images
        else f"_no_gt_imgs_{num_inst_segmented_images}_{num_sem_segmented_images}"
    )
    condition_str = (
        f"{lseg_str}{extra_classes_str}{use_gt_classes_str}{no_gt_images_str}"
    )
    if use_cache and not gt_segmentation_baseline:
        cache_fp = f".cache/{condition_str}_labeled_dataset_{base_scene}_{grid_size}_{image_size}_{num_web_segmented_images}.pt"
        if os.path.exists(cache_fp):
            location_train_dataset_1 = torch.load(cache_fp)
        else:
            location_train_dataset_1 = DeticDenseLabelledDataset(
                view_train_dataset,
                num_images_to_label=num_web_segmented_images,
                use_lseg=use_lseg,
                use_extra_classes=use_extra_classes,
                use_gt_classes=use_gt_classes,
                exclude_gt_images=exclude_gt_images,
                gt_inst_images=clip_train_dataset.loc_dataset._inst_segmented_images,
                gt_sem_images=clip_train_dataset.loc_dataset._sem_segmented_images,
            )
            torch.save(location_train_dataset_1, cache_fp)

    clip_test_dataset._semantic_weight = torch.tensor(gt_semantic_weight)

    if not gt_segmentation_baseline and not web_only:
        clip_train_dataset_concat = ConcatDataset(
            [clip_train_dataset, location_train_dataset_1]
        )
    else:
        if gt_segmentation_baseline:
            clip_train_dataset_concat = clip_train_dataset
        elif web_only:
            clip_train_dataset_concat = location_train_dataset_1

    # Close dataset sim
    dataset.image_extractor.sim.close()
    del dataset.image_extractor.sim

    return (
        list(dataset._id_to_name.values()),
        list(clip_test_dataset.loc_dataset._id_to_name.values()),
        clip_train_dataset,
        clip_train_dataset_concat,
        clip_test_dataset,
    )


def get_real_dataset():
    dataset = RealWorldSemanticDataset(REAL_SCENE_DIRECTORY)
    surface_dataset = RealWorldSurfaceDataset(dataset, sampling_rate=0.2)
    clip_dataset = RealWorldClipDataset(dataset, sampling_rate=0.2)

    # Now split both of the datasets in half
    surface_train_split_size = int(len(surface_dataset) * 0.98)
    surface_train_set, surface_test_set = random_split(
        surface_dataset,
        lengths=[
            surface_train_split_size,
            len(surface_dataset) - surface_train_split_size,
        ],
    )
    clip_train_split_size = int(len(clip_dataset) * 0.98)
    clip_train_dataset, clip_test_dataset = random_split(
        clip_dataset,
        lengths=[clip_train_split_size, len(clip_dataset) - clip_train_split_size],
    )
    return (
        dataset,
        surface_train_set,
        surface_test_set,
        clip_train_dataset,
        clip_test_dataset,
    )


@hydra.main(version_base="1.2", config_path="configs", config_name="train.yaml")
def main(cfg):
    if cfg.base_data_path is not None:
        for i, p in enumerate(cfg.scene.filepath):
            cfg.scene.filepath[i] = p.replace("/checkpoint/notmahi/data/", cfg.base_data_path)
        print("replaced data path according to commandline:")
        print(cfg.scene.filepath)
    seed_everything(cfg.seed)
    # Run basic sanity test on the dataloader.
    # Now, figure out if we are relabelling during test, and if so, add that to the
    # test loader
    if cfg.semantic_class_remapping_in_test and hasattr(
        cfg.scene, "semantic_class_remapping"
    ):
        class_remapping = cfg.scene.semantic_class_remapping
    else:
        class_remapping = None
    (
        train_label_set,
        test_label_set,
        parent_train_dataset,
        clip_train_dataset,
        clip_test_dataset,
    ) = get_habitat_dataset(
        base_scene=cfg.scene.base,
        habitat_scenes=cfg.scene.filepath,
        base_data_path=cfg.base_data_path,
        grid_size=cfg.scene.grid_size,
        image_size=cfg.scene.image_size,
        use_cache=cfg.use_cache,
        point_subsample_prob=cfg.scene.subsample_prob,
        gt_segmentation_baseline=cfg.gt_segmentation_baseline,
        num_inst_segmented_images=cfg.num_inst_segmented_images,
        num_sem_segmented_images=cfg.num_sem_segmented_images,
        num_web_segmented_images=cfg.num_web_segmented_images,
        gt_semantic_weight=cfg.gt_semantic_weight,
        use_lseg=cfg.use_lseg,
        use_extra_classes=cfg.use_extra_classes,
        use_gt_classes=cfg.use_gt_classes_in_detic,
        exclude_diffuse_classes=cfg.exclude_diffuse_classes_in_test,
        eval_only_on_seen_inst=cfg.eval_only_on_seen_inst,
        class_remapping=class_remapping,
    )

    # Setup our model with min and max coordinates.
    max_coords, _ = torch.stack(
        [
            x[0]
            for x in (
                parent_train_dataset.coordinate_range,
                clip_test_dataset.coordinate_range,
            )
        ]
    ).max(0)
    min_coords, _ = torch.stack(
        [
            x[1]
            for x in (
                parent_train_dataset.coordinate_range,
                clip_test_dataset.coordinate_range,
            )
        ]
    ).min(0)
    logging.info(f"Environment bounds: max {max_coords} min {min_coords}")

    # Assume the classes go from 0 up to class labels.
    num_instances = max(parent_train_dataset.loc_dataset.instance_id_to_name.keys())
    num_instances = max(num_instances, len(parent_train_dataset.loc_dataset.instance_id_to_name))
    if cfg.cache_only_run:
        # Caching is done, so we can exit now.
        logging.info("Cache only run, exiting.")
        exit(0)
    train_classifier = ClassificationExtractor(
        clip_model_name=cfg.web_models.clip,
        sentence_model_name=cfg.web_models.sentence,
        class_names=train_label_set,
        device=cfg.device,
    )
    test_classifier = ClassificationExtractor(
        clip_model_name=cfg.web_models.clip,
        sentence_model_name=cfg.web_models.sentence,
        class_names=test_label_set,
        device=cfg.device,
    )

    # Set up our metrics on this dataset.
    train_metric_calculators = {}
    test_metric_calculators = {}

    train_class_count = {
        "semantic": train_classifier.total_label_classes,
        "instance": num_instances + 1,
    }
    test_class_count = {
        "semantic": test_classifier.total_label_classes,
        "instance": num_instances + 1,
    }
    average_style = ["micro", "macro", "weighted"]
    for classes, counts in train_class_count.items():
        train_metric_calculators[classes] = {}
        for metric_name, metric_cls in METRICS.items():
            for avg in average_style:
                if "accuracy" in metric_name:
                    new_metric = metric_cls(
                        num_classes=counts, average=avg, multiclass=True
                    ).to(cfg.device)
                train_metric_calculators[classes][
                    f"{classes}_{metric_name}_{avg}"
                ] = new_metric

    for classes, counts in test_class_count.items():
        test_metric_calculators[classes] = {}
        for metric_name, metric_cls in METRICS.items():
            for avg in average_style:
                new_metric = metric_cls(num_classes=counts, average=avg).to(cfg.device)
                test_metric_calculators[classes][
                    f"{classes}_{metric_name}_{avg}"
                ] = new_metric

                if metric_name == "accuracy":
                    # Add topk
                    new_metric = metric_cls(
                        num_classes=counts, average=avg, top_k=TOP_K
                    ).to(cfg.device)
                    test_metric_calculators[classes][
                        f"{classes}_{metric_name}_{avg}_top{TOP_K}"
                    ] = new_metric

    if cfg.model_type == "MLP":
        labelling_model = ImplicitCLIPModel()
    elif cfg.model_type == "hash":
        labelling_model = GridCLIPModel(
            image_rep_size=parent_train_dataset.image_representation_size,
            text_rep_size=parent_train_dataset.text_representation_size,
            mlp_depth=cfg.mlp_depth,
            mlp_width=cfg.mlp_width,
            log2_hashmap_size=cfg.log2_hashmap_size,
            segmentation_classes=num_instances + 1,  # Quick patch
            num_levels=cfg.num_grid_levels,
            level_dim=cfg.level_dim,
            per_level_scale=cfg.per_level_scale,
            max_coords=max_coords,
            min_coords=min_coords,
        )
    label_voxel_count = int(cfg.label_voxel_count)

    if torch.cuda.device_count() > 1 and cfg.dataparallel:
        batch_multiplier = torch.cuda.device_count()
    else:
        batch_multiplier = 1
    label_sampler = RandomSampler(
        data_source=clip_train_dataset,
        num_samples=label_voxel_count,
        replacement=True,
    )
    clip_train_loader = DataLoader(
        clip_train_dataset,
        batch_size=batch_multiplier * cfg.point_batch_size,
        sampler=label_sampler,
        pin_memory=True,
        num_workers=cfg.num_workers,
    )
    clip_test_loader = DataLoader(
        clip_test_dataset,
        batch_size=cfg.point_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=cfg.num_workers,
    )

    logging.info(f"Train human labelled point sizes: {len(parent_train_dataset)}")
    logging.info(f"Total train dataset sizes: {len(clip_train_dataset)}")
    logging.info(f"Test dataset sizes: {len(clip_test_dataset)}")
    logging.info(
        f"Epochs for one pass over dataset: {len(clip_train_dataset) // label_voxel_count}"
    )

    labelling_model = labelling_model.to(cfg.device)

    save_directory = cfg.save_directory.format(cfg.scene.base)
    loaded = False
    if os.path.exists("outputs/implicit_models/{}/".format(save_directory)):
        # First find out which epoch is the latest one.
        all_files = glob.glob(
            "outputs/implicit_models/{}/implicit_scene_label_model_*.pt".format(
                save_directory
            )
        )
        if len(all_files) > 0:
            # Find out which is the latest checkpoint.
            epoch = 0
            model_path = (
                "outputs/implicit_models/{}/implicit_scene_label_model_{}.pt".format(
                    save_directory, epoch
                )
            )
            while os.path.exists(model_path):
                epoch += EVAL_EVERY
                model_path = "outputs/implicit_models/{}/implicit_scene_label_model_{}.pt".format(
                    save_directory, epoch
                )
            epoch -= EVAL_EVERY
            model_path = (
                "outputs/implicit_models/{}/implicit_scene_label_model_{}.pt".format(
                    save_directory, epoch
                )
            )
            optim_path = "outputs/implicit_models/{}/implicit_scene_label_model_optimizer_{}.pt".format(
                save_directory, epoch
            )
            logging.info(f"Resuming job from: {model_path}")
            # This has already started training, let's load the model
            labelling_model = torch.load(
                model_path,
                map_location=cfg.device,
            )
            optim = torch.optim.Adam(
                labelling_model.parameters(),
                lr=cfg.lr,
                betas=tuple(cfg.betas),
                weight_decay=cfg.weight_decay,
            )
            if os.path.exists(optim_path):
                optim.load_state_dict(torch.load(optim_path))
            resume = "allow"
            loaded = True
            epoch += 1
    if not loaded:
        logging.info("Could not find old runs, starting fresh...")
        os.makedirs(
            "outputs/implicit_models/{}/".format(save_directory),
            exist_ok=True,
        )
        epoch = 0
        resume = False
        optim = torch.optim.Adam(
            labelling_model.parameters(),
            lr=cfg.lr,
            betas=tuple(cfg.betas),
            weight_decay=cfg.weight_decay,
        )

    dataparallel = False
    if torch.cuda.device_count() > 1 and cfg.dataparallel:
        labelling_model = ImplicitDataparallel(labelling_model)
        dataparallel = True

    if cfg.deterministic_id:
        run_id = f"grid_{cfg.scene.base}_{cfg.scene.image_size}i{cfg.num_inst_segmented_images}s{cfg.num_sem_segmented_images}w{cfg.num_web_segmented_images}"
    else:
        run_id = wandb.util.generate_id()

    wandb.init(
        project=cfg.project,
        id=run_id,
        tags=[
            f"model/{cfg.model_type}",
            f"scene/{cfg.scene.base}",
        ],
        config=OmegaConf.to_container(cfg, resolve=True),
        resume=resume,
    )
    # Set the extra parameters.
    wandb.config.human_labelled_points = len(parent_train_dataset)
    wandb.config.web_labelled_points = len(clip_train_dataset) - len(
        parent_train_dataset
    )
    wandb.config.num_seen_instances = len(
        parent_train_dataset.loc_dataset.valid_instance_ids
    )
    seen_instances = wandb.Artifact("seen_instances", "dataset")
    table = wandb.Table(
        columns=["instances"],
        data=[[x] for x in parent_train_dataset.loc_dataset.valid_instance_ids],
    )
    seen_instances.add(table, "my_table")
    wandb.log_artifact(seen_instances)

    # Disable tqdm if we are running inside slurm
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is not None:
        disable_tqdm = True
    else:
        disable_tqdm = False
    test_accuracy = 0
    while epoch <= cfg.epochs:
        train(
            clip_train_loader,
            labelling_model,
            optim,
            train_classifier,
            cfg.device,
            exp_decay_coeff=cfg.exp_decay_coeff,
            image_to_label_loss_ratio=cfg.image_to_label_loss_ratio,
            label_to_image_loss_ratio=cfg.label_to_image_loss_ratio,
            instance_loss_scale=cfg.instance_loss_scale,
            disable_tqdm=disable_tqdm,
            metric_calculators=train_metric_calculators,
        )
        if epoch % EVAL_EVERY == 0:
            test_accuracy = test(
                clip_test_loader,
                labelling_model,
                optim,
                epoch,
                test_classifier,
                cfg.device,
                exp_decay_coeff=cfg.exp_decay_coeff,
                image_to_label_loss_ratio=cfg.image_to_label_loss_ratio,
                label_to_image_loss_ratio=cfg.label_to_image_loss_ratio,
                instance_loss_scale=cfg.instance_loss_scale,
                save_directory=save_directory,
                saving_dataparallel=dataparallel,
                metric_calculators=test_metric_calculators,
            )
        epoch += 1
    return test_accuracy


if __name__ == "__main__":
    main()
