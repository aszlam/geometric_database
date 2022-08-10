import glob
import logging
import os
import random

import hydra
import numpy as np
import torch
import torch.nn.functional as F
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

MODEL_TYPE = "hash"  # MLP or hash
CACHE = True


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
):
    total_loss = 0
    label_loss = 0
    image_loss = 0
    classification_loss = 0
    classification_accuracy = 0
    total_inst_segmentation_loss = 0
    total_accuracy = 0
    total_samples = 0
    total_classification_loss = 0
    total_classification_accuracy = 0
    labelling_model.train()
    total = len(clip_train_loader)
    for clip_data_dict in tqdm.tqdm(clip_train_loader, total=total):
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
        image_label_ratio = (
            (image_label_mask.sum() / torch.numel(image_label_mask))
            .detach()
            .cpu()
            .item()
        )
        lang_label_ratio = (
            (language_label_mask.sum() / torch.numel(language_label_mask))
            .detach()
            .cpu()
            .item()
        )
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
            accuracy = (
                (
                    segmentation_logits[instance_mask].argmax(dim=-1)
                    == instances[instance_mask]
                )
                .float()
                .mean()
            )
            accuracy = accuracy.detach().cpu().item()
        else:
            inst_segmentation_loss = torch.zeros_like(contrastive_loss_images)
            accuracy = 1.0

        # Now figure out semantic segmentation.
        with torch.no_grad():
            class_probs = classifier.calculate_classifications(
                model_text_features=predicted_label_latents,
                model_image_features=predicted_image_latents,
            )
            # Now figure out semantic accuracy and loss.
            semseg_mask = (language_label_index != 0).squeeze(-1)
            if not torch.any(semseg_mask):
                classification_loss = 0
                classification_accuracy = 1.0
            else:
                # Figure out the right classes.
                masked_class_prob = class_probs[semseg_mask]
                masked_labels = language_label_index[semseg_mask].squeeze(-1)
                classification_accuracy = (
                    (masked_class_prob.argmax(dim=-1) == masked_labels).float().mean()
                )
                classification_loss = F.cross_entropy(
                    torch.log(masked_class_prob),
                    masked_labels,
                )

        total_accuracy += accuracy
        total_classification_accuracy += classification_accuracy
        contrastive_loss = (
            image_to_label_loss_ratio * contrastive_loss_images
            + label_to_image_loss_ratio * contrastive_loss_labels
            + instance_loss_scale * inst_segmentation_loss
        )
        final_loss = contrastive_loss
        final_loss.backward()
        optim.step()
        label_loss += contrastive_loss_labels.detach().cpu().item()
        image_loss += contrastive_loss_images.detach().cpu().item()
        total_inst_segmentation_loss += inst_segmentation_loss.detach().cpu().item()
        total_classification_loss += classification_loss.detach().cpu().item()
        total_loss += final_loss.detach().cpu().item()
        total_samples += 1
        wandb.log(
            {
                "train/contrastive_loss_labels": contrastive_loss_labels,
                "train/contrastive_loss_images": contrastive_loss_images,
                "train/instance_loss": inst_segmentation_loss,
                "train/instance_accuracy": accuracy,
                "train/semseg_loss": classification_loss,
                "train/semseg_accuracy": classification_accuracy,
                "train/loss_sum": final_loss,
                "train/image_label_ratio": image_label_ratio,
                "train/lang_label_ratio": lang_label_ratio,
            }
        )

    wandb.log(
        {
            "train_avg/contrastive_loss_labels": label_loss / total_samples,
            "train_avg/contrastive_loss_images": image_loss / total_samples,
            "train_avg/instance_loss": total_inst_segmentation_loss / total_samples,
            "train_avg/instance_accuracy": total_accuracy / total_samples,
            "train_avg/semseg_loss": total_classification_loss / total_samples,
            "train_avg/semseg_accuracy": total_classification_accuracy / total_samples,
            "train_avg/loss_sum": total_loss / total_samples,
            "train_avg/labelling_temp": torch.exp(
                labelling_model.temperature.data.detach()
            ),
        }
    )


def test(
    clip_test_loader,
    labelling_model,
    epoch: int,
    classifier: ClassificationExtractor,
    device=DEVICE,
    exp_decay_coeff=EXP_DECAY_COEFF,
    image_to_label_loss_ratio=IMAGE_TO_LABEL_CLIP_LOSS_SCALE,
    label_to_image_loss_ratio=LABEL_TO_IMAGE_LOSS_SCALE,
    instance_loss_scale=INSTANCE_LOSS_SCALE,
    save_directory=SAVE_DIRECTORY,
):
    labelling_model.eval()
    with torch.no_grad():
        total_loss = 0
        total_samples = 0
        label_loss = 0
        image_loss = 0
        total_acc = 0
        classification_loss = 0
        classification_accuracy = 0
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
            (
                predicted_label_latents,
                predicted_image_latents,
                segmentation_logits,
            ) = labelling_model(xyzs)
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
            inst_segmentation_loss = F.cross_entropy(
                segmentation_logits[instance_mask], instances[instance_mask]
            )
            accuracy = (
                (
                    segmentation_logits[instance_mask].argmax(dim=-1)
                    == instances[instance_mask]
                )
                .float()
                .mean()
            )
            accuracy = accuracy.detach().cpu().item()
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
            semseg_mask = (language_label_index != 0).squeeze(-1)
            if not torch.any(semseg_mask):
                classification_loss = 0
                classification_accuracy = 1.0
            else:
                # Figure out the right classes.
                masked_class_prob = class_probs[semseg_mask]
                masked_labels = language_label_index[semseg_mask].squeeze(-1)
                classification_accuracy = (
                    (masked_class_prob.argmax(dim=-1) == masked_labels).float().mean()
                )
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
            total_acc += accuracy
            total_samples += 1

            wandb.log(
                {
                    "test/contrastive_loss_label": contrastive_loss_labels,
                    "test/contrastive_loss_image": contrastive_loss_images,
                    "test/instance_loss": inst_segmentation_loss,
                    "test/instance_accuracy": accuracy,
                    "test/semseg_loss": classification_loss,
                    "test/semseg_accuracy": classification_accuracy,
                    "test/loss_sum": final_loss,
                }
            )

    wandb.log(
        {
            "test_avg/contrastive_loss_label": label_loss / total_samples,
            "test_avg/contrastive_loss_image": image_loss / total_samples,
            "test_avg/instance_loss": total_inst_segmentation_loss / total_samples,
            "test_avg/instance_accuracy": total_acc / total_samples,
            "test_avg/semseg_loss": total_classification_loss / total_samples,
            "test_avg/semseg_accuracy": total_classification_accuracy / total_samples,
            "test_avg/loss_sum": total_loss / total_samples,
        }
    )
    torch.save(
        labelling_model,
        f"outputs/implicit_models/{save_directory}/implicit_scene_label_model_{epoch}.pt",
    )


def get_habitat_dataset(
    base_scene=SCENE,
    habitat_scenes=SCENE_FILEPATH,
    grid_size=GRID_SIZE,
    image_size=IMAGE_SIZE,
    use_cache=CACHE,
    point_subsample_prob=SUBSAMPLE_PROB,
    gt_segmentation_baseline=False,
    num_inst_segmented_images=NUM_INSTANCE_SEGMENTED_IMAGES,
    num_sem_segmented_images=NUM_SEM_SEGMENTED_IMAGES,
    num_web_segmented_images=NUM_WEB_SEGMENTED_IMAGES,
):
    dataset = HabitatViewDataset(
        habitat_scenes=habitat_scenes,
        pose_extractor_grid_size=grid_size,
        image_size=(image_size, image_size),
        height_levels=0,
    )
    train_split_size = len(dataset) // 2
    view_train_dataset, view_test_dataset = random_split(
        dataset,
        lengths=[train_split_size, len(dataset) - train_split_size],
    )
    if use_cache and not gt_segmentation_baseline:
        cache_fp = f".cache/lseg_labeled_dataset_{base_scene}_{grid_size}_{image_size}_{num_web_segmented_images}.pt"
        if os.path.exists(cache_fp):
            location_train_dataset_1 = torch.load(cache_fp)
        else:
            location_train_dataset_1 = DeticDenseLabelledDataset(
                view_train_dataset, num_images_to_label=num_web_segmented_images
            )
            torch.save(location_train_dataset_1, cache_fp)
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
                num_sem_segmented_images=num_sem_segmented_images
                # In the GT segmentation baseline, we get all image segmentation data.
            )
            # Convert to clip datasets
            clip_train_dataset = ClipLabelledLocation(
                view_train_dataset, location_train_dataset
            )
            torch.save(clip_train_dataset, cache_fp)

    if use_cache:
        cache_fp = (
            f".cache/habitat_gt_dataset_test_{base_scene}_{grid_size}_{image_size}.pt"
        )
        if os.path.exists(cache_fp):
            clip_test_dataset = torch.load(cache_fp)
        else:
            location_test_dataset = HabitatLocationDataset(
                habitat_view_ds=view_test_dataset,
                subsample_prob=point_subsample_prob,
                selective_instance_segmentation=False,
                selective_semantic_segmentation=False,
                # Return segmentation for all images in test.
            )
            clip_test_dataset = ClipLabelledLocation(
                view_test_dataset, location_test_dataset
            )
            torch.save(clip_test_dataset, cache_fp)

    if not gt_segmentation_baseline:
        clip_train_dataset_concat = ConcatDataset(
            [clip_train_dataset, location_train_dataset_1]
        )
    else:
        clip_train_dataset_concat = clip_train_dataset
    return (
        list(dataset._id_to_name.values()),
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
    seed_everything(cfg.seed)
    # Run basic sanity test on the dataloader.
    (
        label_set,
        parent_dataset,
        clip_train_dataset,
        clip_test_dataset,
    ) = get_habitat_dataset(
        base_scene=cfg.scene.base,
        habitat_scenes=cfg.scene.filepath,
        grid_size=cfg.scene.grid_size,
        image_size=cfg.scene.image_size,
        use_cache=cfg.use_cache,
        point_subsample_prob=cfg.scene.subsample_prob,
        gt_segmentation_baseline=cfg.gt_segmentation_baseline,
        num_inst_segmented_images=cfg.num_inst_segmented_images,
        num_sem_segmented_images=cfg.num_sem_segmented_images,
        num_web_segmented_images=cfg.num_web_segmented_images,
    )
    if cfg.cache_only_run:
        # Caching is done, so we can exit now.
        exit(0)
    # Setup our model with min and max coordinates.
    max_coords, _ = torch.stack(
        [
            x[0]
            for x in (
                clip_train_dataset.coordinate_range,
                clip_test_dataset.coordinate_range,
            )
        ]
    ).max(0)
    min_coords, _ = torch.stack(
        [
            x[1]
            for x in (
                clip_train_dataset.coordinate_range,
                clip_test_dataset.coordinate_range,
            )
        ]
    ).min(0)
    logging.info(f"Environment bounds: max {max_coords} min {min_coords}")

    classifier = ClassificationExtractor(
        clip_model_name=cfg.web_models.clip,
        sentence_model_name=cfg.web_models.sentence,
        class_names=label_set,
        device=cfg.device,
    )
    if cfg.model_type == "MLP":
        labelling_model = ImplicitCLIPModel()
    elif cfg.model_type == "hash":
        labelling_model = GridCLIPModel(
            image_rep_size=parent_dataset.image_representation_size,
            text_rep_size=parent_dataset.text_representation_size,
            max_coords=max_coords,
            min_coords=min_coords,
        )
    label_voxel_count = int(cfg.label_voxel_count)
    label_sampler = RandomSampler(
        data_source=clip_train_dataset,
        num_samples=label_voxel_count,
        replacement=True,
    )
    clip_train_loader = DataLoader(
        clip_train_dataset,
        batch_size=cfg.point_batch_size,
        sampler=label_sampler,
        pin_memory=True,
    )
    clip_test_loader = DataLoader(
        clip_test_dataset,
        batch_size=cfg.point_batch_size,
        shuffle=False,
        pin_memory=True,
    )

    logging.info(f"Train loader sizes: clip {len(clip_train_loader)}")
    logging.info(f"Test loader sizes: clip {len(clip_test_loader)}")

    labelling_model = labelling_model.to(cfg.device)
    optim = torch.optim.Adam(
        labelling_model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    wandb.init(
        project="implicit_clip_model",
        tags=[
            f"model/{cfg.model_type}",
        ],
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    save_directory = cfg.save_directory.format(cfg.scene.base)
    os.makedirs(
        "outputs/implicit_models/{}/".format(save_directory),
        exist_ok=True,
    )

    for epoch in range(cfg.epochs):
        train(
            clip_train_loader,
            labelling_model,
            optim,
            classifier,
            cfg.device,
            exp_decay_coeff=cfg.exp_decay_coeff,
            image_to_label_loss_ratio=cfg.image_to_label_loss_ratio,
            label_to_image_loss_ratio=cfg.label_to_image_loss_ratio,
            instance_loss_scale=cfg.instance_loss_scale,
        )
        if epoch % EVAL_EVERY == 0:
            test(
                clip_test_loader,
                labelling_model,
                epoch,
                classifier,
                cfg.device,
                exp_decay_coeff=cfg.exp_decay_coeff,
                image_to_label_loss_ratio=cfg.image_to_label_loss_ratio,
                label_to_image_loss_ratio=cfg.label_to_image_loss_ratio,
                instance_loss_scale=cfg.instance_loss_scale,
                save_directory=save_directory,
            )


if __name__ == "__main__":
    main()
