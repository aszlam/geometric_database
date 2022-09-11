import glob
import logging
import os
import random
from typing import Dict

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import tqdm
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, RandomSampler

import wandb
from dataloaders.clip_labeled_habitat import ClassificationExtractor

from dataloaders.detic_labeled_habitat import DeticDenseLabelledDataset
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
    return total_loss


def save(
    labelling_model,
    optim,
    epoch: int,
    save_directory=SAVE_DIRECTORY,
    saving_dataparallel=False,
):

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
    return 0


def get_real_dataset(dataset_path):
    location_train_dataset_1 = torch.load(dataset_path)
    return location_train_dataset_1


@hydra.main(version_base="1.2", config_path="configs", config_name="train.yaml")
def main(cfg):
    seed_everything(cfg.seed)
    # Run basic sanity test on the dataloader.
    # Now, figure out if we are relabelling during test, and if so, add that to the
    # test loader

    real_dataset: DeticDenseLabelledDataset = get_real_dataset(cfg.saved_dataset_path)

    tfm_1 = np.array(
        [
            0.987125873566,
            0.000000000000,
            -0.159945458174,
            -0.389560431242,
            0.000000000000,
            1.000000000000,
            0.000000000000,
            0.000000000000,
            0.159945458174,
            0.000000000000,
            0.987125873566,
            0.206698656082,
            0.000000000000,
            0.000000000000,
            0.000000000000,
            1.000000000000,
        ]
    ).reshape((4, 4))
    tfm_1 = torch.as_tensor(tfm_1).float()
    tfm_3 = torch.eye(4)
    tfm_3[0, 0] = -1
    tfm_3[2, 2] = -1
    new_coords = (
        tfm_3
        @ tfm_1
        @ torch.cat(
            [
                real_dataset._label_xyz,
                torch.ones_like(real_dataset._label_xyz).mean(dim=-1, keepdim=True),
            ],
            dim=-1,
        ).T
    ).T
    new_coords = new_coords[:, :3]

    max_coords, _ = new_coords.max(dim=0)
    min_coords, _ = new_coords.min(dim=0)
    logging.info(
        f"Environment bounds: max {max_coords} min {min_coords} span {max_coords - min_coords}"
    )

    real_dataset._label_xyz = new_coords
    # Setup our model with min and max coordinates.
    max_coords, _ = real_dataset._label_xyz.max(dim=0)
    min_coords, _ = real_dataset._label_xyz.min(dim=0)
    logging.info(f"Environment bounds: max {max_coords} min {min_coords}")

    train_classifier = ClassificationExtractor(
        clip_model_name=cfg.web_models.clip,
        sentence_model_name=cfg.web_models.sentence,
        class_names=real_dataset._all_classes,
        device=cfg.device,
    )

    # Set up our metrics on this dataset.
    train_metric_calculators = {}

    # Assume the classes go from 0 up to class labels.
    train_class_count = {
        "semantic": train_classifier.total_label_classes,
        # "instance": len(real_dataset._all_classes) + 1,
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

    if cfg.model_type == "MLP":
        labelling_model = ImplicitCLIPModel()
    elif cfg.model_type == "hash":
        labelling_model = GridCLIPModel(
            image_rep_size=real_dataset[0]["clip_image_vector"].shape[-1],
            text_rep_size=real_dataset[0]["clip_vector"].shape[-1],
            mlp_depth=cfg.mlp_depth,
            mlp_width=cfg.mlp_width,
            log2_hashmap_size=cfg.log2_hashmap_size,
            segmentation_classes=len(real_dataset._all_classes) + 1,  # Quick patch
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
        data_source=real_dataset,
        num_samples=label_voxel_count,
        replacement=True,
    )
    clip_train_loader = DataLoader(
        real_dataset,
        batch_size=batch_multiplier * cfg.point_batch_size,
        sampler=label_sampler,
        pin_memory=True,
        num_workers=cfg.num_workers,
    )

    logging.info(f"Total train dataset sizes: {len(real_dataset)}")
    logging.info(
        f"Epochs for one pass over dataset: {len(real_dataset) // label_voxel_count}"
    )

    labelling_model = labelling_model.to(cfg.device)

    run_id = wandb.util.generate_id()
    save_directory = cfg.save_directory_real + f"/{run_id}"
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
    wandb.config.human_labelled_points = 0
    wandb.config.web_labelled_points = len(real_dataset)
    wandb.config.num_seen_instances = 0

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
            disable_tqdm=disable_tqdm,
            metric_calculators=train_metric_calculators,
        )
        epoch += 1
        if epoch % EVAL_EVERY == 0:
            save(
                labelling_model,
                optim,
                epoch,
                save_directory=save_directory,
                saving_dataparallel=dataparallel,
            )
    return test_accuracy


if __name__ == "__main__":
    main()
