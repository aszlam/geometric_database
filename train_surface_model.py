import logging
from dataloaders.habitat_loaders import HabitatViewDataset, HabitatLocationDataset
from dataloaders.clip_labeled_habitat import ClipLabelledLocation
from dataloaders.clip_labeled_real_world import (
    RealWorldSemanticDataset,
    RealWorldClipDataset,
    RealWorldSurfaceDataset,
    get_voxel_normalized_sampler_and_occupied_voxels,
)
from dataloaders.detic_labeled_habitat import DeticDenseLabelledDataset
import tqdm
from torch.utils.data import (
    DataLoader,
    random_split,
    WeightedRandomSampler,
    ConcatDataset,
)
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.mlp import MLP
from models.scene_models.positional_encoding import FourierFeatures
from implicit_models.implicit_mlp import ImplicitSurfaceModel, ImplicitCLIPModel
from implicit_models.grid_hash_model import GridSurfaceModel, GridCLIPModel
from itertools import chain, cycle
import glob
import os
import einops


import wandb

# Replace with the path to your scene file
SCENE_FILEPATH = [
    # "/private/home/notmahi/data/replica_dataset/room_0/habitat/mesh_semantic.ply",
    "/private/home/notmahi/data/replica_dataset/apartment_0/habitat/mesh_semantic.ply",
    # "/private/home/notmahi/data/replica_dataset/room_1/habitat/mesh_semantic.ply",
    # "/private/home/notmahi/data/replica_dataset/room_2/habitat/mesh_semantic.ply",
]

REAL_SCENE_DIRECTORY = glob.glob(
    "/private/home/notmahi/data/stretch_fairmont/trajectories/july10_fremont/bed2/"
)
# REAL_SCENE_DIRECTORY = (
#     "/private/home/notmahi/data/stretch_fairmont/trajectories/july10_fremont/couch3/"
# )
BATCH_SIZE = 256

SAVE_DIRECTORY = "habitat_apt_0_sentence_embed_small_image_loss"

# Create model using a simple MLP and a Fourier projection.
# This model should really tell you the probability of something being a surface point or not.


DEVICE = "cuda"
IMAGE_BATCH_SIZE = 32 * 7 * 7
IMAGE_SIZE = 224
POINT_BATCH_SIZE = 256 * 7 * 7
IMAGE_TO_LABEL_CLIP_LOSS_SCALE = 1.0
LABEL_TO_IMAGE_LOSS_SCALE = 1.0
EXP_DECAY_COEFF = 0.5
# EPOCHS = 5000
EPOCHS = 500
SUBSAMPLE_PROB = 0.2
EVAL_EVERY = 5
SURFACE_LOSS_LAMBDA = 10.0
INSTANCE_LOSS_SCALE = 5.0

MODEL_TYPE = "hash"  # MLP or hash


def extract_positive_and_negative_surface_examples(datapoint_dict):
    xyz_positions = datapoint_dict["xyz_position"].to(DEVICE)
    subsampled_xyz_position = xyz_positions[..., ::2, ::2]
    camera_location = datapoint_dict["camera_pos"].to(DEVICE)
    extended_camera_loc = einops.rearrange(camera_location, "... (d 1 1) -> ... d 1 1")
    # Now figure out negative samples between the two points.
    random_batchsize = torch.rand(len(xyz_positions)).to(DEVICE)
    negative_samples = random_batchsize[:, None, None, None] * extended_camera_loc + (
        (1 - random_batchsize[:, None, None, None]) * subsampled_xyz_position
    )

    # Now reshape everything to shape
    positives = einops.rearrange(subsampled_xyz_position, "b d w h -> (b w h) d")
    negatives = einops.rearrange(negative_samples, "b d w h -> (b w h) d")
    return positives, negatives


# def extract_positive_and_negative_surface_examples(datapoint_dict):
#     xyz_positions = datapoint_dict["xyz_position"].to(DEVICE)
#     camera_location = datapoint_dict["camera_pos"].to(DEVICE)
#     random_batchsize = torch.rand(len(xyz_positions)).to(DEVICE)
#     negative_samples = random_batchsize[:, None] * camera_location + (
#         (1 - random_batchsize[:, None]) * xyz_positions
#     )
#     return xyz_positions, negative_samples


def train(
    train_loader,
    clip_train_loader,
    surface_model,
    labelling_model,
    optim,
):
    total_loss = 0
    surface_loss = 0
    label_loss = 0
    image_loss = 0
    total_inst_segmentation_loss = 0
    total_accuracy = 0
    total_samples = 0
    surface_model.train()
    labelling_model.train()
    total = len(clip_train_loader)
    for clip_data_dict in tqdm.tqdm(clip_train_loader, total=total):
        # positives, negatives = extract_positive_and_negative_surface_examples(
        #     datapoint_dict
        # )
        # data = torch.cat([positives, negatives])
        # labels = (
        #     torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))])
        #     .unsqueeze(-1)
        #     .to(DEVICE)
        # )

        # idx = torch.randperm(data.shape[0]).to(DEVICE)

        optim.zero_grad()
        # First, calculate the loss from the surface model.
        # output = surface_model(data[idx])
        # loss = loss_fn(output, labels[idx])

        # Now calculate loss from the labelling side
        xyzs = clip_data_dict["xyz"].to(DEVICE)
        clip_labels = clip_data_dict["clip_vector"].to(DEVICE)
        clip_image_labels = clip_data_dict["clip_image_vector"].to(DEVICE)
        image_weights = torch.exp(-EXP_DECAY_COEFF * clip_data_dict["distance"]).to(
            DEVICE
        )
        label_weights = clip_data_dict["semantic_weight"].to(DEVICE)
        (
            predicted_label_latents,
            predicted_image_latents,
            segmentation_logits,
        ) = labelling_model(xyzs)

        # Now create the label mask for the contrastive losses.
        # The idea is that we want to push and pull the representations to the right
        # CLIP representations, however, we don't want to include the same label points
        # in representations to push away from.
        image_label_index = clip_data_dict["img_idx"].to(DEVICE).reshape(-1, 1)
        language_label_index = clip_data_dict["label"].to(DEVICE).reshape(-1, 1)
        instances = clip_data_dict["instance"].to(DEVICE).reshape(-1)
        batch_size = len(image_label_index)
        image_label_mask = (
            image_label_index != image_label_index.t()
        ).float() + torch.eye(batch_size, device=DEVICE)
        language_label_mask = (
            language_label_index != language_label_index.t()
        ).float() + torch.eye(batch_size, device=DEVICE)

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
            language_label_index,
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

        total_accuracy += accuracy
        contrastive_loss = (
            IMAGE_TO_LABEL_CLIP_LOSS_SCALE * contrastive_loss_images
            + LABEL_TO_IMAGE_LOSS_SCALE * contrastive_loss_labels
            + INSTANCE_LOSS_SCALE * inst_segmentation_loss
        )
        final_loss = contrastive_loss
        final_loss.backward()
        optim.step()
        # surface_loss += loss.detach().cpu().item()
        label_loss += contrastive_loss_labels.detach().cpu().item()
        image_loss += contrastive_loss_images.detach().cpu().item()
        total_inst_segmentation_loss += inst_segmentation_loss.detach().cpu().item()
        total_loss += final_loss.detach().cpu().item()
        total_samples += 1
        wandb.log(
            {
                # "train/surface_loss": loss,
                "train/contrastive_loss_labels": contrastive_loss_labels,
                "train/contrastive_loss_images": contrastive_loss_images,
                "train/instance_loss": inst_segmentation_loss,
                "train/instance_accuracy": accuracy,
                "train/loss_sum": final_loss,
                "train/image_label_ratio": image_label_ratio,
                "train/lang_label_ratio": lang_label_ratio,
            }
        )

    # print(f"Train loss: {total_loss/total_samples}")
    wandb.log(
        {
            # "train_avg/surface_loss": surface_loss / total_samples,
            "train_avg/contrastive_loss_labels": label_loss / total_samples,
            "train_avg/contrastive_loss_images": image_loss / total_samples,
            "train_avg/instance_loss": total_inst_segmentation_loss / total_samples,
            "train_avg/instance_accuracy": total_accuracy / total_samples,
            "train_avg/loss_sum": total_loss / total_samples,
            "train_avg/labelling_temp": torch.exp(
                labelling_model.temperature.data.detach()
            ),
        }
    )


def test(test_loader, clip_test_loader, surface_model, labelling_model):
    surface_model.eval()
    labelling_model.eval()
    with torch.no_grad():
        total_loss = 0
        total_samples = 0
        # surface_loss = 0
        label_loss = 0
        image_loss = 0
        total_acc = 0
        total_inst_segmentation_loss = 0
        for clip_data_dict in clip_test_loader:
            # positives, negatives = extract_positive_and_negative_surface_examples(
            #     datapoint_dict
            # )
            # data = torch.cat([positives, negatives])
            # labels = (
            #     torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))])
            #     .unsqueeze(-1)
            #     .to(DEVICE)
            # )

            # # idx = torch.randperm(data.shape[0])

            # output = surface_model(data)
            # loss = 0  # loss_fn(output, labels)

            xyzs = clip_data_dict["xyz"].to(DEVICE)
            clip_labels = clip_data_dict["clip_vector"].to(DEVICE)
            clip_image_labels = clip_data_dict["clip_image_vector"].to(DEVICE)
            weights = torch.exp(-EXP_DECAY_COEFF * clip_data_dict["distance"]).to(
                DEVICE
            )
            label_weights = clip_data_dict["semantic_weight"].to(DEVICE)
            # predicted_latents = labelling_model(xyzs)
            (
                predicted_label_latents,
                predicted_image_latents,
                segmentation_logits,
            ) = labelling_model(xyzs)
            image_label_index = clip_data_dict["img_idx"].to(DEVICE).reshape(-1, 1)
            language_label_index = clip_data_dict["label"].to(DEVICE).reshape(-1, 1)
            instances = clip_data_dict["instance"].to(DEVICE).reshape(-1)
            batch_size = len(image_label_index)
            image_label_mask = (
                image_label_index != image_label_index.t()
            ).float() + torch.eye(batch_size, device=DEVICE)
            language_label_mask = (
                language_label_index != language_label_index.t()
            ).float() + torch.eye(batch_size, device=DEVICE)

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
                language_label_index,
                language_label_mask,
            )
            contrastive_loss = (
                IMAGE_TO_LABEL_CLIP_LOSS_SCALE * contrastive_loss_images
                + LABEL_TO_IMAGE_LOSS_SCALE * contrastive_loss_labels
                + INSTANCE_LOSS_SCALE * inst_segmentation_loss
            )
            final_loss = contrastive_loss

            # surface_loss += loss.cpu().item()
            label_loss += contrastive_loss_labels.cpu().item()
            image_loss += contrastive_loss_images.cpu().item()
            total_loss += final_loss.cpu().item()
            total_inst_segmentation_loss += inst_segmentation_loss.cpu().item()
            total_acc += accuracy
            total_samples += 1

            wandb.log(
                {
                    # "test/surface_loss": loss,
                    "test/contrastive_loss_label": contrastive_loss_labels,
                    "test/contrastive_loss_image": contrastive_loss_images,
                    "test/instance_loss": inst_segmentation_loss,
                    "test/instance_accuracy": accuracy,
                    "test/loss_sum": final_loss,
                }
            )

    # print(f"Test loss: {total_loss/total_samples}")
    wandb.log(
        {
            # "test_avg/surface_loss": surface_loss / total_samples,
            "test_avg/contrastive_loss_label": label_loss / total_samples,
            "test_avg/contrastive_loss_image": image_loss / total_samples,
            "test_avg/instance_loss": total_inst_segmentation_loss / total_samples,
            "test_avg/instance_accuracy": total_acc / total_samples,
            "test_avg/loss_sum": total_loss / total_samples,
        }
    )
    torch.save(
        surface_model,
        f"outputs/implicit_models/{SAVE_DIRECTORY}/implicit_scene_surface_model_{epoch}.pt",
    )
    torch.save(
        labelling_model,
        f"outputs/implicit_models/{SAVE_DIRECTORY}/implicit_scene_label_model_{epoch}.pt",
    )


def get_dataset():
    dataset = HabitatViewDataset(
        habitat_scenes=SCENE_FILEPATH,
        pose_extractor_grid_size=6,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        height_levels=0,
    )
    train_split_size = len(dataset) // 2
    view_train_dataset, view_test_dataset = random_split(
        dataset,
        lengths=[train_split_size, len(dataset) - train_split_size],
    )
    location_train_dataset_1 = DeticDenseLabelledDataset(
        view_train_dataset,
    )
    # Now we will have to create more dataloaders for the CLIP dataset.
    location_train_dataset = HabitatLocationDataset(
        habitat_view_ds=view_train_dataset,
        subsample_prob=SUBSAMPLE_PROB,
        return_nonsegmented_images=False,
    )
    location_test_dataset = HabitatLocationDataset(
        habitat_view_ds=view_test_dataset,
        subsample_prob=SUBSAMPLE_PROB,
        selective_instance_segmentation=False,
    )
    # Convert to clip datasets
    clip_train_dataset = ClipLabelledLocation(location_train_dataset)
    clip_test_dataset = ClipLabelledLocation(location_test_dataset)
    clip_train_dataset_concat = ConcatDataset(
        [clip_train_dataset, location_train_dataset_1]
    )
    return (
        clip_train_dataset,
        location_train_dataset,
        location_test_dataset,
        clip_train_dataset_concat,
        clip_test_dataset,
    )


# def get_dataset():
#     dataset = RealWorldSemanticDataset(REAL_SCENE_DIRECTORY)
#     surface_dataset = RealWorldSurfaceDataset(dataset, sampling_rate=0.2)
#     clip_dataset = RealWorldClipDataset(dataset, sampling_rate=0.2)

#     # Now split both of the datasets in half
#     surface_train_split_size = int(len(surface_dataset) * 0.98)
#     surface_train_set, surface_test_set = random_split(
#         surface_dataset,
#         lengths=[
#             surface_train_split_size,
#             len(surface_dataset) - surface_train_split_size,
#         ],
#     )
#     clip_train_split_size = int(len(clip_dataset) * 0.98)
#     clip_train_dataset, clip_test_dataset = random_split(
#         clip_dataset,
#         lengths=[clip_train_split_size, len(clip_dataset) - clip_train_split_size],
#     )
#     return (
#         dataset,
#         surface_train_set,
#         surface_test_set,
#         clip_train_dataset,
#         clip_test_dataset,
#     )


if __name__ == "__main__":
    # Run basic sanity test on the dataloader.
    (
        parent_dataset,
        view_train_dataset,
        view_test_dataset,
        clip_train_dataset,
        clip_test_dataset,
    ) = get_dataset()

    if MODEL_TYPE == "MLP":
        surface_model = ImplicitSurfaceModel()
        labelling_model = ImplicitCLIPModel()
    elif MODEL_TYPE == "hash":
        surface_model = GridSurfaceModel()
        labelling_model = GridCLIPModel(
            image_rep_size=parent_dataset.image_representation_size,
            text_rep_size=parent_dataset.text_representation_size,
            # segmentation_classes=256,
        )

    # Now, make the dataloader weights
    (
        surface_weights,
        surface_voxel_count,
    ) = get_voxel_normalized_sampler_and_occupied_voxels(view_train_dataset)
    surface_sampler = WeightedRandomSampler(
        weights=surface_weights, num_samples=surface_voxel_count
    )
    train_loader = DataLoader(
        view_train_dataset,
        batch_size=IMAGE_BATCH_SIZE,
        sampler=surface_sampler,
        pin_memory=True,
    )
    test_loader = DataLoader(
        view_test_dataset, batch_size=IMAGE_BATCH_SIZE, shuffle=False, pin_memory=True
    )
    (
        label_point_weights,
        label_voxel_count,
    ) = get_voxel_normalized_sampler_and_occupied_voxels(clip_train_dataset)
    label_sampler = WeightedRandomSampler(
        weights=label_point_weights, num_samples=label_voxel_count
    )
    clip_train_loader = DataLoader(
        clip_train_dataset,
        batch_size=POINT_BATCH_SIZE,
        sampler=label_sampler,
        pin_memory=True,
    )
    clip_test_loader = DataLoader(
        clip_test_dataset, batch_size=POINT_BATCH_SIZE, shuffle=False, pin_memory=True
    )

    logging.info(
        f"Train loader sizes: surface {len(train_loader)}, clip {len(clip_train_loader)}"
    )
    logging.info(
        f"Test loader sizes: surface {len(test_loader)}, clip {len(clip_test_loader)}"
    )

    surface_model = surface_model.to(DEVICE)
    labelling_model = labelling_model.to(DEVICE)
    optim = torch.optim.Adam(
        chain(surface_model.parameters(), labelling_model.parameters()),
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    wandb.init(
        project="implicit_clip_model",
        tags=[
            f"model/{MODEL_TYPE}",
        ],
    )

    os.makedirs(
        "outputs/implicit_models/{}/".format(SAVE_DIRECTORY),
        exist_ok=True,
    )

    for epoch in range(EPOCHS):
        train(
            train_loader,
            clip_train_loader,
            surface_model,
            labelling_model,
            optim,
        )
        if epoch % EVAL_EVERY == 0:
            test(test_loader, clip_test_loader, surface_model, labelling_model)
