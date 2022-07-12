from dataloaders.habitat_loaders import HabitatViewDataset, HabitatLocationDataset
from dataloaders.clip_labeled_habitat import ClipLabelledLocation
import tqdm
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from utils.mlp import MLP
from models.scene_models.positional_encoding import FourierFeatures
from itertools import chain, cycle

import wandb

# Replace with the path to your scene file
SCENE_FILEPATH = [
    "/private/home/notmahi/data/replica_dataset/room_0/habitat/mesh_semantic.ply",
    # "/private/home/notmahi/data/replica_dataset/room_1/habitat/mesh_semantic.ply",
    # "/private/home/notmahi/data/replica_dataset/room_2/habitat/mesh_semantic.ply",
]
BATCH_SIZE = 256

# Create model using a simple MLP and a Fourier projection.
# This model should really tell you the probability of something being a surface point or not.


class ImplicitSurfaceModel(nn.Module):
    def __init__(
        self,
        depth: int = 3,
        width: int = 512,
        batchnorm: bool = False,
        fourier_dim: int = 256,
    ):
        super().__init__()
        self.fourier_proj = FourierFeatures(
            input_dim=3, fourier_embedding_dim=fourier_dim
        )
        self.trunk = MLP(
            input_dim=fourier_dim,
            output_dim=1,
            batchnorm=batchnorm,
            hidden_depth=depth,
            hidden_dim=width,
        )

    def to(self, device):
        self.fourier_proj = self.fourier_proj.to(device)
        self.trunk = self.trunk.to(device)
        return self

    def forward(self, x: torch.Tensor):
        return self.trunk(self.fourier_proj(x))


class ImplicitCLIPModel(nn.Module):
    def __init__(
        self,
        depth: int = 4,
        width: int = 512,
        batchnorm: bool = False,
        use_camera_dir: bool = False,
        fourier_dim: int = 256,
        clip_dim: int = 512,
    ):
        super().__init__()
        input_dim = 3
        if use_camera_dir:
            input_dim += 4
        self.fourier_proj = FourierFeatures(
            input_dim=input_dim,
            fourier_embedding_dim=fourier_dim,
            fourier_embedding_scale=1 / (2.0**2),
        )
        self.trunk_1 = MLP(
            input_dim=fourier_dim,
            output_dim=width,
            batchnorm=batchnorm,
            hidden_depth=depth // 2,
            hidden_dim=width,
        )
        self.trunk_2 = MLP(
            input_dim=width + fourier_dim,
            output_dim=clip_dim,
            batchnorm=batchnorm,
            hidden_depth=depth // 2,
            hidden_dim=width,
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def to(self, device):
        self.fourier_proj = self.fourier_proj.to(device)
        self.trunk_1 = self.trunk_1.to(device)
        self.trunk_2 = self.trunk_2.to(device)
        self.temperature.data = self.temperature.data.to(device)
        return self

    def forward(self, x: torch.Tensor):
        projected_x = self.fourier_proj(x)
        return self.trunk_2(torch.cat([self.trunk_1(projected_x), projected_x], dim=-1))

    def compute_loss(self, predicted_latents, actual_latents, weights=None):
        temp = torch.exp(self.temperature)
        sim = torch.einsum("i d, j d -> i j", predicted_latents, actual_latents) * temp
        labels = torch.arange(len(predicted_latents), device=predicted_latents.device)
        if weights is None:
            loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
        else:
            loss = (
                F.cross_entropy(sim, labels, reduction="none")
                + F.cross_entropy(sim.t(), labels, reduction="none")
            ) / 2
            loss = (loss * weights).mean()
        return loss


DEVICE = "cuda"
IMAGE_BATCH_SIZE = 256
IMAGE_SIZE = 224
POINT_BATCH_SIZE = IMAGE_BATCH_SIZE * 7 * 7
IMAGE_TO_LABEL_CLIP_LOSS_SCALE = 2
EXP_DECAY_COEFF = 0.5
# EPOCHS = 5000
EPOCHS = 100
SUBSAMPLE_PROB = 0.03
EVAL_EVERY = 1
SURFACE_LOSS_LAMBDA = 10.0


if __name__ == "__main__":
    # Run basic sanity test on the dataloader.
    dataset = HabitatViewDataset(
        habitat_scenes=SCENE_FILEPATH,
        pose_extractor_grid_size=5,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        height_levels=3,
    )
    train_split_size = len(dataset) // 2
    view_train_dataset, view_test_dataset = random_split(
        dataset,
        lengths=[train_split_size, len(dataset) - train_split_size],
    )
    # Now we will have to create more dataloaders for the CLIP dataset.
    location_train_dataset = HabitatLocationDataset(
        habitat_view_ds=view_train_dataset, subsample_prob=SUBSAMPLE_PROB
    )
    location_test_dataset = HabitatLocationDataset(
        habitat_view_ds=view_test_dataset, subsample_prob=SUBSAMPLE_PROB
    )
    # Convert to clip datasets
    clip_train_dataset = ClipLabelledLocation(location_train_dataset)
    clip_test_dataset = ClipLabelledLocation(location_test_dataset)

    train_loader = DataLoader(
        view_train_dataset, batch_size=IMAGE_BATCH_SIZE, shuffle=True, pin_memory=True
    )
    test_loader = DataLoader(
        view_test_dataset, batch_size=IMAGE_BATCH_SIZE, shuffle=False, pin_memory=True
    )

    clip_train_loader = DataLoader(
        clip_train_dataset, batch_size=POINT_BATCH_SIZE, shuffle=True, pin_memory=True
    )
    clip_test_loader = DataLoader(
        clip_test_dataset, batch_size=POINT_BATCH_SIZE, shuffle=False, pin_memory=True
    )

    surface_model = ImplicitSurfaceModel()
    labelling_model = ImplicitCLIPModel()

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
    )

    for epoch in range(EPOCHS):
        total_loss = 0
        surface_loss = 0
        label_loss = 0
        image_loss = 0
        total_samples = 0
        surface_model.train()
        labelling_model.train()
        for datapoint_dict, clip_data_dict in tqdm.tqdm(
            zip(cycle(train_loader), clip_train_loader), total=len(clip_train_loader)
        ):
            xyz_positions = datapoint_dict["xyz_position"].to(DEVICE)
            subsampled_xyz_position = xyz_positions[..., ::7, ::7]
            camera_location = datapoint_dict["camera_pos"].to(DEVICE)
            extended_camera_loc = einops.rearrange(
                camera_location, "... (d 1 1) -> ... d 1 1"
            )
            # Now figure out negative samples between the two points.
            random_batchsize = torch.rand(len(xyz_positions)).to(DEVICE)
            negative_samples = random_batchsize[
                :, None, None, None
            ] * extended_camera_loc + (
                (1 - random_batchsize[:, None, None, None]) * subsampled_xyz_position
            )

            # Now reshape everything to shape
            positives = einops.rearrange(
                subsampled_xyz_position, "b d w h -> (b w h) d"
            )
            negatives = einops.rearrange(negative_samples, "b d w h -> (b w h) d")

            data = torch.cat([positives, negatives])
            labels = (
                torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))])
                .unsqueeze(-1)
                .to(DEVICE)
            )

            idx = torch.randperm(data.shape[0]).to(DEVICE)

            optim.zero_grad()
            # First, calculate the loss from the surface model.
            output = surface_model(data[idx])
            loss = loss_fn(output, labels[idx])

            # Now calculate loss from the labelling side
            xyzs = clip_data_dict["xyz"].to(DEVICE)
            clip_labels = clip_data_dict["clip_vector"].to(DEVICE)
            clip_image_labels = clip_data_dict["clip_image_vector"].to(DEVICE)
            weights = torch.exp(-EXP_DECAY_COEFF * clip_data_dict["distance"]).to(
                DEVICE
            )
            predicted_latents = labelling_model(xyzs)
            contrastive_loss_labels = labelling_model.compute_loss(
                predicted_latents, clip_labels
            )
            contrastive_loss_images = labelling_model.compute_loss(
                predicted_latents, clip_image_labels, weights=weights
            )
            contrastive_loss = (
                IMAGE_TO_LABEL_CLIP_LOSS_SCALE * contrastive_loss_images
                + contrastive_loss_labels
            )
            final_loss = contrastive_loss + SURFACE_LOSS_LAMBDA * loss
            final_loss.backward()
            optim.step()
            surface_loss += loss.detach().cpu().item()
            label_loss += contrastive_loss_labels.detach().cpu().item()
            image_loss += contrastive_loss_images.detach().cpu().item()
            total_loss += final_loss.detach().cpu().item()

            total_samples += 1
            wandb.log(
                {
                    "train/surface_loss": loss,
                    "train/contrastive_loss_labels": contrastive_loss_labels,
                    "train/contrastive_loss_images": contrastive_loss_images,
                    "train/loss_sum": final_loss,
                }
            )

        # print(f"Train loss: {total_loss/total_samples}")
        wandb.log(
            {
                "train_avg/surface_loss": surface_loss / total_samples,
                "train_avg/contrastive_loss_labels": label_loss / total_samples,
                "train_avg/contrastive_loss_images": image_loss / total_samples,
                "train_avg/loss_sum": total_loss / total_samples,
            }
        )

        if epoch % EVAL_EVERY == 0:
            surface_model.eval()
            labelling_model.eval()
            with torch.no_grad():
                total_loss = 0
                total_samples = 0
                surface_loss = 0
                label_loss = 0
                image_loss = 0
                for datapoint_dict, clip_data_dict in zip(
                    test_loader, clip_test_loader
                ):
                    xyz_positions = datapoint_dict["xyz_position"].to(DEVICE)
                    subsampled_xyz_position = xyz_positions[..., ::7, ::7]
                    camera_location = datapoint_dict["camera_pos"].to(DEVICE)
                    extended_camera_loc = einops.rearrange(
                        camera_location, "... (d 1 1) -> ... d 1 1"
                    )
                    # Now figure out negative samples between the two points.
                    random_batchsize = torch.rand(len(xyz_positions)).to(DEVICE)
                    # Test loop
                    negative_samples = random_batchsize[
                        :, None, None, None
                    ] * extended_camera_loc + (
                        (1 - random_batchsize[:, None, None, None])
                        * subsampled_xyz_position
                    )

                    # Now reshape everything to shape
                    positives = einops.rearrange(
                        subsampled_xyz_position, "b d w h -> (b w h) d"
                    )
                    negatives = einops.rearrange(
                        negative_samples, "b d w h -> (b w h) d"
                    )

                    data = torch.cat([positives, negatives])
                    labels = (
                        torch.cat(
                            [torch.ones(len(positives)), torch.zeros(len(negatives))]
                        )
                        .unsqueeze(-1)
                        .to(DEVICE)
                    )

                    # idx = torch.randperm(data.shape[0])

                    output = surface_model(data)
                    loss = loss_fn(output, labels)

                    xyzs = clip_data_dict["xyz"].to(DEVICE)
                    clip_labels = clip_data_dict["clip_vector"].to(DEVICE)
                    clip_image_labels = clip_data_dict["clip_image_vector"].to(DEVICE)
                    weights = torch.exp(
                        -EXP_DECAY_COEFF * clip_data_dict["distance"]
                    ).to(DEVICE)
                    predicted_latents = labelling_model(xyzs)
                    contrastive_loss_labels = labelling_model.compute_loss(
                        predicted_latents, clip_labels
                    )
                    contrastive_loss_images = labelling_model.compute_loss(
                        predicted_latents, clip_image_labels, weights=weights
                    )

                    contrastive_loss = (
                        IMAGE_TO_LABEL_CLIP_LOSS_SCALE * contrastive_loss_images
                        + contrastive_loss_labels
                    )
                    final_loss = contrastive_loss + SURFACE_LOSS_LAMBDA * loss

                    surface_loss += loss.cpu().item()
                    label_loss += contrastive_loss_labels.cpu().item()
                    image_loss += contrastive_loss_images.cpu().item()
                    total_loss += final_loss.cpu().item()
                    total_samples += 1

                    wandb.log(
                        {
                            "test/surface_loss": loss,
                            "test/contrastive_loss_label": contrastive_loss_labels,
                            "test/contrastive_loss_image": contrastive_loss_images,
                            "test/loss_sum": final_loss,
                        }
                    )

            # print(f"Test loss: {total_loss/total_samples}")
            wandb.log(
                {
                    "test_avg/surface_loss": surface_loss / total_samples,
                    "test_avg/contrastive_loss_label": label_loss / total_samples,
                    "test_avg/contrastive_loss_image": image_loss / total_samples,
                    "test_avg/loss_sum": total_loss / total_samples,
                }
            )
            torch.save(
                surface_model,
                f"outputs/2022-07-06/implicit_model/implicit_scene_surface_model_{epoch}.pt",
            )
            torch.save(
                labelling_model,
                f"outputs/2022-07-06/implicit_model/implicit_scene_label_model_{epoch}.pt",
            )
