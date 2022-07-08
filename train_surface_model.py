from dataloaders.habitat_loaders import HabitatViewDataset, HabitatLocationDataset
import tqdm
from torch.utils.data import DataLoader, random_split
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.mlp import MLP
from models.scene_models.positional_encoding import FourierFeatures


# Replace with the path to your scene file
SCENE_FILEPATH = [
    "/private/home/notmahi/data/replica_dataset/room_0/habitat/mesh_semantic.ply",
    # "/private/home/notmahi/data/replica_dataset/room_1/habitat/mesh_semantic.ply",
    # "/private/home/notmahi/data/replica_dataset/room_2/habitat/mesh_semantic.ply",
]
BATCH_SIZE = 128

# Run basic sanity test on the dataloader.
dataset = HabitatViewDataset(
    habitat_scenes=SCENE_FILEPATH,
    pose_extractor_grid_size=5,
    image_size=(224, 224),
    height_levels=3,
)


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


DEVICE = "cuda"
BATCH_SIZE = 64
EPOCHS = 5000

train_split_size = len(dataset) // 2

# Now train this model.

view_train_dataset, view_test_dataset = random_split(
    dataset,
    lengths=[train_split_size, len(dataset) - train_split_size],
)

train_loader = DataLoader(
    view_train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
)
test_loader = DataLoader(
    view_test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True
)

model = ImplicitSurfaceModel()
model = model.to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
loss_fn = nn.BCEWithLogitsLoss()

for epoch in tqdm.trange(EPOCHS):
    total_loss = 0
    total_samples = 0
    for datapoint_dict in train_loader:
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
        positives = einops.rearrange(subsampled_xyz_position, "b d w h -> (b w h) d")
        negatives = einops.rearrange(negative_samples, "b d w h -> (b w h) d")

        data = torch.cat([positives, negatives])
        labels = (
            torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))])
            .unsqueeze(-1)
            .to(DEVICE)
        )

        idx = torch.randperm(data.shape[0]).to(DEVICE)

        optim.zero_grad()
        output = model(data[idx])
        loss = loss_fn(output, labels[idx])
        loss.backward()
        optim.step()
        total_loss += loss.detach().cpu().item()
        total_samples += len(labels)

    print(f"Train loss: {total_loss/total_samples}")

    if epoch % 5 == 0:
        with torch.no_grad():
            total_loss = 0
            total_samples = 0
            total_correct = 0
            for datapoint_dict in test_loader:
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
                negatives = einops.rearrange(negative_samples, "b d w h -> (b w h) d")

                data = torch.cat([positives, negatives])
                labels = (
                    torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))])
                    .unsqueeze(-1)
                    .to(DEVICE)
                )

                idx = torch.randperm(data.shape[0])

                output = model(data[idx])
                loss = loss_fn(output, labels[idx])
                output_label = (torch.sigmoid(output) >= 0.5).long()
                total_correct += (output_label == labels[idx]).sum()
                total_loss += loss
                total_samples += len(labels)

        print(
            f"Test loss: {total_loss/total_samples}, total correct {total_correct/total_samples}"
        )
        torch.save(
            model, f"outputs/2022-07-06/implicit_model/implicit_scene_model_{epoch}.pt"
        )
