import clip
import einops
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader, random_split

from dataloaders.habitat_loaders import HabitatViewDataset
from models.scene_models.positional_encoding import FourierFeatures
from utils.mlp import MLP

import wandb

# Create new dataset which is just the mapping of scenes to CLIP vectors.
class ClipDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        scene_dataset: HabitatViewDataset,
        clip_model_name: str = "ViT-B/32",
        device: str = "cuda",
        batch_size: int = 64,
    ):
        model, preprocess = clip.load(clip_model_name, device=device)
        self._all_camera_loc = []
        self._all_camera_dir = []
        self._all_clip_embeddings = []
        # Create dataloader
        dataloader = DataLoader(
            scene_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
        )
        self.device = device
        self._preprocess_and_generate_dataset(dataloader, model)

    def _preprocess_and_generate_dataset(self, dataloader, model):
        for data_dict in tqdm.tqdm(dataloader):
            self._all_camera_loc.append(data_dict["camera_pos"])
            self._all_camera_dir.append(data_dict["camera_direction"])
            embedding_vectors = (
                model.encode_image(
                    einops.rearrange(
                        data_dict["rgb"][..., :3], "b w h c -> b c w h"
                    ).to(self.device)
                )
                .detach()
                .cpu()
            )
            self._all_clip_embeddings.append(embedding_vectors)
        self._all_camera_loc = torch.cat(self._all_camera_loc, dim=0)
        self._all_camera_dir = torch.cat(self._all_camera_dir, dim=0)
        self._all_clip_embeddings = torch.cat(self._all_clip_embeddings, dim=0)

    def __getitem__(self, idx):
        return {
            "location": self._all_camera_loc[idx],
            "direction": self._all_camera_dir[idx],
            "embedding": self._all_clip_embeddings[idx],
        }

    def __len__(self):
        return len(self._all_camera_loc)


class ImplicitCLIPModel(nn.Module):
    def __init__(
        self,
        depth: int = 4,
        width: int = 1024,
        batchnorm: bool = False,
        use_camera_dir: bool = True,
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
            fourier_embedding_scale=1 / (2.0 ** 2),
        )
        self.trunk = MLP(
            input_dim=fourier_dim,
            output_dim=clip_dim,
            batchnorm=batchnorm,
            hidden_depth=depth,
            hidden_dim=width,
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def to(self, device):
        self.fourier_proj = self.fourier_proj.to(device)
        self.trunk = self.trunk.to(device)
        self.temperature.data = self.temperature.data.to(device)
        return self

    def forward(self, x: torch.Tensor):
        return self.trunk(self.fourier_proj(x))

    def compute_loss(self, predicted_latents, actual_latents):
        temp = torch.exp(self.temperature)
        sim = torch.einsum("i d, j d -> i j", predicted_latents, actual_latents) * temp
        labels = torch.arange(len(predicted_latents), device=predicted_latents.device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
        return loss


def compute_contrastive_loss(predicted_latents, actual_latents, temp=1.0):
    sim = torch.einsum("i d, j d -> i j", predicted_latents, actual_latents) * temp
    labels = torch.arange(len(predicted_latents), device=predicted_latents.device)
    loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
    return loss


if __name__ == "__main__":
    # Replace with the path to your scene file
    SCENE_FILEPATH = [
        "/private/home/notmahi/data/replica_dataset/room_0/habitat/mesh_semantic.ply",
    ]
    GRID_SIZE = 20
    HEIGHT_LEVELS = 3
    DEVICE = "cuda"
    BATCH_SIZE = 256
    EPOCHS = 2500

    dataset = HabitatViewDataset(
        habitat_scenes=SCENE_FILEPATH,
        pose_extractor_grid_size=GRID_SIZE,
        image_size=(224, 224),
        height_levels=HEIGHT_LEVELS,
        use_cache=False,
    )

    train_split_size = len(dataset) // 2

    # Now train this model.

    view_train_dataset, view_test_dataset = random_split(
        dataset,
        lengths=[train_split_size, len(dataset) - train_split_size],
    )

    if not os.path.exists("/private/home/notmahi/data/clip_embeddings/train_data.pt"):
        clip_train_data = ClipDataset(view_train_dataset, batch_size=BATCH_SIZE)
        torch.save(
            clip_train_data, "/private/home/notmahi/data/clip_embeddings/train_data.pt"
        )
    else:
        clip_train_data = torch.load(
            "/private/home/notmahi/data/clip_embeddings/train_data.pt"
        )

    if not os.path.exists("/private/home/notmahi/data/clip_embeddings/test_data.pt"):
        clip_test_data = ClipDataset(view_test_dataset, batch_size=BATCH_SIZE)
        torch.save(
            clip_test_data, "/private/home/notmahi/data/clip_embeddings/test_data.pt"
        )
    else:
        clip_test_data = torch.load(
            "/private/home/notmahi/data/clip_embeddings/test_data.pt"
        )

    train_loader = DataLoader(
        clip_train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
    )
    test_loader = DataLoader(
        clip_test_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True
    )

    model = ImplicitCLIPModel(
        depth=4,
        width=512,
        batchnorm=True,
    ).float()
    model = model.to(DEVICE)
    optim = torch.optim.Adam(
        model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01
    )
    loss_fn = compute_contrastive_loss

    wandb.init(
        project="implicit_clip_model",
    )

    for epoch in tqdm.trange(EPOCHS):
        total_loss = 0
        total_samples = 0
        model.train()
        for datapoint_dict in train_loader:
            camera_location = datapoint_dict["location"].to(DEVICE)
            camera_direction = datapoint_dict["direction"].to(DEVICE)
            latent_vectors = datapoint_dict["embedding"].to(DEVICE).float()

            data = torch.cat([camera_location, camera_direction], dim=-1).float()
            labels = latent_vectors

            optim.zero_grad()
            output = model(data)
            output = F.normalize(output, p=2, dim=-1)
            labels = F.normalize(labels, p=2, dim=-1)
            loss = model.compute_loss(output, labels)
            loss.backward()
            optim.step()
            total_loss += loss.detach().cpu().item()
            total_samples += len(labels)

        # wandb.log({"Train loss" : total_loss/total_samples})
        train_loss = total_loss / total_samples
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                total_loss = 0
                total_samples = 0
                for datapoint_dict in test_loader:
                    camera_location = datapoint_dict["location"].to(DEVICE)
                    camera_direction = datapoint_dict["direction"].to(DEVICE)
                    latent_vectors = datapoint_dict["embedding"].to(DEVICE).float()

                    data = torch.cat(
                        [camera_location, camera_direction], dim=-1
                    ).float()
                    labels = latent_vectors
                    labels = F.normalize(labels, p=2, dim=-1)

                    output = model(data)
                    output = F.normalize(output, p=2, dim=-1)
                    loss = model.compute_loss(output, labels)
                    total_loss += loss.detach().cpu().item()
                    total_samples += len(labels)

            wandb.log(
                {"train/loss": train_loss, "test/loss": total_loss / total_samples}
            )
            torch.save(
                model,
                f"outputs/2022-07-06/implicit_model/implicit_clip_small_fourier_{epoch}.pt",
            )
