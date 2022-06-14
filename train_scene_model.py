from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
import tqdm

from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from models.view_encoders.abstract_view_encoder import AbstractViewEncoder
from dataloaders.habitat_loaders import HabitatLocationDataset, HabitatViewDataset
from models.scene_models.positional_encoding import PositionalEmbedding
from models.scene_transformer import SceneTransformer


class Workspace:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        self.habitat_view_encoder: AbstractViewEncoder = hydra.utils.instantiate(
            self.cfg.view_encoder
        )
        self.view_xyz_encoder: PositionalEmbedding = hydra.utils.instantiate(
            self.cfg.positional_encoder_xyz
        )
        self.query_xyz_encoder: PositionalEmbedding = hydra.utils.instantiate(
            self.cfg.positional_encoder_xyz
        )
        self.view_quat_encoder: PositionalEmbedding = hydra.utils.instantiate(
            self.cfg.positional_encoder_quat
        )

        # Make one scene transformer for each scene
        self.scene_transformers: Dict[str, SceneTransformer] = {}
        self.scene_names: List[str] = list(self.cfg.dataset.view_dataset.habitat_scenes)
        for scene_name in self.scene_names:
            current_scene_transformer: SceneTransformer = hydra.utils.instantiate(
                self.cfg.scene_model.scene_transformer
            )
            current_scene_transformer.register_encoders(
                view_encoder=self.habitat_view_encoder,
                positional_encoder_view=self.view_xyz_encoder,
                positional_encoder_query=self.query_xyz_encoder,
                quat_encoder=self.view_quat_encoder,
            )
            self.scene_transformers[scene_name] = current_scene_transformer

        self.view_dataset: HabitatViewDataset = hydra.utils.instantiate(
            self.cfg.dataset.view_dataset
        )
        self.xyz_dataset: HabitatLocationDataset = hydra.utils.instantiate(
            self.cfg.dataset.xyz_dataset, habitat_view_ds=self.view_dataset
        )

        self.view_dataloader = DataLoader(
            self.view_dataset,
            batch_size=self.cfg.batch_size,
            # num_workers=self.cfg.num_workers,
            pin_memory=True,
        )
        self.xyz_dataloader = DataLoader(
            self.xyz_dataset,
            batch_size=self.cfg.batch_size,
            # num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

    def run(self):
        for epoch in range(self.cfg.train_epochs):
            self.train_epoch()
            if (epoch + 1) % self.cfg.eval_every == 0:
                self.test_epoch()

    def train_epoch(self):
        for views, xyz in tqdm.tqdm(
            zip(self.view_dataloader, self.xyz_dataloader),
            total=len(self.view_dataloader),
        ):
            xyz_coordinates = xyz["xyz"].to(self.cfg.device)
            views_dict = {
                k: v.to(self.cfg.device)
                for k, v in views.items()
                if k not in ["scene_name"]
            }
            xyz_coordinates = xyz_coordinates.to(self.cfg.device)
            encoded_query_response = self.scene_transformers[self.scene_names[0]](
                views_dict, xyz_coordinates
            )
        print(encoded_query_response.shape)
        # TODO: Decode and compute losses.

    def test_epoch(self):
        print("TODO: Test")


@hydra.main(version_base="1.2", config_path="configs", config_name="scene_model.yaml")
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
