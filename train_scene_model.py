from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra

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
            self.cfg.positional_encoder_view
        )
        self.query_xyz_encoder: PositionalEmbedding = hydra.utils.instantiate(
            self.cfg.positional_encoder_query
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

    def run(self):
        pass

    def train_epoch(self):
        pass

    def test_epoch(self):
        pass


@hydra.main(version_base="1.2", config_path="configs", config_name="scene_model.yaml")
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
