import logging
from typing import Dict, List
import torch
import hydra
import tqdm
import wandb

from itertools import chain
from torch.utils.data import DataLoader, random_split
from omegaconf import DictConfig, OmegaConf
from models.task_decoders.abstract_decoder import AbstractDecoder
from models.view_encoders.abstract_view_encoder import AbstractViewEncoder
from dataloaders.habitat_loaders import HabitatLocationDataset, HabitatViewDataset
from models.scene_models.positional_encoding import PositionalEmbedding
from models.scene_transformer import SceneTransformer
from utils import cycle


logging.basicConfig(filename="training.log", level=logging.DEBUG)


class Workspace:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        wandb.config = config_dict
        wandb.init(project=cfg.wandb.project, tags=cfg.wandb.tags, config=config_dict)

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

        self.decoders: List[AbstractDecoder] = []
        for decoder_cfg in [self.cfg.decoder]:
            self.decoders.append(hydra.utils.instantiate(decoder_cfg))
            self.decoders[-1].register_embedding_map(
                self.habitat_view_encoder.semantic_embedding_layer
            )

        # Setup optimizers.
        optimizable_params = [
            self.habitat_view_encoder.parameters(),
            self.view_quat_encoder.parameters(),
            self.view_xyz_encoder.parameters(),
            self.query_xyz_encoder.parameters(),
        ]
        for st in self.scene_transformers.values():
            optimizable_params.append(st.parameters())
        for decoder in self.decoders:
            optimizable_params.append(decoder.parameters())
        # self.optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
        #     config=self.cfg.optimizer,
        #     params=chain(*optimizable_params),
        # )
        self.optimizer = torch.optim.Adam(
            params=chain(*optimizable_params),
            lr=1e-4,
        )

        self._setup_datasets()

    def _setup_datasets(self):
        self.view_dataset: HabitatViewDataset = hydra.utils.instantiate(
            self.cfg.dataset.view_dataset
        )
        train_split_size = int(len(self.view_dataset) * self.cfg.train_split_size)
        self.view_train_dataset, self.view_test_dataset = random_split(
            self.view_dataset,
            lengths=[train_split_size, len(self.view_dataset) - train_split_size],
        )
        self.xyz_train_dataset: HabitatLocationDataset = hydra.utils.instantiate(
            self.cfg.dataset.xyz_dataset, habitat_view_ds=self.view_train_dataset
        )
        self.xyz_test_dataset: HabitatLocationDataset = hydra.utils.instantiate(
            self.cfg.dataset.xyz_dataset, habitat_view_ds=self.view_test_dataset
        )

        self.view_train_dataloader = DataLoader(
            self.view_train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            pin_memory=True,
        )

        self.view_test_dataloader = DataLoader(
            self.view_test_dataset,
            batch_size=self.cfg.batch_size,
            pin_memory=True,
        )

        self.xyz_train_dataloader = DataLoader(
            self.xyz_train_dataset,
            shuffle=True,
            batch_size=self.cfg.xyz_batch_size,
            pin_memory=True,
        )

        self.xyz_test_dataloader = DataLoader(
            self.xyz_test_dataset,
            shuffle=True,
            batch_size=self.cfg.xyz_batch_size,
            pin_memory=True,
        )

    def run(self):
        postfix_dict = {}
        iterator = tqdm.trange(self.cfg.train_epochs)
        for epoch in iterator:
            postfix_dict["train_loss"] = self.train_epoch()
            if (epoch + 1) % self.cfg.eval_every == 0:
                save_data = (self.cfg.train_epochs - epoch) <= self.cfg.eval_every
                postfix_dict["test_loss"] = self.test_epoch(
                    save_decoded_results=save_data
                )
            iterator.set_postfix(postfix_dict)
            logging.info(str(postfix_dict))

    def train_epoch(self) -> float:
        avg_loss = 0
        iters = 0
        for views, xyz in tqdm.tqdm(
            zip(cycle(self.view_train_dataloader), self.xyz_train_dataloader),
            total=len(self.xyz_train_dataloader),
        ):
            self.optimizer.zero_grad(set_to_none=True)
            xyz_coordinates = xyz["xyz"].to(self.cfg.device)
            views_dict, xyz_dict = (
                {
                    k: v.to(self.cfg.device)
                    for k, v in data.items()
                    if k not in ["scene_name"]
                }
                for data in (views, xyz)
            )
            encoded_view, encoded_response = self.scene_transformers[
                self.scene_names[0]
            ](views_dict, xyz_coordinates)
            total_loss = 0.0
            for decoder in self.decoders:
                decoded_view, decoded_response = decoder.decode_representations(
                    encoded_view, encoded_response
                )
                loss, loss_dict = decoder.compute_detailed_loss(
                    decoded_view, decoded_response, ground_truth=(views_dict, xyz_dict)
                )
                wandb.log({f"train/{k}": v for k, v in loss_dict.items()})
                total_loss += loss

            avg_loss += total_loss.detach().cpu().item()
            iters += len(xyz_coordinates)
            total_loss.backward()
            self.optimizer.step()
            wandb.log({f"train/avg_loss": avg_loss / iters})
        return avg_loss / iters

    def test_epoch(self, save_decoded_results: bool = True) -> float:
        epoch_loss = 0
        epoch_samples = 0
        if save_decoded_results:
            decoded_views, decoded_responses = [], []
        for views, xyz in zip(self.view_test_dataloader, self.xyz_test_dataloader):
            with torch.no_grad():
                views_dict = {
                    k: v.to(self.cfg.device)
                    for k, v in views.items()
                    if k not in ["scene_name"]
                }
                xyz_dict = {k: v.to(self.cfg.device) for k, v in xyz.items()}
                xyz_coordinates = xyz_dict["xyz"]
                encoded_view, encoded_response = self.scene_transformers[
                    self.scene_names[0]
                ](views_dict, xyz_coordinates)
                total_loss = 0.0
                for decoder in self.decoders:
                    decoded_view, decoded_response = decoder.decode_representations(
                        encoded_view, encoded_response
                    )
                    loss, loss_dict = decoder.compute_detailed_loss(
                        decoded_view,
                        decoded_response,
                        ground_truth=(views_dict, xyz_dict),
                    )
                    if save_decoded_results:
                        decoded_views.append(decoded_view.detach().cpu())
                        decoded_responses.append(decoded_response.detach().cpu())
                    wandb.log({f"test/{k}": v for k, v in loss_dict.items()})
                    total_loss += loss
                epoch_loss += total_loss.detach().cpu().item()
                epoch_samples += len(xyz_coordinates)
        wandb.log({f"test/avg_loss": epoch_loss / epoch_samples})
        if save_decoded_results:
            torch.save(
                torch.cat(decoded_views), f"{self.cfg.save_path}/decoded_views.pt"
            )
            torch.save(
                torch.cat(decoded_responses),
                f"{self.cfg.save_path}/decoded_responses.pt",
            )
        return epoch_loss / epoch_samples


@hydra.main(version_base="1.2", config_path="configs", config_name="scene_model.yaml")
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
