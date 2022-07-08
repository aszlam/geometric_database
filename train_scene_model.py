import logging
from typing import Dict, List, Optional
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
            lr=self.cfg.optimizer.lr,
            betas=(0.9, 0.99),
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
            self._num_epoch = epoch
            postfix_dict["train_loss"] = self.train_epoch(epoch=epoch)
            if (epoch + 1) % self.cfg.eval_every == 0:
                # save_data = (self.cfg.train_epochs - epoch) <= self.cfg.eval_every
                save_data = True
                postfix_dict["test_loss"] = self.test_epoch(
                    epoch=epoch, save_decoded_results=save_data
                )
            iterator.set_postfix(postfix_dict)
            logging.info(str(postfix_dict))

    def train_epoch(self, epoch: Optional[int] = None) -> float:
        avg_loss = 0
        iters = 0
        self.scene_transformers[self.scene_names[0]].set_mask_prob(
            mask_prob=0.1, pos_mask_prob=0.2 + 0.8 * (epoch / self.cfg.train_epochs)
        )
        self.scene_transformers[self.scene_names[0]].train()
        for decoder in self.decoders:
            decoder.train()
        for views, xyz in tqdm.tqdm(
            zip(self.view_train_dataloader, cycle(self.xyz_train_dataloader)),
            total=len(self.view_train_dataloader),
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

            total_loss.backward()
            avg_loss += total_loss.detach().cpu().item()
            iters += 1
            self.optimizer.step()
        wandb.log({f"train/avg_loss": avg_loss / iters})
        return avg_loss / iters

    def test_epoch(
        self, epoch: Optional[int] = None, save_decoded_results: bool = True
    ) -> float:
        epoch_loss = 0
        epoch_samples = 0
        if save_decoded_results:
            decoded_views, decoded_responses, actual_xyz, actual_rgb = [], [], [], []
        self.scene_transformers[self.scene_names[0]].set_mask_prob(
            mask_prob=0.05, pos_mask_prob=0.95
        )

        for decoder in self.decoders:
            decoder.eval()
        self.scene_transformers[self.scene_names[0]].eval()
        with torch.no_grad():
            for views, xyz in zip(
                self.view_test_dataloader, cycle(self.xyz_test_dataloader)
            ):
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
                    decoder.eval()
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
                        actual_xyz.append(views["xyz_position"][..., ::32, ::32])
                        actual_rgb.append(views["rgb"][..., :3, ::32, ::32])
                    wandb.log({f"test/{k}": v for k, v in loss_dict.items()})
                    total_loss += loss
                epoch_loss += total_loss.detach().cpu().item()
                epoch_samples += 1
        if save_decoded_results:
            torch.save(
                torch.cat(decoded_views),
                f"{self.cfg.save_path}/{self._num_epoch}_decoded_views.pt",
            )
            torch.save(
                torch.cat(decoded_responses),
                f"{self.cfg.save_path}/{self._num_epoch}_decoded_responses.pt",
            )
            torch.save(
                torch.cat(actual_rgb),
                f"{self.cfg.save_path}/{self._num_epoch}_gt_rgb.pt",
            )
            torch.save(
                torch.cat(actual_xyz),
                f"{self.cfg.save_path}/{self._num_epoch}_gt_xyz.pt",
            )
        wandb.log({f"test/avg_loss": epoch_loss / epoch_samples})
        return epoch_loss / epoch_samples


@hydra.main(version_base="1.2", config_path="configs", config_name="scene_model.yaml")
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
