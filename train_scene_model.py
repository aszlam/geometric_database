import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra


class Workspace:
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self):
        pass

    def train_epoch(self):
        pass

    def test_epoch(self):
        pass


@hydra.main(config_path="configs", config_name="scene_model.yaml")
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
