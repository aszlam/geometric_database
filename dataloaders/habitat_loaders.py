from typing import List, Union
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from habitat_sim.utils.data import ImageExtractor

import numpy as np
import matplotlib.pyplot as plt

from utils.habitat_utils import CUSTOM_POSE_EXTRACTOR, custom_pose_extractor_factory


class HabitatViewDataset(Dataset):
    """
    A dataloader that helps us load data from Habitat datasets.

    This custom dataloader gives us
    """

    def __init__(
        self,
        habitat_scenes: Union[str, List[str]],
        point_density: float = 0.1,
        view_components: List[str] = ["rgba", "depth", "semantic"],
        pose_extractor_grid_size: int = 50,
        height_levels: int = 5,
        transforms=transforms.Compose([transforms.ToTensor()]),
    ):
        # Sets the grid size and the height levels in the pose extractor
        custom_pose_extractor_factory(pose_extractor_grid_size, height_levels)
        self.image_extractor = ImageExtractor(
            scene_filepath=habitat_scenes,
            pose_extractor_name=CUSTOM_POSE_EXTRACTOR,
            meters_per_pixel=point_density,
            output=view_components,
        )
        self.poses = self.image_extractor.poses

        # We will perform preprocessing transforms on the data
        self.transforms = transforms

    def __len__(self):
        return len(self.image_extractor)

    def __getitem__(self, idx):
        sample = self.image_extractor[idx]
        # self.extractor.poses gives you the pose information
        # (both x y z and also quarternions)
        raw_semantic_output = sample["semantic"]

        output = {
            "rgb": sample["rgba"],
            "truth": raw_semantic_output.astype(int),
            "depth": sample["depth"],
        }

        if self.transforms:
            output["rgb"] = self.transforms(output["rgb"])
            output["truth"] = self.transforms(output["truth"]).squeeze(0)
            output["depth"] = self.transforms(output["depth"]).squeeze(0)

        return output
