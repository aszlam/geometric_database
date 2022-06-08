from typing import List, Tuple, Union
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from habitat_sim.utils.data import ImageExtractor

import numpy as np
import matplotlib.pyplot as plt

from utils.habitat_utils import CUSTOM_POSE_EXTRACTOR, custom_pose_extractor_factory


class HabitatViewDataset(Dataset):
    """
    A dataset that helps us load data from Habitat datasets.

    This custom dataset gives us a POV (x, y, z) coordinate alongside a quaternion for the
    direction the camera is looking towards, and at the same time the RGB, depth, and
    semantic segmentation data from the camera view.

    Parameters:
    habitat_scenes: a list of strings specifiying the path to habitat scenes to load.
    view_components: components of view info that we load in the dataset.
    pose_extractor_grid_size: number of grid points we are dividing the scene into (width, height).
    height_levels: number of grids to divide height into.
    image_size: desired image size from the dataloader.
    transforms: torchvision transforms for the loaded image from the dataset.
    """

    def __init__(
        self,
        habitat_scenes: Union[str, List[str]],
        view_components: List[str] = ["rgba", "depth", "semantic"],
        pose_extractor_grid_size: int = 50,
        height_levels: int = 5,
        image_size: Tuple[int, int] = (512, 512),
        transforms=transforms.Compose([transforms.ToTensor()]),
    ):
        # Sets the grid size and the height levels in the pose extractor
        custom_pose_extractor_factory(pose_extractor_grid_size, height_levels)
        self.image_extractor = ImageExtractor(
            scene_filepath=habitat_scenes,
            pose_extractor_name=CUSTOM_POSE_EXTRACTOR,
            img_size=image_size,
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
        pose_data = self.image_extractor.poses[idx]
        camera_pos, camera_direction, scene_fp = pose_data

        output = {
            "rgb": sample["rgba"],
            "truth": raw_semantic_output.astype(int),
            "depth": sample["depth"],
            "camera_pos": camera_pos,
            "camera_direction": camera_direction,
            "scene_name": scene_fp,
        }

        if self.transforms:
            output["rgb"] = self.transforms(output["rgb"])
            output["truth"] = self.transforms(output["truth"]).squeeze(0)
            output["depth"] = self.transforms(output["depth"]).squeeze(0)

        return output
