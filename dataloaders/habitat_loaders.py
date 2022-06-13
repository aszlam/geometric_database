from typing import List, Tuple, Union

import habitat_sim
import numpy as np
import tqdm
import quaternion
from habitat_sim.utils.common import quat_rotate_vector
from habitat_sim.utils.data import ImageExtractor
from torch.utils.data import Dataset
from torchvision import transforms
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
            "camera_direction": quaternion.as_float_array(camera_direction),
            "scene_name": scene_fp,
        }

        if self.transforms:
            output["rgb"] = self.transforms(output["rgb"])
            output["truth"] = self.transforms(output["truth"]).squeeze(0)
            output["depth"] = self.transforms(output["depth"]).squeeze(0)

        return output


class HabitatLocationDataset(Dataset):
    """
    A dataset that helps us load data from Habitat datasets.

    This custom dataset gives us a world (x, y, z) coordinate alongside a semantic tag that
    exists in that point.

    The algorithm to determine what exists at any point works in a somewhat unintelligent way
    which takes a view, marches the camera axis ray until it hits something, labels that obj,
    and labels everything between camera and that obj as "empty"/"air"/etc.

    Parameters:
    habitat_view_dataset: a view dataset constructed already that we can iterate over and
    find object semantic ids as well as their positions.
    """

    def __init__(
        self, habitat_view_ds: HabitatViewDataset, object_extraction_depth: float = 0.1
    ):
        self.habitat_view_dataset = habitat_view_ds
        self.image_extractor = habitat_view_ds.image_extractor
        self.poses = self.image_extractor.poses

        self.coordinates = []
        self.semantic_label = []

        self._extract_dataset()

    def _extract_dataset(self):
        # Itereate over the view dataset to extract all possible object tags.
        for idx in tqdm.trange(len(self.habitat_view_dataset)):
            data_dict = self.habitat_view_dataset[idx]
            camera_pos, camera_dir = (
                data_dict["camera_pos"],
                data_dict["camera_direction"],
            )
            depth_map = data_dict["depth"]
            frame_size = depth_map.shape[0]  # Assuming depth map is a square image.
            # Now, find out how far the center of the image is from depth map.
            direction_vector = quat_rotate_vector(
                q=quaternion.from_float_array(camera_dir), v=habitat_sim.geo.FRONT
            )
            distance = depth_map[frame_size // 2, frame_size // 2].item()
            xyz_loc = camera_pos + (direction_vector * distance) / np.linalg.norm(
                direction_vector
            )
            self.coordinates.append(xyz_loc)
            self.semantic_label.append(
                data_dict["truth"][frame_size // 2, frame_size // 2]
            )

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx):
        return {"xyz": self.coordinates[idx], "label": self.semantic_label[idx]}
