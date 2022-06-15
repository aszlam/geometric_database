from typing import Iterable, List, Tuple, Union

import torch
import habitat_sim
import numpy as np
import tqdm
import quaternion
from habitat_sim.utils.common import quat_rotate_vector
from habitat_sim.utils.data import ImageExtractor
from torch.utils.data import Dataset
from torchvision import transforms
from utils.habitat_utils import CUSTOM_POSE_EXTRACTOR, custom_pose_extractor_factory
import einops


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
        habitat_scenes: Union[str, Iterable[str]],
        view_components: List[str] = ["rgba", "depth", "semantic"],
        pose_extractor_grid_size: int = 50,
        height_levels: int = 5,
        image_size: Iterable[int] = (512, 512),
        transforms=transforms.Compose([transforms.ToTensor()]),
    ):
        # Sets the grid size and the height levels in the pose extractor
        custom_pose_extractor_factory(pose_extractor_grid_size, height_levels)
        self.habitat_scenes = (
            [habitat_scenes]
            if isinstance(habitat_scenes, str)
            else list(habitat_scenes)
        )
        assert len(image_size) == 2
        self.image_size = tuple(image_size)
        self.image_extractor = ImageExtractor(
            scene_filepath=self.habitat_scenes,
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
            "camera_pos": np.array(camera_pos),
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
    object_extraction_depth: along the ray between the camera pose and where it hits the wall
    every object_extarction_depth length, we add a new point to the x y z dataloader saying
    it is empty.
    """

    def __init__(
        self,
        habitat_view_ds: HabitatViewDataset,
        object_extraction_depth: float = 0.5,
        subsample_prob: float = 0.02,
    ):
        self.habitat_view_dataset = habitat_view_ds
        self.subsample_prob = subsample_prob
        self.image_extractor = habitat_view_ds.image_extractor
        self.poses = self.image_extractor.poses

        self.coordinates = []
        self.semantic_label = []

        self._extract_dataset()

    def _get_intrinsics(self):
        """
        Returns the instrinsic matrix of the camera
        :return: the intrinsic matrix (shape: :math:`[3, 3]`)
        :rtype: np.ndarray
        """
        image_size_x, image_size_y = self.habitat_view_dataset.image_size
        self.fx, self.fy, self.cx, self.cy = (
            image_size_x // 2,
            image_size_y // 2,
            image_size_x // 2,
            image_size_y // 2,
        )
        Itc = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
        return Itc

    def _convert_rgbd_to_world_coordinates(
        self,
        camera_position: np.ndarray,
        camera_direction: quaternion.quaternion,
        depth_map: np.ndarray,
        subsampled_flat: np.ndarray,
    ):
        # First, calculate the world coordinates assuming camera is centered at 0 facing forward
        image_size_x, image_size_y = self.habitat_view_dataset.image_size
        grid_uv = np.meshgrid(np.arange(image_size_x), np.arange(image_size_y))
        grid_uv = np.stack(grid_uv, axis=-1).astype(float)
        grid_uv = einops.rearrange(grid_uv, "x y d -> (x y) d")[subsampled_flat]
        # Assume depth view is perfect, so we don't need to do filtering as seen in
        # https://github.com/facebookresearch/fairo/blob/main/droidlet/lowlevel/locobot/remote/remote_locobot.py#L100-L139
        # Converted from
        # https://gist.github.com/cbaus/6e04f7fe5355f67a90e99d7be0563e88#file-convert-py
        xy_over_z = (
            einops.rearrange(np.array([self.cx, self.cy]), "(x d) -> x d", x=1)
            - grid_uv
        )
        xy_over_z /= einops.rearrange(np.array([self.fx, self.fy]), "(x d) -> x d", x=1)
        depth_subsampled = einops.rearrange(depth_map.numpy(), "x y -> (x y)")[
            subsampled_flat
        ]
        z = einops.repeat(depth_subsampled, "w -> w d", d=1) / np.sqrt(
            1 + np.sum((xy_over_z**2), axis=-1, keepdims=True)
        )
        # Negative sign since "Front" is -z direction in pinhole camera notation.
        xyz = -np.concatenate([xy_over_z * z, z], axis=-1)
        # Now, rotate and translate this to find true world coordinates.
        rotated_xyz = self._rotate_set_of_vectors_by_quat(camera_direction, xyz)
        translated_xyz = rotated_xyz + camera_position
        return translated_xyz

    @staticmethod
    def _rotate_set_of_vectors_by_quat(q: quaternion.quaternion, M: np.ndarray):
        # Assume M has shape ... x 3
        assert M.shape[-1] == 3
        return M @ quaternion.as_rotation_matrix(q).T

    def _extract_dataset(self):
        # Precompute the camera intrinsics from the view dataset.
        self._get_intrinsics()
        # Itereate over the view dataset to extract all possible object tags.
        for idx in tqdm.trange(len(self.habitat_view_dataset)):
            data_dict = self.habitat_view_dataset[idx]
            camera_pos, camera_dir = (
                data_dict["camera_pos"],
                data_dict["camera_direction"],
            )
            depth_map = data_dict["depth"]
            frame_size = depth_map.shape[0]  # Assuming depth map is a square image.
            # Only process a subsampled portion of the image.
            subsampled = (
                np.random.uniform(low=0.0, high=1.0, size=(frame_size, frame_size))
                <= self.subsample_prob
            )
            subsampled_flat = einops.rearrange(subsampled, "x y -> (x y)")
            # Now, convert everything to their world coordinates.
            all_xyz = self._convert_rgbd_to_world_coordinates(
                camera_pos,
                quaternion.from_float_array(camera_dir),
                depth_map,
                subsampled_flat,
            )
            self.coordinates.append(all_xyz)
            self.semantic_label.append(
                einops.rearrange(data_dict["truth"], "w h -> (w h)")[subsampled_flat]
            )

        # Now combine everything in one array.
        self.coordinates = torch.from_numpy(np.concatenate(self.coordinates, axis=0))
        self.semantic_label = torch.from_numpy(
            np.concatenate(self.semantic_label, axis=0)
        )

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx):
        return {"xyz": self.coordinates[idx], "label": self.semantic_label[idx]}
