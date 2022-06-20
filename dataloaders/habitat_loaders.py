from typing import Iterable, List, Tuple, Union

import torch
import json
import numpy as np
import tqdm
import quaternion
from habitat_sim.utils.data import ImageExtractor
from habitat_sim.agent.agent import AgentState
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
        canonical_object_ids: bool = False,
        canonical_names_path: str = "dataloaders/object_maps.json",
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

        self._use_canonical_id = canonical_object_ids
        if canonical_object_ids:
            # We need to deduplicate multiples of same object with different IDs.
            self.instance_id_to_name = self.image_extractor.instance_id_to_name
            # Load the canonical name to ID mapper.
            self._name_to_id = {}
            for obj in json.load(open(canonical_names_path)):
                self._name_to_id[obj["name"]] = obj["id"]
            self._instance_id_to_canonical_id = {
                instance_id: self._name_to_id.get(name, 0)
                for (instance_id, name) in self.instance_id_to_name.items()
            }
            self.map_to_class_id = np.vectorize(
                lambda x: self._instance_id_to_canonical_id.get(x, 0)
            )

    def __len__(self):
        return len(self.image_extractor)

    def __getitem__(self, idx):
        sample = self.image_extractor[idx]
        # self.extractor.poses gives you the pose information
        # (both x y z and also quarternions)
        raw_semantic_output = sample["semantic"]
        truth_mask = (
            self.map_to_class_id(raw_semantic_output)
            if self._use_canonical_id
            else raw_semantic_output
        )
        pose_data = self.image_extractor.poses[idx]
        agent_pos, agent_direction, scene_fp = pose_data
        # Now use habitat sim to figure out sensor pos and direction.
        new_state = AgentState()
        new_state.position = agent_pos
        new_state.rotation = agent_direction
        self.image_extractor.sim.agents[0].set_state(new_state)
        full_state = self.image_extractor.sim.agents[0].get_state()
        # Extract camera state now.
        camera_state = full_state.sensor_states["depth_sensor"]
        camera_pos: np.ndarray = camera_state.position
        camera_direction: quaternion.quaternion = camera_state.rotation

        output = {
            "rgb": sample["rgba"],
            "truth": truth_mask.astype(int),
            "depth": sample["depth"],
            "camera_pos": np.array(camera_pos),
            "camera_direction": quaternion.as_float_array(camera_direction),
            "agent_pos": np.array(agent_pos),
            "agent_direction": quaternion.as_float_array(agent_direction),
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
        self.habitat_view_data = (
            habitat_view_ds.dataset
            if isinstance(habitat_view_ds, torch.utils.data.Subset)
            else habitat_view_ds
        )
        self.habitat_view_dataset = habitat_view_ds
        self.subsample_prob = subsample_prob
        self.image_extractor = self.habitat_view_data.image_extractor
        self.poses = self.image_extractor.poses

        self.coordinates = []
        self.semantic_label = []
        self.rgb_data = []

        self._extract_dataset()

    def _get_intrinsics(self):
        """
        Returns the instrinsic matrix of the camera
        :return: the intrinsic matrix (shape: :math:`[3, 3]`)
        :rtype: np.ndarray
        """
        image_size_x, image_size_y = self.habitat_view_data.image_size
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
    ):
        # Adapted from https://aihabitat.org/docs/habitat-lab/view-transform-warp.html
        image_size_x, image_size_y = self.habitat_view_data.image_size
        xs, ys = np.meshgrid(
            np.linspace(-1, 1, image_size_x), np.linspace(1, -1, image_size_y)
        )
        xs = xs.reshape(1, image_size_x, image_size_y)
        ys = ys.reshape(1, image_size_x, image_size_y)
        z = depth_map.numpy().reshape(1, image_size_x, image_size_y)
        xyz_one = np.vstack((xs * z, ys * z, -z, np.ones_like(z)))
        xyz_one = xyz_one.reshape(4, -1)

        # Now create the camera-to-world matrix.
        T_world_camera = np.eye(4)
        T_world_camera[0:3, 3] = camera_position
        camera_rotation = quaternion.as_rotation_matrix(camera_direction)
        T_world_camera[0:3, 0:3] = camera_rotation

        xyz_world = np.matmul(T_world_camera, xyz_one)
        xyz_world = xyz_world[:3, :]
        return xyz_world.T

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
                np.random.uniform(low=0.0, high=1.0, size=(frame_size * frame_size))
                <= self.subsample_prob
            )
            # Now, convert everything to their world coordinates.
            all_xyz = self._convert_rgbd_to_world_coordinates(
                camera_pos, quaternion.from_float_array(camera_dir), depth_map
            )
            all_rgb = data_dict["rgb"][:3, ...]
            self.coordinates.append(all_xyz[subsampled])
            self.semantic_label.append(
                einops.rearrange(data_dict["truth"], "w h -> (w h)")[subsampled]
            )
            self.rgb_data.append(
                einops.rearrange(all_rgb, "d w h -> (w h) d")[subsampled]
            )

        # Now combine everything in one array.
        self.coordinates = torch.from_numpy(np.concatenate(self.coordinates, axis=0))
        self.semantic_label = torch.from_numpy(
            np.concatenate(self.semantic_label, axis=0)
        )
        self.rgb_data = torch.from_numpy(np.concatenate(self.rgb_data, axis=0))

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx):
        return {
            "xyz": self.coordinates[idx],
            "label": self.semantic_label[idx],
            "rgb": self.rgb_data[idx],
        }
