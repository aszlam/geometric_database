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
from utils.habitat_utils import (
    CUSTOM_POSE_EXTRACTOR,
    custom_pose_extractor_factory,
    depth_and_camera_to_global_xyz,
    depth_and_camera_to_local_xyz,
)
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
        canonical_object_ids: bool = True,
        canonical_names_path: str = "dataloaders/object_maps.json",
        use_cache: bool = True,
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
        self.id_to_name = self.image_extractor.instance_id_to_name

        # We will perform preprocessing transforms on the data
        self.transforms = transforms

        self._use_canonical_id = canonical_object_ids
        if canonical_object_ids:
            # We need to deduplicate multiples of same object with different IDs.
            self.instance_id_to_name = self.image_extractor.instance_id_to_name
            # Load the canonical name to ID mapper.
            self._name_to_id = {}
            self._id_to_name = {}
            for obj in json.load(open(canonical_names_path)):
                self._name_to_id[obj["name"]] = obj["id"]
                self._id_to_name[obj["id"]] = obj["name"]
            self._instance_id_to_canonical_id = {
                instance_id: self._name_to_id.get(name, 0)
                for (instance_id, name) in self.instance_id_to_name.items()
            }
            self.map_to_class_id = np.vectorize(
                lambda x: self._instance_id_to_canonical_id.get(x, 0)
            )

        self._cache = {}
        self._use_cache = use_cache

    def __len__(self):
        return len(self.image_extractor)

    def __getitem__(self, idx):
        if self._use_cache and (idx in self._cache):
            return self._cache[idx]
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
        depth_shape = sample["depth"].shape

        local_xyz = depth_and_camera_to_local_xyz(
            image_sizes=depth_shape,
            depth_map=sample["depth"],
        )

        output = {
            "rgb": sample["rgba"],
            "semantic_segmentation": truth_mask.astype(int),
            "instance_segmentation": raw_semantic_output.astype(int),
            "depth": sample["depth"],
            "camera_pos": np.array(camera_pos),
            "camera_direction": quaternion.as_float_array(camera_direction),
            "camera_direction_matrix": quaternion.as_rotation_matrix(camera_direction),
            "agent_pos": np.array(agent_pos),
            "agent_direction": quaternion.as_float_array(agent_direction),
            "xyz_position": depth_and_camera_to_global_xyz(
                local_xyz,
                image_sizes=depth_shape,
                camera_position=np.array(camera_pos),
                camera_direction=camera_direction,
            ),
            "local_xyz_position": local_xyz,
            "scene_name": scene_fp,
        }

        if self.transforms:
            output["rgb"] = einops.rearrange(
                self.transforms(output["rgb"]).float(), "... c h w -> ... h w c"
            )
            output["semantic_segmentation"] = self.transforms(
                output["semantic_segmentation"]
            ).squeeze(0)
            output["instance_segmentation"] = self.transforms(
                output["instance_segmentation"]
            ).squeeze(0)
            output["depth"] = self.transforms(output["depth"]).squeeze(0).float()
            output["xyz_position"] = (
                self.transforms(output["xyz_position"]).squeeze(0).float()
            )
            output["local_xyz_position"] = einops.rearrange(
                torch.from_numpy(output["local_xyz_position"]).squeeze(0).float(),
                "... c h w -> ... h w c",
            )

        output["xyz_position"] = einops.rearrange(
            output["xyz_position"],
            "(w h) d -> d w h",
            w=depth_shape[0],
        )
        if self._use_cache:
            self._cache[idx] = output
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
    every object_extarction_depth length, we add a new point to the x y z dataloader saying
    find object semantic ids as well as their positions.
    object_extraction_depth: along the ray between the camera pose and where it hits the wall
    it is empty.
    """

    def __init__(
        self,
        habitat_view_ds: HabitatViewDataset,
        object_extraction_depth: float = 0.5,
        subsample_prob: float = 0.2,
        selective_instance_segmentation: bool = True,
        num_segmented_images: int = 5,
        return_nonsegmented_images: bool = True,
    ):
        self.habitat_view_data = (
            habitat_view_ds.dataset
            if isinstance(habitat_view_ds, torch.utils.data.Subset)
            else habitat_view_ds
        )
        self.habitat_view_dataset = habitat_view_ds
        self._selective_segments = selective_instance_segmentation
        if selective_instance_segmentation:
            self._segmented_images = self.get_best_instance_segment_images(
                num_segmented_images
            )
        else:
            self._segmented_images = range(len(self.habitat_view_dataset))
        self.subsample_prob = subsample_prob
        self.image_extractor = self.habitat_view_data.image_extractor
        self.poses = self.image_extractor.poses

        self._return_nonsegmented_images = return_nonsegmented_images
        self.coordinates = []
        self.semantic_label = []
        self.instance_label = []
        self.rgb_data = []
        self.distance_data = []
        self.indices = []

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

    def _extract_dataset(self):
        # Precompute the camera intrinsics from the view dataset.
        self._get_intrinsics()
        # Itereate over the view dataset to extract all possible object tags.
        for idx in tqdm.trange(len(self.habitat_view_dataset)):
            # Only keep segmented images here
            if not self._return_nonsegmented_images:
                if idx not in self._segmented_images:
                    continue
            data_dict = self.habitat_view_dataset[idx]
            depth_map = data_dict["depth"]
            frame_size = depth_map.shape[0]  # Assuming depth map is a square image.
            # Only process a subsampled portion of the image.
            subsampled = (
                np.random.uniform(low=0.0, high=1.0, size=(frame_size * frame_size))
                <= self.subsample_prob
            )
            # Now, convert everything to their world coordinates.
            all_xyz: np.ndarray = data_dict["xyz_position"]
            all_rgb: np.ndarray = data_dict["rgb"][..., :3]
            self.coordinates.append(
                einops.rearrange(all_xyz, "d w h-> (w h) d")[subsampled]
            )
            # Right now, using the ground truth semantic segmentation.
            self.semantic_label.append(
                self._get_semantic_labels_from_image(data_dict, subsampled)
            )
            # Only selectively give instance segmentation.
            if idx in self._segmented_images:
                self.instance_label.append(
                    einops.rearrange(
                        data_dict["instance_segmentation"], "w h -> (w h)"
                    )[subsampled]
                )
            else:
                # Fill it with -1s
                self.instance_label.append(np.ones_like(self.semantic_label[-1]) * -1)
            self.rgb_data.append(
                einops.rearrange(all_rgb, "w h d -> (w h) d")[subsampled]
            )
            self.indices.append(np.ones_like(self.semantic_label[-1]) * idx)
            if not self._return_nonsegmented_images:
                self.distance_data.append(
                    einops.rearrange(torch.ones_like(depth_map) * 100, "w h -> (w h)")[
                        subsampled
                    ]
                )
            else:
                self.distance_data.append(
                    einops.rearrange(depth_map, "w h -> (w h)")[subsampled]
                )

        # Now combine everything in one array.
        self.coordinates = torch.from_numpy(np.concatenate(self.coordinates, axis=0))
        self.semantic_label = torch.from_numpy(
            np.concatenate(self.semantic_label, axis=0)
        )
        self.instance_label = torch.from_numpy(
            np.concatenate(self.instance_label, axis=0)
        )
        self.rgb_data = torch.from_numpy(np.concatenate(self.rgb_data, axis=0))
        self.distance_data = torch.from_numpy(
            np.concatenate(self.distance_data, axis=0)
        )
        self.indices = torch.from_numpy(np.concatenate(self.indices, axis=0))

    def _get_semantic_labels_from_image(self, current_image_data_dict, subsampled):
        return einops.rearrange(
            current_image_data_dict["semantic_segmentation"], "w h -> (w h)"
        )[subsampled]

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx):
        return {
            "xyz": self.coordinates[idx],
            "rgb": self.rgb_data[idx],
            "label": self.semantic_label[idx],
            "instance": self.instance_label[idx],
            "img_idx": self.indices[idx],
            "distance": self.distance_data[idx],
        }

    def get_best_instance_segment_images(self, num_segmented_image=5):
        best_set_so_far = set()
        chosen_images_so_far = set()
        num_chosen_images = num_segmented_image
        for _ in range(num_chosen_images):
            best_set_index = -1
            best_set_score = -1
            for idx, data in enumerate(self.habitat_view_dataset):
                current_new_set_under_consideration = set(
                    torch.unique(data["instance_segmentation"]).tolist()
                )
                current_set_score = len(
                    current_new_set_under_consideration - best_set_so_far
                )
                if current_set_score > best_set_score:
                    best_set_score = current_set_score
                    best_set_index = idx

            chosen_images_so_far.add(best_set_index)
            best_set_so_far = best_set_so_far | set(
                torch.unique(
                    self.habitat_view_dataset[best_set_index]["instance_segmentation"]
                ).tolist()
            )

        chosen_images_so_far = list(chosen_images_so_far)
        return chosen_images_so_far
