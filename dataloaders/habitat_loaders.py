import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import json
import numpy as np
import tqdm
import quaternion

# from habitat_sim.utils.data import ImageExtractor
from habitat_sim.agent.agent import AgentState
from torch.utils.data import Dataset
from torchvision import transforms
from utils.habitat_utils import (
    CUSTOM_POSE_EXTRACTOR,
    custom_pose_extractor_factory,
    depth_and_camera_to_global_xyz,
    depth_and_camera_to_local_xyz,
    ImageExtractor,
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
        pose_extractor_grid_size: int = 10,
        height_levels: int = 0,
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
        if "hm3d" in habitat_scenes[0]:
            scene_cfg = "/checkpoint/notmahi/data/hm3d_semantic/data/versioned_data/hm3d-1.0/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
        else:
            scene_cfg = None
        assert len(image_size) == 2
        self.image_size = tuple(image_size)

        self.image_extractor = ImageExtractor(
            scene_filepath=self.habitat_scenes,
            pose_extractor_name=CUSTOM_POSE_EXTRACTOR,
            scene_dataset_config_file=scene_cfg,
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
            if not "hm3d" in self.habitat_scenes[0]:
                # Habitat scenes
                for obj in json.load(open(canonical_names_path)):
                    self._name_to_id[obj["name"]] = obj["id"]
                    self._id_to_name[obj["id"]] = obj["name"]
                # Explicitly set 0 as Other, since habitat has no other class.
                self._name_to_id["other"] = 0
                self._id_to_name[0] = "other"
            else:
                semantic_class_names = sorted(
                    self.image_extractor.get_semantic_class_names()
                )
                for idx, name in enumerate(semantic_class_names):
                    self._name_to_id[name] = idx
                    self._id_to_name[idx] = name
            self._instance_id_to_canonical_id = {
                instance_id: self._name_to_id.get(name, -1)
                for (instance_id, name) in self.instance_id_to_name.items()
            }
            # Now, just log the instances for which semantic label could not be found.
            ungrounded_instance_id_set = set()
            for instance_id, sem_id in self._instance_id_to_canonical_id.items():
                if sem_id == -1:
                    ungrounded_instance_id_set.add(instance_id)
            if len(ungrounded_instance_id_set) > 0:
                logging.info(
                    f"Ungrounded instance IDs: {str(ungrounded_instance_id_set)}"
                )
            self.map_to_class_id = np.vectorize(
                lambda x: self._instance_id_to_canonical_id.get(x, -1)
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
        if idx == 0:
            print(raw_semantic_output.max(), raw_semantic_output.min())
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
    selective_instance_segmentation: Only return instance segmentation for the top n images.
    """

    DIFFUSE_CLASSES = ("wall", "floor", "ceiling")

    def __init__(
        self,
        habitat_view_ds: HabitatViewDataset,
        subsample_prob: float = 0.2,
        selective_instance_segmentation: bool = True,
        selective_semantic_segmentation: bool = True,
        num_inst_segmented_images: int = 5,
        num_sem_segmented_images: int = 100,
        return_nonsegmented_images: bool = True,
        semantically_segment_instance_labeled: bool = False,
        use_only_valid_instance_ids: bool = False,
        valid_instance_ids: Optional[Iterable[int]] = None,
        exclude_diffuse_classes: bool = False,
        class_remapping: Optional[Dict[str, str]] = None,
    ):
        habitat_view_data = (
            habitat_view_ds.dataset
            if isinstance(habitat_view_ds, torch.utils.data.Subset)
            else habitat_view_ds
        )
        habitat_view_dataset = habitat_view_ds
        dataset_len = len(habitat_view_dataset)
        self._selective_segments = selective_instance_segmentation
        if selective_instance_segmentation:
            self._inst_segmented_images = self.get_best_instance_segment_images(
                habitat_view_ds, num_inst_segmented_images
            )
        else:
            self._inst_segmented_images = range(dataset_len)

        if selective_semantic_segmentation:
            self._sem_segmented_images = self.get_best_sem_segmented_images(
                habitat_view_ds, num_segmented_images=num_sem_segmented_images
            )
        else:
            self._sem_segmented_images = range(dataset_len)

        if use_only_valid_instance_ids:
            assert (
                valid_instance_ids is not None
            ), "Valid instance IDs must be assigned."
            self._valid_instance_ids = set(valid_instance_ids)
            logging.info(f"Set of valid instance ids: {str(self._valid_instance_ids)}")
            self.map_to_valid_instance_id = np.vectorize(
                lambda x: x if x in self._valid_instance_ids else -1
            )
        else:
            # Identity function.
            self.map_to_valid_instance_id = lambda x: x

        self.subsample_prob = subsample_prob
        image_extractor = habitat_view_data.image_extractor
        self.poses = image_extractor.poses
        self.instance_id_to_name = image_extractor.instance_id_to_name

        self._return_nonsegmented_images = return_nonsegmented_images
        self._semantically_segment_instance_labeled = (
            semantically_segment_instance_labeled
        )
        self.coordinates = []
        self.semantic_label = []
        self.instance_label = []
        self.rgb_data = []
        self.distance_data = []
        self.indices = []

        # Fix the instance id to name first
        if class_remapping:
            self._old_instance_id_to_name = self.instance_id_to_name
            self.instance_id_to_name = {
                x: class_remapping.get(name, name)
                for x, name in self._old_instance_id_to_name.items()
            }
            # Now, have to create an ID to name as well.
            self._id_to_name = {}
            self._name_to_id = {}
            set_of_semantic_classes = list(set(self.instance_id_to_name.values()))
            for id, name in enumerate(sorted(set_of_semantic_classes)):
                self._id_to_name[id] = name
                self._name_to_id[name] = id
            # No guarantees for which classes are labelled "other", though.
            self._old_sem_id_to_new_sem_id = {
                old_sem_id: self._name_to_id[class_remapping.get(name, name)]
                for old_sem_id, name in habitat_view_data._id_to_name.items()
            }
            self.map_to_right_class_id = np.vectorize(
                lambda x: self._old_sem_id_to_new_sem_id.get(x, -1)
            )
            self._remap_classes = True

        else:
            self._remap_classes = False
            self.map_to_right_class_id = lambda x: x
            self._id_to_name = habitat_view_data._id_to_name

        self._excluded_classes = set()
        if exclude_diffuse_classes:
            # Figure out the diffuse class semantic ids.
            for id, name in self._id_to_name.items():
                if name.lstrip().rstrip().lower() in self.DIFFUSE_CLASSES:
                    self._excluded_classes.add(id)

            self.mask_excluded_class = np.vectorize(
                lambda x: True if x in self._excluded_classes else False
            )
        else:
            self.mask_excluded_class = lambda x: np.zeros_like(x).astype(bool)

        # Precompute the camera intrinsics from the view dataset.
        self._get_intrinsics(habitat_view_data)
        self._extract_dataset(habitat_view_ds)

    def _get_intrinsics(self, habitat_view_data: HabitatViewDataset):
        """
        Returns the instrinsic matrix of the camera
        :return: the intrinsic matrix (shape: :math:`[3, 3]`)
        :rtype: np.ndarray
        """
        image_size_x, image_size_y = habitat_view_data.image_size
        self.fx, self.fy, self.cx, self.cy = (
            image_size_x // 2,
            image_size_y // 2,
            image_size_x // 2,
            image_size_y // 2,
        )
        Itc = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
        return Itc

    def _extract_dataset(self, habitat_view_dataset):
        # Itereate over the view dataset to extract all possible object tags.
        for idx in tqdm.trange(len(habitat_view_dataset)):
            # index needs to either be in instance segmented or sem segmented images.
            if (
                idx not in self._inst_segmented_images
                and idx not in self._sem_segmented_images
            ):
                continue
            data_dict = habitat_view_dataset[idx]
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
            semantic_data = self._get_semantic_labels_from_image(data_dict, subsampled)
            if self._semantically_segment_instance_labeled or (idx in self._sem_segmented_images):
                self.semantic_label.append(
                    semantic_data
                )
            else:
                self.semantic_label.append(
                    -1 * np.ones_like(semantic_data)
                )
            # Only selectively give instance segmentation.
            if idx in self._inst_segmented_images:
                valid_instance_labels = self.map_to_valid_instance_id(
                    einops.rearrange(
                        data_dict["instance_segmentation"], "w h -> (w h)"
                    )[subsampled]
                )
            else:
                # Fill it with -1s
                valid_instance_labels = np.ones_like(self.semantic_label[-1]) * -1
            # Also replace the diffuse classes, if necessary
            diffuse_class_mask = self.mask_excluded_class(self.semantic_label[-1])
            valid_instance_labels[diffuse_class_mask] = -1
            self.instance_label.append(valid_instance_labels)

            self.rgb_data.append(
                einops.rearrange(all_rgb, "w h d -> (w h) d")[subsampled]
            )
            self.indices.append(np.ones_like(self.semantic_label[-1]) * idx)
            self.distance_data.append(
                einops.rearrange(torch.ones_like(depth_map), "w h -> (w h)")[subsampled]
            )

        # Now combine everything in one array.
        self.coordinates = torch.from_numpy(
            np.concatenate(self.coordinates, axis=0)
        ).float()
        self.semantic_label = torch.from_numpy(
            np.concatenate(self.semantic_label, axis=0)
        ).long()
        self.instance_label = torch.from_numpy(
            np.concatenate(self.instance_label, axis=0)
        ).long()
        self.rgb_data = torch.from_numpy(np.concatenate(self.rgb_data, axis=0)).float()
        self.distance_data = torch.from_numpy(
            np.concatenate(self.distance_data, axis=0)
        ).float()
        self.indices = torch.from_numpy(np.concatenate(self.indices, axis=0)).long()

        # Now, figure out the maximum and minimum coordinates.
        self.max_coords, _ = torch.max(self.coordinates, dim=0)
        self.min_coords, _ = torch.min(self.coordinates, dim=0)

        del self.map_to_valid_instance_id
        del self.map_to_right_class_id
        del self.mask_excluded_class

    @property
    def valid_instance_ids(self) -> List[int]:
        return torch.unique(self.instance_label).cpu().numpy().tolist()

    def _exclude_diffuse_classes(self, instance_id: int, class_id: int) -> int:
        return -1 if (class_id in self._excluded_classes) else instance_id

    def _get_semantic_labels_from_image(self, current_image_data_dict, subsampled):
        return self.map_to_right_class_id(
            einops.rearrange(
                current_image_data_dict["semantic_segmentation"], "w h -> (w h)"
            )[subsampled]
        )

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

    def get_best_instance_segment_images(
        self, habitat_view_dataset, num_segmented_image=5
    ):
        best_set_so_far = set()
        chosen_images_so_far = set()
        num_chosen_images = num_segmented_image
        dataset = habitat_view_dataset
        unique_sets = [
            set(torch.unique(dataset[idx]["instance_segmentation"]).tolist())
            for idx in range(len(dataset))
        ]
        for _ in range(num_chosen_images):
            best_set_index = -1
            best_set_score = -1
            for idx in range(len(dataset)):
                if idx in chosen_images_so_far:
                    # Only consider new images.
                    continue
                current_new_set_under_consideration = unique_sets[idx]
                current_set_score = len(
                    current_new_set_under_consideration - best_set_so_far
                )
                if current_set_score > best_set_score:
                    best_set_score = current_set_score
                    best_set_index = idx

            chosen_images_so_far.add(best_set_index)
            best_set_so_far = best_set_so_far | unique_sets[best_set_index]

        chosen_images_so_far = list(chosen_images_so_far)
        return chosen_images_so_far

    def get_best_sem_segmented_images(
        self, habitat_view_dataset, num_segmented_images=50
    ):
        dataset = habitat_view_dataset
        # Using depth as a proxy for object diversity in a scene.
        num_objects_and_images = [
            (dataset[idx]["depth"].max() - dataset[idx]["depth"].min(), idx)
            for idx in range(len(dataset))
        ]
        sorted_num_object_and_img = sorted(
            num_objects_and_images, key=lambda x: x[0], reverse=True
        )
        return [x[1] for x in sorted_num_object_and_img[:num_segmented_images]]
