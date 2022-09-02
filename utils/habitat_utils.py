from typing import List, Tuple
import einops
from quaternion import quaternion, as_rotation_matrix
import numpy as np
import habitat_sim.registry as registry

from habitat_sim.utils.data import PoseExtractor
from typing import Callable, List, Union, Optional

import habitat_sim
from habitat_sim import bindings as hsim
from habitat_sim import registry as registry
from habitat_sim.agent.agent import AgentConfiguration, AgentState
from habitat_sim.utils.data.data_structures import ExtractorLRUCache
from habitat_sim.utils.data.pose_extractor import PoseExtractor, TopdownView
import logging

import torch


CUSTOM_POSE_EXTRACTOR = "3d_pose_extract"


def custom_pose_extractor_factory(grid_subdivision_size=50, height_grids=0):
    @registry.register_pose_extractor(name=CUSTOM_POSE_EXTRACTOR)
    class PoseExtractor3d(PoseExtractor):
        def extract_poses(self, view, fp):
            height, width = view.shape
            dist = min(height, width) // grid_subdivision_size
            if dist == 0:
                logging.warn(
                    "Too fine of a subdivision, grid dimensions: {height} x {width}".format(
                        height=height, width=width
                    )
                )

            # Create a grid of camera positions
            n_gridpoints_width, n_gridpoints_height = (
                width // dist - 1,
                height // dist - 1,
            )
            logging.warn(f"{dist} {n_gridpoints_height} {n_gridpoints_width}")

            # Exclude camera positions at invalid positions
            # and find all the valid positions for our camera.
            gridpoints = []
            for h in range(n_gridpoints_height):
                for w in range(n_gridpoints_width):
                    point = (dist + h * dist, dist + w * dist)
                    if self._valid_point(*point, view):
                        gridpoints.append(point)

            # Find the closest point of the target class to each gridpoint
            poses = []
            for point in gridpoints:
                point_label_pairs = self._panorama_extraction(point, view, dist)
                poses.extend(
                    [(point, point_, fp) for point_, label in point_label_pairs]
                )

            # Returns poses in the coordinate system of the topdown view
            return poses

        def _panorama_extraction(
            self, point: Tuple[int, int], view: np.ndarray, dist: int
        ) -> List[Tuple[Tuple[int, int], float]]:
            in_bounds_of_topdown_view = lambda row, col: 0 <= row < len(
                view
            ) and 0 <= col < len(view[0])
            point_label_pairs = []
            r, c = point
            assert dist >= 2, "Neighbors are overlapping with original points."
            neighbor_dist = dist // 2
            neighbors = [
                (r - neighbor_dist, c),
                (r, c - neighbor_dist),
                (r, c + neighbor_dist),
                # (r + neighbor_dist, c), # Exclude the pose that is in the opposite direction of habitat_sim.geo.FRONT, causes the quaternion computation to mess up
                (r - neighbor_dist, c - neighbor_dist),
                (r - neighbor_dist, c + neighbor_dist),
                (r + neighbor_dist, c - neighbor_dist),
                (r + neighbor_dist, c + neighbor_dist),
            ]

            for n in neighbors:
                # Only add the neighbor point if it is navigable. This prevents camera poses that
                # are just really close-up photos of some object
                if in_bounds_of_topdown_view(*n) and self._valid_point(*n, view):
                    point_label_pairs.append((n, 0.0))

            return point_label_pairs

        def _convert_to_scene_coordinate_system(
            self,
            poses: List[Tuple[Tuple[int, int], Tuple[int, int], str]],
            ref_point: Tuple[np.float32, np.float32, np.float32],
        ) -> List[Tuple[Tuple[int, int], quaternion, str]]:
            # Convert from topdown map coordinate system to that of the scene
            startw, starty, starth = ref_point
            new_poses = []
            height_low, height_high = -height_grids // 2, height_grids // 2 + 1
            for _, pose in enumerate(poses):
                camera_height = 0
                for target_height in range(height_low, height_high):
                    pos, cpi, filepath = pose
                    r1, c1 = pos
                    r2, c2 = cpi
                    new_pos = np.array(
                        [
                            startw + c1 * self.meters_per_pixel,
                            starty + camera_height * self.meters_per_pixel,
                            starth + r1 * self.meters_per_pixel,
                        ]
                    )
                    new_cpi = np.array(
                        [
                            startw + c2 * self.meters_per_pixel,
                            starty + target_height * self.meters_per_pixel,
                            starth + r2 * self.meters_per_pixel,
                        ]
                    )
                    cam_normal = new_cpi - new_pos
                    new_rot = self._compute_quat(cam_normal)
                    if np.isnan(new_rot).any():
                        # Sometimes the _compute_quat function messes up, so we skip over
                        # such coordinates.
                        logging.warn(
                            "Quarternion conversion failed for coordinates:\n"
                            f"Camera: {new_pos}, target: {new_cpi}"
                        )
                        raise ValueError("Wrong dimensions")
                    new_pos_t: Tuple[int, int] = tuple(new_pos)  # type: ignore[assignment]
                    new_poses.append((new_pos_t, new_rot, filepath))

            logging.debug("Camera poses found: {}".format(len(new_poses)))

            return new_poses


def local_to_global_xyz(
    local_positions: torch.Tensor,
    camera_position: torch.Tensor,
    camera_direction_matrix: torch.Tensor,
):
    # Assume everything exists on matrices that are batched up, and we want to operate on them.
    batch_size = local_positions.size(0)
    camera_matrix = torch.zeros(batch_size, 4, 4)
    camera_matrix[:, 0:3, 0:3] = camera_direction_matrix
    camera_matrix[:, 0:3, -1] = camera_position
    camera_matrix[:, -1, -1] = 1

    local_positions_one = torch.concat(
        (local_positions, torch.ones(local_positions.shape[:-1] + (1,))), dim=-1
    )

    rearranged_local_xyz = einops.rearrange(local_positions_one, "b n four -> b four n")
    global_positions = torch.bmm(local_positions_one, rearranged_local_xyz)
    return einops.rearrange(global_positions, "b four n -> b n four")[:, :, :-3]


def depth_and_camera_to_local_xyz(
    image_sizes: Tuple[int, int],
    depth_map: np.ndarray,
):
    # Adapted from https://aihabitat.org/docs/habitat-lab/view-transform-warp.html
    image_size_x, image_size_y = image_sizes
    xs, ys = np.meshgrid(
        np.linspace(-1, 1, image_size_x), np.linspace(1, -1, image_size_y)
    )
    xs = xs.reshape(1, image_size_x, image_size_y)
    ys = ys.reshape(1, image_size_x, image_size_y)
    z = depth_map.reshape(1, image_size_x, image_size_y)
    xyz = np.vstack((xs * z, ys * z, -z))
    return xyz


def depth_and_camera_to_global_xyz(
    local_xyz: np.ndarray,
    image_sizes: Tuple[int, int],
    camera_position: np.ndarray,
    camera_direction: quaternion,
) -> np.ndarray:
    # Adapted from https://aihabitat.org/docs/habitat-lab/view-transform-warp.html
    image_size_x, image_size_y = image_sizes
    xyz_one = np.vstack((local_xyz, np.ones((1, image_size_x, image_size_y))))
    xyz_one = xyz_one.reshape(4, -1)

    # Now create the camera-to-world matrix.
    T_world_camera = np.eye(4)
    T_world_camera[0:3, 3] = camera_position
    camera_rotation = as_rotation_matrix(camera_direction)
    T_world_camera[0:3, 0:3] = camera_rotation

    xyz_world = np.matmul(T_world_camera, xyz_one)
    xyz_world = xyz_world[:3, :]
    return xyz_world.T


def make_pose_extractor(name: str) -> Callable[..., PoseExtractor]:
    r"""Constructs a pose_extractor using the given name and keyword arguments

    :param name: The name of the pose_extractor in the `habitat_sim.registry`
    :param kwargs: The keyword arguments to be passed to the constructor of the pose extractor
    """

    model = registry.get_pose_extractor(name)
    assert model is not None, "Could not find a pose extractor for name '{}'".format(
        name
    )

    return model


class ImageExtractor:
    r"""Main class that extracts data by creating a simulator and generating a topdown map from which to
    iteratively generate image data.

    :property scene_filepaths: The location of the scene mesh or directory containing the meshes given to the simulator
    :property cur_fp: The current scene filepath. This is only relevant when the extractor is operating on multiple scenes
    :property labels: class labels of things to gather images of (currently not used)
    :property img_size: Tuple of image output dimensions (height, width)
    :property cfg: configuration for simulator of type SimulatorConfiguration
    :property sim: Simulator object
    :property meters_per_pixel: Resolution of topdown map. 0.1 means each pixel in the topdown map
        represents 0.1 x 0.1 meters in the coordinate system of the pathfinder

    :property tdv_fp_ref_triples: List of tuples containing (TopdownView Object, scene_filepath, reference point)
        information for each scene. Each scene requires:
            TopdownView: To extract poses
            scene_filepath: The file path to the mesh file. Necessary for scene switches.
            reference point: A reference point from the coordinate system of the scene. Necessary for specifying poses
                in the scene's coordinate system.

    :property pose_extractor_name: name of the pose_extractor in habitat.registry
    :property poses: list of camera poses gathered from pose_extractor
    :property train: A subset of poses used for training
    :property test: A subset of poses used for testing
    :property mode: Mode that determines which poses to use, train poses, test poses, or all (full) poses
    :property instance_id_to_name: Maps instance_ids in the scene to their english names. E.g. 31 -> 'wall'
    :property cache: LRU cache used to reduce the number of scene switches when extracting data. Can be disabled
        with the 'use_caching' variable.

    :property output: list of output names that the user wants e.g. ['rgba', 'depth']
    """

    def __init__(
        self,
        scene_filepath: Union[str, List[str]],
        labels: List[float] = None,
        img_size: tuple = (512, 512),
        output: List[str] = None,
        pose_extractor_name: str = "closest_point_extractor",
        scene_dataset_config_file: Optional[str] = None,
        sim=None,
        shuffle: bool = True,
        split: tuple = (70, 30),
        use_caching: bool = True,
        meters_per_pixel: float = 0.1,
    ):
        if labels is None:
            labels = [0.0]
        if output is None:
            output = ["rgba"]
        if sum(split) != 100:
            raise Exception("Train/test split must sum to 100.")

        self.scene_filepaths = None
        self.cur_fp = None
        if isinstance(scene_filepath, list):
            self.scene_filepaths = scene_filepath
        else:
            self.scene_filepaths = [scene_filepath]
            self.cur_fp = scene_filepath

        self.labels = set(labels)
        self.img_size = img_size
        self._scene_dataset_cfg = scene_dataset_config_file
        self.cfg = self._config_sim(
            self.scene_filepaths[0], self.img_size, self._scene_dataset_cfg
        )

        sim_provided = sim is not None
        if not sim_provided:
            sim = habitat_sim.Simulator(self.cfg)
        else:
            # If a sim is provided we have to make a new cfg
            self.cfg = self._config_sim(
                sim.config.sim_cfg.scene_id, img_size, self._scene_dataset_cfg
            )
            sim.reconfigure(self.cfg)

        self.sim = sim
        self.meters_per_pixel = meters_per_pixel
        if not sim_provided:
            self.tdv_fp_ref_triples = self._preprocessing(
                self.sim, self.scene_filepaths, self.meters_per_pixel
            )
        else:
            ref_point = self._get_pathfinder_reference_point(self.sim.pathfinder)
            self.tdv_fp_ref_triples = [
                (
                    TopdownView(self.sim, ref_point[1], meters_per_pixel),
                    self.sim.config.sim_cfg.scene_id,
                    ref_point,
                )
            ]

        args = (self.tdv_fp_ref_triples, self.meters_per_pixel)
        self.pose_extractor = make_pose_extractor(pose_extractor_name)(*args)
        self.poses = self.pose_extractor.extract_all_poses()

        if shuffle:
            np.random.shuffle(self.poses)

        self.train, self.test = self._handle_split(split, self.poses)
        self.mode = "full"
        self.mode_to_data = {
            "full": self.poses,
            "train": self.train,
            "test": self.test,
            None: self.poses,
        }
        self.instance_id_to_name = self._generate_label_map(self.sim.semantic_scene)
        self.out_name_to_sensor_name = {
            "rgba": "color_sensor",
            "depth": "depth_sensor",
            "semantic": "semantic_sensor",
            "instance": "semantic",
        }
        self.output = output
        self.use_caching = use_caching
        if self.use_caching:
            self.cache = ExtractorLRUCache()

    def __len__(self):
        return len(self.mode_to_data[self.mode])

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.start, idx.stop, idx.step
            if start is None:
                start = 0
            if stop is None:
                stop = len(self.mode_to_data[self.mode])
            if step is None:
                step = 1

            return [
                self.__getitem__(i)
                for i in range(start, stop, step)
                if i < len(self.mode_to_data[self.mode])
            ]

        mymode = self.mode.lower()
        if self.use_caching:
            cache_entry = (idx, mymode)
            if cache_entry in self.cache:
                return self.cache[cache_entry]

        poses = self.mode_to_data[mymode]
        pos, rot, fp = poses[idx]

        # Only switch scene if it is different from the last one accessed
        if fp != self.cur_fp:
            self.sim.reconfigure(
                self._config_sim(fp, self.img_size, self._scene_dataset_cfg)
            )
            self.cur_fp = fp

        new_state = AgentState()
        new_state.position = pos
        new_state.rotation = rot
        self.sim.agents[0].set_state(new_state)
        obs = self.sim.get_sensor_observations()
        sample = {
            out_name: obs[self.out_name_to_sensor_name[out_name]]
            for out_name in self.output
        }

        if self.use_caching:
            self.cache.add(cache_entry, sample)

        return sample

    def close(self) -> None:
        r"""Deletes the instance of the simulator. Necessary for instantiating a different ImageExtractor."""
        if self.sim is not None:
            self.sim.close()
            del self.sim
            self.sim = None

    def set_mode(self, mode: str) -> None:
        r"""Sets the mode of the simulator. This controls which poses to use; train, test, or all (full)"""
        mymode = mode.lower()
        if mymode not in ["full", "train", "test"]:
            raise Exception(
                f'Mode {mode} is not a valid mode for ImageExtractor. Please enter "full, train, or test"'
            )

        self.mode = mymode

    def get_semantic_class_names(self) -> List[str]:
        r"""Returns a list of english class names in the scene(s). E.g. ['wall', 'ceiling', 'chair']"""
        class_names = list(set(self.instance_id_to_name.values()))
        return class_names

    def _preprocessing(self, sim, scene_filepaths, meters_per_pixel):
        tdv_fp_ref = []
        for filepath in scene_filepaths:
            cfg = self._config_sim(filepath, self.img_size, self._scene_dataset_cfg)
            sim.reconfigure(cfg)
            ref_point = self._get_pathfinder_reference_point(sim.pathfinder)
            tdv = TopdownView(sim, ref_point[1], meters_per_pixel=meters_per_pixel)
            tdv_fp_ref.append((tdv, filepath, ref_point))

        return tdv_fp_ref

    def _handle_split(self, split, poses):
        train, test = split
        num_poses = len(self.poses)
        last_train_idx = int((train / 100) * num_poses)
        train_poses = poses[:last_train_idx]
        test_poses = poses[last_train_idx:]
        return train_poses, test_poses

    def _get_pathfinder_reference_point(self, pf):
        bound1, bound2 = pf.get_bounds()
        startw = min(bound1[0], bound2[0])
        starth = min(bound1[2], bound2[2])
        starty = pf.get_random_navigable_point()[
            1
        ]  # Can't think of a better way to get a valid y-axis value
        return (startw, starty, starth)  # width, y, height

    def _generate_label_map(self, scene, verbose=False):
        if verbose:
            print(
                f"House has {len(scene.levels)} levels, {len(scene.regions)} regions and {len(scene.objects)} objects"
            )
            print(f"House center:{scene.aabb.center} dims:{scene.aabb.sizes}")

        instance_id_to_name = {}
        for obj in scene.objects:
            if obj and obj.category:
                obj_id = int(obj.id.split("_")[-1])
                instance_id_to_name[obj_id] = obj.category.name()

        return instance_id_to_name

    def _config_sim(self, scene_filepath, img_size, scene_dataset_cfg=None):
        settings = {
            "width": img_size[1],  # Spatial resolution of the observations
            "height": img_size[0],
            "scene": scene_filepath,  # Scene path
            "default_agent": 0,
            "sensor_height": 1.5,  # Height of sensors in meters
            "color_sensor": True,  # RGBA sensor
            "semantic_sensor": True,  # Semantic sensor
            "depth_sensor": True,  # Depth sensor
            "instance_sensor": True,  # Instance sensor
            "silent": True,
        }

        sim_cfg = hsim.SimulatorConfiguration()
        sim_cfg.enable_physics = False
        sim_cfg.gpu_device_id = 0
        sim_cfg.scene_id = settings["scene"]
        sim_cfg.requires_textures = True
        if scene_dataset_cfg is not None:
            sim_cfg.scene_dataset_config_file = scene_dataset_cfg

        # define default sensor parameters (see src/esp/Sensor/Sensor.h)
        sensor_specs = []
        if settings["color_sensor"]:
            color_sensor_spec = habitat_sim.CameraSensorSpec()
            color_sensor_spec.uuid = "color_sensor"
            color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
            color_sensor_spec.resolution = [settings["height"], settings["width"]]
            color_sensor_spec.postition = [0.0, settings["sensor_height"], 0.0]
            color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            sensor_specs.append(color_sensor_spec)

        if settings["depth_sensor"]:
            depth_sensor_spec = habitat_sim.CameraSensorSpec()
            depth_sensor_spec.uuid = "depth_sensor"
            depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
            depth_sensor_spec.resolution = [settings["height"], settings["width"]]
            depth_sensor_spec.postition = [0.0, settings["sensor_height"], 0.0]
            depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            sensor_specs.append(depth_sensor_spec)

        if settings["semantic_sensor"]:
            semantic_sensor_spec = habitat_sim.CameraSensorSpec()
            semantic_sensor_spec.uuid = "semantic_sensor"
            semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
            semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
            semantic_sensor_spec.postition = [0.0, settings["sensor_height"], 0.0]
            semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            sensor_specs.append(semantic_sensor_spec)

        # create agent specifications
        agent_cfg = AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        return habitat_sim.Configuration(sim_cfg, [agent_cfg])
