from threading import local
from typing import List, Tuple
import einops
from quaternion import quaternion, as_rotation_matrix
import numpy as np
import habitat_sim.registry as registry

from habitat_sim.utils.data import PoseExtractor
from itertools import product
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
