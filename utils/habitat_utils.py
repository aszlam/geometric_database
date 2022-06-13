from typing import List, Tuple
from quaternion import quaternion
import numpy as np
import habitat_sim.registry as registry

from habitat_sim.utils.data import ImageExtractor, PoseExtractor
from itertools import product
import logging


CUSTOM_POSE_EXTRACTOR = "3d_pose_extract"


def custom_pose_extractor_factory(grid_subdivision_size=50, height_grids=2):
    @registry.register_pose_extractor(name=CUSTOM_POSE_EXTRACTOR)
    class PoseExtractor3d(PoseExtractor):
        def extract_poses(self, view, fp):
            height, width = view.shape
            logging.debug(
                "Grid dimensions: {height} x {width}".format(height=height, width=width)
            )
            dist = min(height, width) // grid_subdivision_size

            # Create a grid of camera positions
            n_gridpoints_width, n_gridpoints_height = (
                width // dist - 1,
                height // dist - 1,
            )

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
                for camera_height, target_height in product(
                    range(height_low, height_high), range(height_low, height_high)
                ):
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
                        continue
                    new_pos_t: Tuple[int, int] = tuple(new_pos)  # type: ignore[assignment]
                    new_poses.append((new_pos_t, new_rot, filepath))

            logging.debug("Camera poses found: {}".format(len(new_poses)))

            return new_poses
