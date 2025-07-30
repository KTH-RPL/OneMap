"""
This is the core mapping module, which contains the OneMap class.
"""
from transforms3d.derivations.angle_axes import point

from mapping import (precompute_gaussian_kernel_components,
                     precompute_gaussian_sum_els, gaussian_kernel_sum,
                     compute_gaussian_kernel_components,
                     detect_frontiers,
                     )
from config import MappingConf

from onemap_utils import ceildiv

import time

# enum
from enum import Enum

# NumPy
import numpy as np

# typing
from typing import Tuple, List, Optional

# rerun
import rerun as rr

# torch
import torch

# warnings
import warnings

# cv2
import cv2

# functools
from functools import wraps


def rotate_pcl(
        pointcloud: torch.Tensor,
        tf_camera_to_episodic: torch.Tensor,
) -> torch.Tensor:
    # TODO We might be interested in a complete 3d rotation if the camera is not perfectly horizontal
    rotation_matrix = tf_camera_to_episodic[:3, :3]

    yaw = torch.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    # print(yaw)
    r = torch.tensor([[torch.cos(yaw), -torch.sin(yaw)], [torch.sin(yaw), torch.cos(yaw)]], dtype=torch.float32).to("cuda")
    pointcloud[:, :2] = (r @ pointcloud[:, :2].T).T
    return pointcloud

def print_memory_stats(label):
    print(f"\n--- Memory Stats for {label} ---")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1e6:.2f} MB")
    print(f"Max Allocated: {torch.cuda.max_memory_allocated() / 1e6:.2f} MB")

class DenseProjectionType(Enum):
    INTERPOLATE = "interpolate"
    SUBSAMPLE = "subsample"


class FusionType(Enum):
    EMA = "EMA"
    SPATIAL = "Spatial"


class OneMap:
    feature_map: torch.Tensor  # map where first dimension is x direction, second dimension is y, and last direction is
    # feature_dim
    obstacle_map: torch.Tensor  # map where first dimension is x direction, second dimension is y, and last direction is
    # obstacle likelihood
    navigable_map: np.ndarray  # binary traversability map where first dimension is x direction, second dimension is y
    # navigable likelihood
    fully_explored_map: np.ndarray  # binary explored map where first dimension is x direction, second dimension is y
    checked_map: np.ndarray  # binary checked map where first dimension is x direction, second dimension is y,
    # can be reset
    confidence_map: torch.Tensor
    checked_conf_map: torch.Tensor
    updated_mask: torch.Tensor  # tracks which cells have been updated, for lazy similarity computation

    def __init__(self,
                 feature_dim: int,
                 config: MappingConf,
                 dense_projection: DenseProjectionType = DenseProjectionType.INTERPOLATE,
                 fusion_type: FusionType = FusionType.EMA,
                 map_device: str = "cuda",
                 ) -> None:
        """

        :param feature_dim: The dimension of the feature space
        :param n_cells: The number of cells in the x and y direction respectively
        :param size: The size of the map in meters
        :param dense_projection: The type of dense projection to use, must be one of DenseProjectionType
        :param fusion_type: The type of fusion to use, must be one of FusionType
        """
        assert isinstance(dense_projection,
                          DenseProjectionType), "Invalid dense_projection. It should be one of DenseProjection."
        assert isinstance(fusion_type, FusionType), "Invalid fusion_type. It should be one of FusionType."

        self.dense_projection = dense_projection
        self.fusion_type = fusion_type
        self.map_device = map_device

        self.n_cells = config.n_points
        self.map_center_cells = self.map_center_cells = torch.tensor([self.n_cells // 2, self.n_cells // 2],
                                                                     dtype=torch.int32).to("cuda")
        self.size = config.size
        self.cell_size = self.size / self.n_cells
        self.feature_dim = feature_dim
        self.feature_map = torch.zeros((self.n_cells, self.n_cells, feature_dim), dtype=torch.float32)
        self.feature_map = self.feature_map.to(self.map_device)

        self.obstacle_map = torch.zeros((self.n_cells, self.n_cells), dtype=torch.float32)
        self.agent_radius = config.agent_radius
        col_kernel_size = self.n_cells / self.size * self.agent_radius
        col_kernel_size = int(col_kernel_size) + (int(col_kernel_size) % 2 == 0)
        self.navigable_map = np.ones((self.n_cells, self.n_cells), dtype=bool)
        self.occluded_map = np.zeros((self.n_cells, self.n_cells), dtype=bool)
        self.navigable_kernel = np.ones((col_kernel_size, col_kernel_size), np.uint8)

        self.fully_explored_map = np.zeros((self.n_cells, self.n_cells), dtype=bool)
        self.checked_map = np.zeros((self.n_cells, self.n_cells), dtype=bool)

        self.confidence_map = torch.zeros((self.n_cells, self.n_cells), dtype=torch.float32)
        self.confidence_map = self.confidence_map.to(self.map_device)
        self.checked_conf_map = torch.zeros((self.n_cells, self.n_cells), dtype=torch.float32)
        self.checked_conf_map = self.checked_conf_map.to(self.map_device)

        self.updated_mask = torch.zeros((self.n_cells, self.n_cells), dtype=torch.bool).to(self.map_device)

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.camera_initialized = False
        self.agent_height_0 = None

        self.kernel_half = int(np.round(config.blur_kernel_size / self.cell_size))
        self.kernel_size = self.kernel_half * 2 + 1
        self.kernel_components_sum = precompute_gaussian_sum_els(self.kernel_size).to("cuda")
        self.kernel_components = precompute_gaussian_kernel_components(self.kernel_size).to("cuda")
        self.kernel_ids = torch.arange(-self.kernel_half, self.kernel_half + 1).to("cuda")
        self.kernel_ids_x, self.kernel_ids_y = torch.meshgrid(self.kernel_ids, self.kernel_ids)
        self.kernel_ids_x = self.kernel_ids_x.unsqueeze(0)
        self.kernel_ids_y = self.kernel_ids_y.unsqueeze(0)
        print("ValueMap initialized. The map contains {} cells, each storing {} features. The resulting"
              " size is {} Mb".format(self.n_cells ** 2, feature_dim, self.feature_map.element_size() *
                                      self.feature_map.nelement() / 1024 / 1024))

        self.obstacle_map_threshold = config.obstacle_map_threshold
        self.fully_explored_threshold = config.fully_explored_threshold
        self.checked_map_threshold = config.checked_map_threshold
        self.depth_factor = config.depth_factor
        self.gradient_factor = config.gradient_factor
        self.optimal_object_distance = config.optimal_object_distance
        self.optimal_object_factor = config.optimal_object_factor
        self.obstacle_min = config.obstacle_min
        self.obstacle_max = config.obstacle_max
        self.filter_stairs = config.filter_stairs
        self.floor_threshold = config.floor_threshold
        self.floor_level = config.floor_level

        self._iters = 0

    def reset(self):
        # Reset value map
        self.feature_map = torch.zeros((self.n_cells, self.n_cells, self.feature_dim), dtype=torch.float32).to(
            self.map_device)

        # Reset obstacle map
        self.obstacle_map = torch.zeros((self.n_cells, self.n_cells), dtype=torch.float32).to(self.map_device)

        # Reset navigable map
        self.navigable_map = np.ones((self.n_cells, self.n_cells), dtype=bool)
        self.occluded_map = np.zeros((self.n_cells, self.n_cells), dtype=bool)

        # Reset fully explored map
        self.fully_explored_map = np.zeros((self.n_cells, self.n_cells), dtype=bool)

        # Reset checked map
        self.checked_map = np.zeros((self.n_cells, self.n_cells), dtype=bool)

        # Reset confidence map
        self.confidence_map = torch.zeros((self.n_cells, self.n_cells), dtype=torch.float32).to(self.map_device)

        # Reset checked confidence map
        self.checked_conf_map = torch.zeros((self.n_cells, self.n_cells), dtype=torch.float32).to(self.map_device)

        # Reset updated mask
        self.updated_mask = torch.zeros((self.n_cells, self.n_cells), dtype=torch.bool).to(self.map_device)

        # Reset iteration counter
        self._iters = 0
        self.agent_height_0 = None

    def reset_updated_mask(self):
        self.updated_mask = torch.zeros((self.n_cells, self.n_cells), dtype=torch.bool).to(self.map_device)

    def reset_checked_map(self):
        self.checked_map = np.zeros((self.n_cells, self.n_cells), dtype=bool)
        self.checked_conf_map = torch.zeros((self.n_cells, self.n_cells), dtype=torch.float32)

    def set_camera_matrix(self,
                          camera_matrix: np.ndarray
                          ) -> None:
        """
        Sets the camera matrix for the map
        :param camera_matrix: 3x3 numpy array representing the camera matrix
        :return:
        """
        self.camera_initialized = True
        self.fx = camera_matrix[0, 0]
        self.fy = camera_matrix[1, 1]
        self.cx = camera_matrix[0, 2]
        self.cy = camera_matrix[1, 2]

    def update(self,
               values: torch.Tensor,
               depth: np.ndarray,
               tf_camera_to_episodic: np.ndarray,
               artifical_obstacles: Optional[List[Tuple[float]]] = None
               ) -> None:
        """
        Updates the map with values by projecting them into the map from depth
        :param values: torch tensor of values. Either a 3D array of shape (feature_dim, hf, wf)
                        or a 1D array of shape (feature_dim)
        :param depth:  numpy array of depth values of shape (h, w)
        :param tf_camera_to_episodic: 4x4 numpy array representing the transformation from camera to episodic
        """
        assert values.shape[0] == self.feature_dim
        if not self.camera_initialized:
            warnings.warn("Camera matrix must be set before updating the map")
            return
        if self.agent_height_0 is None:
            self.agent_height_0 = tf_camera_to_episodic[2, 3] / tf_camera_to_episodic[3, 3]
        if len(values.shape) == 1 or (values.shape[-1] == 1 and values.shape[-2] == 1):
            confidences_mapped, values_mapped = self.project_single(values, depth,
                                                                    tf_camera_to_episodic, self.fx, self.fy,
                                                                    self.cx, self.cy)
        elif len(values.shape) == 3:
            values = values.permute(1, 2, 0)  # feature_dim last for convenience
            (confidences_mapped, values_mapped,
             obstacle_mapped, obstcl_confidence_mapped) = self.project_dense(values, torch.Tensor(depth).to("cuda"),
                                                                             torch.tensor(tf_camera_to_episodic),
                                                                             self.fx, self.fy,
                                                                             self.cx, self.cy)
        else:
            raise Exception("Provided Value observation of unsupported format")
        self.fuse_maps(confidences_mapped, values_mapped, obstacle_mapped, obstcl_confidence_mapped, artifical_obstacles)

    def fuse_maps(self,
                  confidences_mapped: torch.Tensor,
                  values_mapped: torch.Tensor,
                  obstacle_mapped: torch.Tensor,
                  obstcl_confidence_mapped: torch.Tensor,
                  artifical_obstacles: Optional[List[Tuple[float]]] = None
                  ) -> None:
        """
        Fuses the mapped values into the value map using the confidence estimates and tracked confidences
        This function takes in sparse tensors of confidences and values, and fuses them into the map, only updating
        the cells that have been updated.
        :param confidences_mapped: torch: sparse COO tensor of confidences
        :param values_mapped: torchL sparse COO tensor of values
        :return:
        """
        if self.fusion_type == FusionType.EMA:
            indices = confidences_mapped.indices()
            indices_obstacle = obstacle_mapped.indices()
            confs_new = confidences_mapped.values().data.squeeze()
            confs_old = self.confidence_map[indices[0], indices[1]]

            confs_old_obs = self.confidence_map[indices_obstacle[0], indices_obstacle[1]]

            confidence_denominator = confs_new + confs_old
            weight_1 = torch.nan_to_num(confs_old / confidence_denominator)
            weight_2 = torch.nan_to_num(confs_new / confidence_denominator)

            self.updated_mask[indices[0], indices[1]] = True

            self.feature_map[indices[0], indices[1]] = self.feature_map[indices[0], indices[1]] * weight_1.unsqueeze(-1) + \
                                                       values_mapped.values().data * weight_2.unsqueeze(-1)

            self.confidence_map[indices[0], indices[1]] = confidence_denominator

            # we also need to update the checked confidence
            confs_old_checked = self.checked_conf_map[indices[0], indices[1]]
            confidence_denominator_checked = confs_new + confs_old_checked
            self.checked_conf_map[indices[0], indices[1]] = confidence_denominator_checked

            # Obstacle Map update
            confs_new = obstcl_confidence_mapped.values().data.squeeze()
            confidence_denominator = confs_new + confs_old_obs
            weight_1 = torch.nan_to_num(confs_old_obs / confidence_denominator)
            weight_2 = torch.nan_to_num(confs_new / confidence_denominator)

            self.obstacle_map[indices_obstacle[0], indices_obstacle[1]] = self.obstacle_map[
                                                                              indices_obstacle[0], indices_obstacle[
                                                                                  1]] * weight_1 + \
                                                                          obstacle_mapped.values().data.squeeze() * weight_2

            self.occluded_map = (self.obstacle_map > self.obstacle_map_threshold).cpu().numpy()
            if artifical_obstacles is not None:
                for obs in artifical_obstacles:
                    self.occluded_map[obs[0], obs[1]] = True
            self.navigable_map = 1 - cv2.dilate((self.occluded_map).astype(np.uint8),
                                                self.navigable_kernel, iterations=1).astype(bool)


            self.fully_explored_map = (np.nan_to_num(1.0 / self.confidence_map.cpu().numpy())
                                       < self.fully_explored_threshold)

            self.checked_map = (np.nan_to_num(1.0 / self.checked_conf_map.cpu().numpy())
                                < self.checked_map_threshold)

    @torch.no_grad()
    # @torch.compile
    def project_dense(self,
                      values: torch.Tensor,
                      depth: torch.Tensor,
                      tf_camera_to_episodic: torch.Tensor,
                      fx, fy, cx, cy
                      ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Projects the dense features into the map
        TODO We could get rid of sparse tensors entirely and instead use arrays of indices and values to reduce overhead
        :param values: torch tensor of values, shape (hf, wf, feature_dim)
        :param depth: torch tensor of depth values, shape (h, w)
        :param tf_camera_to_episodic:
        :param fx:
        :param fy:
        :param cx:
        :param cy:
        :return: (confidences_mapped, values_mapped, obstacle_mapped, obstcl_confidence_mapped), sparse COO tensor in map coordinates
        """
        # check if values is on cuda
        if not values.is_cuda:
            print("Warning: Provided value array is not on cuda, which it should be as an output of a model. Moving to "
                  "Cuda, which will slow things down.")
            values = values.to("cuda")
        if not depth.is_cuda:
            print(
                "Warning: Provided depth array is not on cuda, which it could be if is an output of a model. Moving to "
                "Cuda, which will slow things down.")
            depth = depth.to("cuda")

        if values.shape[0:2] == depth.shape[0:2]:
            # our values align with the depth pixels
            depth_aligned = depth
        else:
            # our values are to be considered "patch wise" where we need to project each patch, by averaging the
            # depth values within that patch
            if self.dense_projection == DenseProjectionType.SUBSAMPLE:
                nh = values.shape[0]
                nw = values.shape[1]
                h = depth.shape[0]
                w = depth.shape[1]
                # TODO: this is possibly inaccurate, the patch_size might not add up and introduce errors
                patch_size_h = ceildiv(h, nh)
                patch_size_w = ceildiv(w, nw)

                pad_h = patch_size_h * nh - h
                pad_w = patch_size_w * nw - w
                pad_h_before = pad_h // 2
                pad_h_after = pad_h - pad_h_before
                pad_w_before = pad_w // 2
                pad_w_after = pad_w - pad_w_before

                depth_padded = np.pad(depth, ((pad_h_before, pad_h_after), (pad_w_before, pad_w_after)))
                depth_aligned = depth_padded.reshape(nh, patch_size_h, nw, patch_size_w).mean(axis=(1, 3))
            elif self.dense_projection == DenseProjectionType.INTERPOLATE:
                values = torch.nn.functional.interpolate(values.permute(2, 0, 1).unsqueeze(0),
                                                         size=depth.shape,
                                                         mode='bilinear',
                                                         align_corners=False).squeeze(0).permute(1, 2, 0)
                depth_aligned = depth
            else:
                raise Exception("Unsupported Dense Projection Mode.")

        # TODO this will be wrong for sub-sampled as e.g. fx will be wrong
        depth_image_smoothed = depth_aligned

        # mask = depth_image_smoothed == float('inf')
        # depth_image_smoothed[mask] = depth_image_smoothed[~mask].max()
        kernel_size = 11
        pad = kernel_size // 2

        depth_image_smoothed = -torch.nn.functional.max_pool2d(-depth_image_smoothed.unsqueeze(0), kernel_size,
                                                               padding=pad,
                                                               stride=1).squeeze(0)
        # depth_image_smoothed = F.gaussian_blur(depth_image_smoothed, [31, 31], sigma=4.0)
        # TODO Gaussian Blur temporarily disabled
        dx = torch.gradient(depth_image_smoothed, dim=1)[0] / (fx / depth.shape[1])
        dy = torch.gradient(depth_image_smoothed, dim=0)[0] / (fy / depth.shape[0])
        gradient_magnitude = torch.sqrt(dx ** 2 + dy ** 2)
        gradient_magnitude = torch.nn.functional.max_pool2d(gradient_magnitude.unsqueeze(0), 11, stride=1,
                                                            padding=5).squeeze(0)
        scores = ((1 - torch.tanh(gradient_magnitude * self.gradient_factor)) *
                  torch.exp(-((self.optimal_object_distance - depth) / self.optimal_object_factor) ** 2 / 3.0))
        scores_aligned = scores.reshape(-1)

        projected_depth, hole_mask = self.project_depth_camera(depth_aligned, (depth.shape[0], depth.shape[1]), fx,
                                                    fy, cx, cy)

        rotated_pcl = rotate_pcl(projected_depth, tf_camera_to_episodic)
        cam_x, cam_y = tf_camera_to_episodic[:2, 3] / tf_camera_to_episodic[3, 3]
        rotated_pcl[:, :2] += torch.tensor([cam_x, cam_y], device='cuda')

        values_aligned = values.reshape((-1, values.shape[-1]))

        pcl_grid_ids = torch.floor(rotated_pcl[:, :2] / self.cell_size).to(torch.int32)
        pcl_grid_ids[:, 0] += self.map_center_cells[0]
        pcl_grid_ids[:, 1] += self.map_center_cells[1]

        # Filter valid updates
        mask = (depth_aligned.flatten() != float('inf')) & (depth_aligned.flatten() != 0) & (pcl_grid_ids[:, 0] >= self.kernel_half + 1) & (
                pcl_grid_ids[:, 0] < self.n_cells - self.kernel_half - 1) & (
                       pcl_grid_ids[:, 1] >= self.kernel_half + 1) & (
                       pcl_grid_ids[:, 1] < self.n_cells - self.kernel_half - 1)  # for value map
        if hole_mask.nelement() == 0:
            mask_obstacle = mask & (((rotated_pcl[:, 2]> self.obstacle_min) & (
                                         rotated_pcl[:, 2]  < self.obstacle_max)) )
        else:
            mask_obstacle = mask & (((rotated_pcl[:, 2] > self.obstacle_min) & (
                    rotated_pcl[:, 2] < self.obstacle_max)) | hole_mask)
        mask &= (scores_aligned > 1e-5)
        mask_obstacle_masked = mask_obstacle[mask]
        scores_masked = scores_aligned[mask]

        pcl_grid_ids_masked = pcl_grid_ids[mask].T
        values_to_add = values_aligned[mask] * scores_masked.unsqueeze(1)

        combined_data = torch.cat((
            values_to_add,
            mask_obstacle_masked.unsqueeze(1),
            torch.ones((values_to_add.shape[0], 1), dtype=torch.uint8, device="cuda"),
            scores_masked.unsqueeze(1)),
            dim=1)  # prepare to aggregate doubles (values pointing to the same grid cell)

        # define the map from unique ids to all ids
        pcl_grid_ids_masked_unique, pcl_mapping = pcl_grid_ids_masked.unique(dim=1, return_inverse=True)
        # coalesce the data
        coalesced_combined_data = torch.zeros((pcl_grid_ids_masked_unique.shape[1], combined_data.shape[-1]),
                                              dtype=torch.float32, device="cuda")
        coalesced_combined_data.index_add_(0, pcl_mapping, combined_data)

        # Extract the data
        data_dim = combined_data.shape[-1]
        obstacle_mapped = coalesced_combined_data[:, data_dim - 3]
        scores_mapped = coalesced_combined_data[:, data_dim - 1].unsqueeze(1)
        sums_per_cell = coalesced_combined_data[:, data_dim - 2].unsqueeze(1)
        new_map = coalesced_combined_data[:, :data_dim - 3]

        # Normalize (from sum to mean)
        new_map /= scores_mapped
        scores_mapped /= sums_per_cell
        obstcl_confidence_mapped = scores_mapped


        # Get all the ids that are affected by the kernel (depth noise blurring)
        ids = pcl_grid_ids_masked_unique
        all_ids_ = torch.zeros((2, ids.shape[1], self.kernel_size, self.kernel_size), device="cuda")
        all_ids_[0] = (ids[0].unsqueeze(-1).unsqueeze(-1) + self.kernel_ids_x)
        all_ids_[1] = (ids[1].unsqueeze(-1).unsqueeze(-1) + self.kernel_ids_y)
        all_ids, mapping = all_ids_.reshape(2, -1).unique(dim=1, return_inverse=True)

        # Compute the corresponding depths
        depths = ((all_ids - self.map_center_cells.unsqueeze(1)) * self.cell_size - torch.tensor([cam_x, cam_y],
                                                                                 dtype=torch.float32, device="cuda")
                  .unsqueeze(1))

        # And the depth noise
        depth_noise = torch.sqrt(torch.sum(depths ** 2, dim=0)) * self.depth_factor / self.cell_size


        # Compute the sum for each kernel centered around a grid cell
        kernel_sums = gaussian_kernel_sum(self.kernel_components_sum, depth_noise).unsqueeze(-1)  # all unique ids

        # remap the depths to all the id's to kernels centered around the original points in ids and
        # compute the sparse inverse kernel elements
        kernels = compute_gaussian_kernel_components(self.kernel_components, depth_noise[mapping].reshape(-1,
                                                                                  self.kernel_size, self.kernel_size))

        coalesced_map_data = torch.zeros((all_ids.shape[1], self.feature_dim), dtype=torch.float32, device="cuda")
        coalesced_scores = torch.zeros((all_ids.shape[1], 1), dtype=torch.float32, device="cuda")
        # Compute the blurred map and blurred scores
        coalesced_map_data.index_add_(0, mapping, (kernels.unsqueeze(-1) *
                                                   new_map.unsqueeze(1).unsqueeze(1)).reshape(-1, self.feature_dim))
        coalesced_scores.index_add_(0, mapping, (kernels * scores_mapped.unsqueeze(1)).reshape(-1, 1))

        # Free up memory to avoid OOM
        torch.cuda.empty_cache()

        # Normalize the map and scores
        coalesced_map_data /= kernel_sums
        coalesced_scores /= kernel_sums

        # Compute the obstacle map
        obstacle_mapped[:] = (obstacle_mapped > 0).to(torch.float32)

        obstacle_mapped = torch.sparse_coo_tensor(pcl_grid_ids_masked_unique, obstacle_mapped.unsqueeze(1), (self.n_cells, self.n_cells, 1), is_coalesced=True).cpu()
        obstcl_confidence_mapped = torch.sparse_coo_tensor(pcl_grid_ids_masked_unique, obstcl_confidence_mapped, (self.n_cells, self.n_cells, 1), is_coalesced=True).cpu()
        # print("Updating with sparse matrix of size {}x{} with {} non-zero elements, resulting size is {} Mb".format(
        #     self.n_cells, self.n_cells, new_map.values().shape[0] * self.feature_dim,
        #                                 new_map.element_size() * new_map.values().shape[
        #                                     0] * self.feature_dim / 1024 / 1024))
        return torch.sparse_coo_tensor(all_ids, coalesced_scores, (self.n_cells, self.n_cells, 1), is_coalesced=True).cpu(), torch.sparse_coo_tensor(all_ids, coalesced_map_data, (self.n_cells, self.n_cells, self.feature_dim), is_coalesced=True).cpu(), obstacle_mapped.cpu(), obstcl_confidence_mapped.cpu()

    def project_single(self,
                       values: torch.Tensor,
                       depth: np.ndarray,
                       tf_camera_to_episodic,
                       fx, fy, cx, cy
                       ) -> (torch.Tensor, torch.Tensor):
        """
        Projects a single value observation into the map using a heuristic, similar to VLFM
        :param values:
        :param depth:
        :param tf_camera_to_episodic:
        :param fx:
        :param fy:
        :param cx:
        :param cy:
        :return:
        """
        projected_depth = self.project_depth_camera(depth, *(depth.shape[0:2]), fx, fy, cx, cy)
        # TODO needs to be implemented
        raise NotImplementedError

    def project_depth_camera(self,
                             depth: torch.Tensor,
                             camera_resolution: Tuple[int, int],
                             fx, fy, cx, cy
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Projects the depth into 3D pointcloud. Camera resolution is passed if the depth is subsampled,
        to match value array resolution.
        :param depth: torch Tensor of shape (h, w), not necessarily the same as camera resolution
        :param camera_resolution: tuple of original camera resolution to correct depth if necessary (w, h)
        :param fx:
        :param fy:
        :param cx:
        :param cy:
        :return: a point cloud of shape (h * w, 3), where x is depth (points into the image),
                                                          y is horizontal (points left),
                                                          z is vertical (points up)
        """
        # TODO are the "-1" necessary?
        x = torch.arange(0, depth.shape[1], device="cuda") * (camera_resolution[1] - 1) / (depth.shape[1] - 1)
        y = torch.arange(0, depth.shape[0], device="cuda") * (camera_resolution[0] - 1) / (depth.shape[0] - 1)
        xx, yy = torch.meshgrid(x, y, indexing="xy")
        xx = xx.flatten()
        yy = yy.flatten()
        zz = depth.flatten()
        x_world = (xx - cx) * zz / fx
        y_world = (yy - cy) * zz / fy
        z_world = zz
        point_cloud = torch.vstack((z_world, -x_world, -y_world)).T
        if self.filter_stairs:
            hole_mask = -y_world < self.floor_threshold # todo threshold parameter
            if hole_mask.any():
                scale_factor = self.floor_level / -y_world[hole_mask]
                point_cloud[hole_mask] *= scale_factor.unsqueeze(-1)
                return point_cloud, hole_mask

        return point_cloud, torch.empty((0,))

    def metric_to_px(self, x, y):
        epsilon = 1e-9  # Small value to account for floating-point imprecision

        return (
            int(x / self.cell_size + self.map_center_cells[0].item() + epsilon),
            int(y / self.cell_size + self.map_center_cells[1].item() + epsilon))

    def px_to_metric(self, px, py):
        return ((px - self.map_center_cells[0].item()) * self.cell_size,
                (py - self.map_center_cells[1].item()) * self.cell_size)


if __name__ == "__main__":
    rr.init("rerun_example_points3d", spawn=False)
    rr.connect("127.0.0.1:1234")
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)  # Set an up-axis
    rr.log(
        "world/xyz",
        rr.Arrows3D(
            vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ),
    )
    from detectron2.data.detection_utils import read_image

    map = OneMap(1)
    depth = read_image('test_images/depth2.png', format="BGR") * (-1) + 255
    depth2 = read_image('test_images/depth.png', format="BGR")

    fac = 10
    x = torch.arange(0, depth.shape[1] / fac, dtype=torch.float32)
    y = torch.arange(0, depth.shape[0], dtype=torch.float32)
    xx, yy, = torch.meshgrid(x, y)

    values = torch.sin(xx / (50.0 / fac)).T.unsqueeze(0)

    # values[:, :depth.shape[1]//2] = 1.0
    start = time.time()
    # map.update(torch.zeros((depth.shape[0], depth.shape[1], 3)), depth, np.eye(4))
    map.update(values, depth[:, :, 0], np.eye(4))
    map.update(-values, depth2[:, :, 0], np.eye(4))
    print(time.time() - start)
