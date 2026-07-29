"""
CUDA backend adapter for OneMap. Wraps cuonemap.OneMap to present the same interface as
the Python OneMap class in feature_map.py, so the two are interchangeable.

See https://github.com/finnBsch/CuOneMap for the backend itself, including the known
differences from the PyTorch implementation.
"""
import numpy as np
import torch

import cuonemap as _onemap_cuda


class CUDAOneMap:
    """Drop-in replacement for OneMap using the CUDA backend.

    Accepts the same constructor arguments and exposes the same properties/methods
    as the Python OneMap class, translating between the Python and CUDA interfaces
    internally.
    """

    def __init__(self,
                 feature_dim,
                 config,
                 dense_projection=None,
                 fusion_type=None,
                 map_device="cuda"):
        assert feature_dim == 768, (
            f"CUDA OneMap backend requires feature_dim=768 (CLIP), got {feature_dim}"
        )
        cuda_conf = _onemap_cuda.MappingConf()
        cuda_conf.n_cells = config.n_points
        cuda_conf.size = float(config.size)
        cuda_conf.agent_radius = config.agent_radius
        cuda_conf.blur_kernel_size = config.blur_kernel_size
        cuda_conf.obstacle_map_threshold = config.obstacle_map_threshold
        cuda_conf.fully_explored_threshold = config.fully_explored_threshold
        cuda_conf.checked_map_threshold = config.checked_map_threshold
        cuda_conf.depth_factor = config.depth_factor
        cuda_conf.gradient_factor = config.gradient_factor
        cuda_conf.optimal_object_distance = config.optimal_object_distance
        cuda_conf.optimal_object_factor = config.optimal_object_factor
        cuda_conf.obstacle_min = config.obstacle_min
        cuda_conf.obstacle_max = config.obstacle_max
        cuda_conf.filter_stairs = config.filter_stairs
        cuda_conf.floor_level = config.floor_level
        cuda_conf.floor_threshold = config.floor_threshold
        cuda_conf.use_cpu_storage = (map_device == "cpu")

        self._impl = _onemap_cuda.OneMap(cuda_conf)
        self.map_device = map_device
        self.feature_dim = feature_dim
        self.n_cells = config.n_points
        self.size = float(config.size)
        self.cell_size = self.size / self.n_cells
        center = self._impl.map_center_cells
        self._map_center_cells = torch.tensor(
            [center[0], center[1]], dtype=torch.int32
        ).to("cuda")

    @property
    def map_center_cells(self):
        return self._map_center_cells

    @property
    def camera_initialized(self):
        return self._impl.camera_initialized

    @property
    def fx(self):
        return self._impl.fx

    @property
    def fy(self):
        return self._impl.fy

    @property
    def cx(self):
        return self._impl.cx

    @property
    def cy(self):
        return self._impl.cy

    # --- Map properties ---

    @property
    def feature_map(self):
        return self._impl.feature_map

    @property
    def confidence_map(self):
        return self._impl.confidence_map

    @property
    def checked_conf_map(self):
        return self._impl.checked_conf_map

    @property
    def obstacle_map(self):
        return self._impl.obstacle_map

    @property
    def updated_mask(self):
        return self._impl.updated_mask

    @property
    def navigable_map(self):
        t = self._impl.navigable_map
        if isinstance(t, np.ndarray):
            return t
        return t.cpu().numpy() if hasattr(t, 'cpu') else np.asarray(t)

    @property
    def fully_explored_map(self):
        t = self._impl.fully_explored_map
        if isinstance(t, np.ndarray):
            return t
        return t.cpu().numpy() if hasattr(t, 'cpu') else np.asarray(t)

    @property
    def checked_map(self):
        t = self._impl.checked_map
        if isinstance(t, np.ndarray):
            return t
        return t.cpu().numpy() if hasattr(t, 'cpu') else np.asarray(t)

    @property
    def occluded_map(self):
        t = self._impl.occluded_map
        if isinstance(t, np.ndarray):
            return t
        return t.cpu().numpy() if hasattr(t, 'cpu') else np.asarray(t)

    # --- Methods ---

    def set_camera_matrix(self, camera_matrix):
        self._impl.set_camera_matrix(camera_matrix)

    def update(self, values, depth, tf_camera_to_episodic, artifical_obstacles=None):
        # values: the Python implementation is given (F, H, W); the backend wants (H, W, F)
        if len(values.shape) == 3 and values.shape[0] == self.feature_dim:
            values = values.permute(1, 2, 0).contiguous()
        if not values.is_cuda:
            values = values.to("cuda")

        # depth: numpy or CPU tensor -> float32 CUDA tensor
        if isinstance(depth, np.ndarray):
            depth = torch.from_numpy(depth.astype(np.float32)).to("cuda")
        elif isinstance(depth, torch.Tensor):
            if not depth.is_cuda:
                depth = depth.to(torch.float32).to("cuda")
        else:
            depth = torch.as_tensor(depth, dtype=torch.float32).to("cuda")

        # tf stays on the host: the binding takes a CPU array
        if isinstance(tf_camera_to_episodic, torch.Tensor):
            tf_camera_to_episodic = tf_camera_to_episodic.cpu().numpy()
        tf_camera_to_episodic = tf_camera_to_episodic.astype(np.float32)

        obstacles = []
        if artifical_obstacles:
            obstacles = [(float(o[0]), float(o[1])) for o in artifical_obstacles]

        self._impl.update(values, depth, tf_camera_to_episodic, obstacles)

    def reset(self):
        self._impl.reset()

    def reset_updated_mask(self):
        self._impl.reset_updated_mask()

    def reset_checked_map(self):
        self._impl.reset_checked_map()

    def metric_to_px(self, x, y):
        return self._impl.metric_to_px(x, y)

    def px_to_metric(self, px, py):
        return self._impl.px_to_metric(px, py)
