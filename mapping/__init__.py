__all__ = [
    'OneMap',
    'create_onemap',
    'Navigator',
    'FusionType',
    'DenseProjectionType',
    'gaussian_kernel',
    'local_gaussian_blur',
    'precompute_gaussian_sum_els',
    'gaussian_kernel_sum',
    'precompute_gaussian_kernel_components',
    'compute_gaussian_kernel_components',
    'detect_frontiers',
    'get_frontier_midpoint',
    'cluster_high_similarity_regions','find_local_maxima', 'Cluster', 'NavGoal', 'Frontier',
    'watershed_clustering', 'gradient_based_clustering',
    'cluster_thermal_image'
]

import os
import warnings

from mapping.nav_goals.frontier import detect_frontiers, get_frontier_midpoint, Frontier

from mapping.nav_goals.navigation_goals import NavGoal

from mapping.nav_goals.clustering import cluster_high_similarity_regions, find_local_maxima, Cluster, watershed_clustering, gradient_based_clustering, cluster_thermal_image

from .varying_blur import gaussian_kernel, local_gaussian_blur, gaussian_kernel_sum, precompute_gaussian_sum_els, \
                            precompute_gaussian_kernel_components, compute_gaussian_kernel_components

from .feature_map import OneMap, FusionType, DenseProjectionType


def _cuda_available():
    try:
        import cuonemap
        return True
    except ImportError:
        return False


def create_onemap(feature_dim, config,
                  dense_projection=DenseProjectionType.INTERPOLATE,
                  fusion_type=FusionType.EMA,
                  map_device="cuda",
                  use_cuda=None):
    """Factory function to create a OneMap instance with the appropriate backend.

    Args:
        use_cuda: If None, auto-detect from config.use_cuda_backend or
                  ONEMAP_BACKEND=cuda env var. If True/False, force that backend.
    """
    if use_cuda is None:
        use_cuda = (
            getattr(config, 'use_cuda_backend', False)
            or os.environ.get("ONEMAP_BACKEND", "").lower() == "cuda"
        )

    if use_cuda:
        if not _cuda_available():
            warnings.warn(
                "CUDA backend requested but the 'cuonemap' package is not installed. "
                "Falling back to the PyTorch implementation. Install with: "
                "pip install git+https://github.com/finnBsch/CuOneMap.git"
            )
            use_cuda = False

    if use_cuda:
        from .cuda_feature_map import CUDAOneMap
        return CUDAOneMap(feature_dim, config, dense_projection,
                          fusion_type, map_device)

    return OneMap(feature_dim, config, dense_projection,
                  fusion_type, map_device)


# Navigator imported last since it depends on create_onemap from this module
from .navigator import Navigator
