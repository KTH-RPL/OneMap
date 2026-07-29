from spock import spock

@spock
class MappingConf:
    n_points: int
    size: int
    agent_radius: float
    blur_kernel_size: float
    obstacle_map_threshold: float
    fully_explored_threshold: float
    checked_map_threshold: float

    depth_factor: float
    gradient_factor: float

    optimal_object_distance: float
    optimal_object_factor: float

    obstacle_min: float
    obstacle_max: float

    filter_stairs: bool
    floor_level: float
    floor_threshold: float

    use_cuda_backend: bool = False

