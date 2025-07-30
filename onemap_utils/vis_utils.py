# numpy
import numpy as np

# cv2
import cv2

# rerun
import rerun as rr


def log_map_rerun(map_, path, needs_orientation=False):
    """
    Applies the inferno colormap to the map and logs it to rerun at the given path
    :param map_: 2D array
    :param path: logging path
    :param needs_orientation:
    :return:
    """
    if needs_orientation:
        map_ = map_.transpose((1, 0))
        map_ = np.flip(map_, axis=0)
    map_ = monochannel_to_inferno_rgb(map_)
    rr.log(path, rr.Image(np.flip(map_, axis=-1)))


def publish_sim_map(sim_map, br, publisher):
    sim_map = sim_map.transpose((1, 0))
    sim_map = np.flip(sim_map, axis=0)
    sim_map = monochannel_to_inferno_rgb(sim_map)
    # upscale to 1000x1000
    sim_map = cv2.resize(sim_map, (1000, 1000))
    img_msg = br.cv2_to_imgmsg(sim_map, encoding="bgr8")
    publisher.publish(img_msg)

def monochannel_to_inferno_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a monochannel float32 image to an RGB representation using the Inferno
    colormap.

    Args:
        image (numpy.ndarray): The input monochannel float32 image.

    Returns:
        numpy.ndarray: The RGB image with Inferno colormap.
    """
    # Normalize the input image to the range [0, 1]
    min_val, max_val = np.min(image), np.max(image)
    peak_to_peak = max_val - min_val
    if peak_to_peak == 0:
        normalized_image = np.zeros_like(image)
    else:
        normalized_image = (image - min_val) / peak_to_peak

    # Apply the Inferno colormap
    inferno_colormap = cv2.applyColorMap((normalized_image * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

    return inferno_colormap
