import cv2
import numpy as np


def create_heatmap(diff: np.ndarray) -> np.ndarray:
    """Creates a heatmap from a difference map.

    Args:
        diff: The difference map.

    Returns:
        The heatmap image.
    """
    heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_HOT)
    return heatmap


def save_image(image: np.ndarray, path: str):
    """Saves an image to a given path.

    Args:
        image: The image to save.
        path: The path to save the image to.
    """
    cv2.imwrite(path, image)