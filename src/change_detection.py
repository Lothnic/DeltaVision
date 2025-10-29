
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compute_difference(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """Computes the difference between two images.

    Args:
        image1: The first image.
        image2: The second image.

    Returns:
        The difference map.
    """
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute absolute difference
    diff = cv2.absdiff(gray1, gray2)

    return diff
