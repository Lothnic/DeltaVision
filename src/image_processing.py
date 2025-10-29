import cv2
import numpy as np

def load_image(image_path: str) -> np.ndarray:
    """Loads an image from a given path.

    Args:
        image_path: The path to the image.

    Returns:
        The loaded image as a NumPy array.
    """
    return cv2.imread(image_path)

def resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resizes an image to a given size.

    Args:
        image: The image to resize.
        size: The target size.

    Returns:
        The resized image.
    """
    return cv2.resize(image, size)

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Converts an image to grayscale.

    Args:
        image: The image to convert.

    Returns:
        The grayscale image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def align_images(image1: np.ndarray, image2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Aligns two images using feature-based registration.

    Args:
        image1: The first image.
        image2: The second image.

    Returns:
        A tuple containing the aligned images.
    """
    # Convert images to grayscale
    gray1 = to_grayscale(image1)
    gray2 = to_grayscale(image2)

    # Detect SIFT features and compute descriptors.
    sift = cv2.SIFT_create(nfeatures=1000, contrastThreshold=0.03, edgeThreshold=15, sigma=1.6)
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # Match features.
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)

    if len(good) > 10:
        src_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(
            -1, 1, 2
        )

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
        if M is None:
            print("Homography could not be computed.")
            return image1, image2
        matchesMask = mask.ravel().tolist()

        # Warp image2 to match image1
        im2_reg = cv2.warpPerspective(image2, M, (image1.shape[1], image1.shape[0]))

        return image1, im2_reg
    else:
        print("Not enough matches are found - {}/{}".format(len(good), 10))
        matchesMask = None
        return image1, image2