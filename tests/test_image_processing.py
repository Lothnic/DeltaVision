import cv2
import numpy as np
import pytest
from src.image_processing import (
    align_images,
    load_image,
    resize_image,
    to_grayscale,
)


@pytest.fixture
def image1() -> np.ndarray:
    return load_image("data/image1.png")


@pytest.fixture
def image2() -> np.ndarray:
    return load_image("data/image2.png")


def test_load_image(image1: np.ndarray):
    assert image1 is not None
    assert isinstance(image1, np.ndarray)


def test_resize_image(image1: np.ndarray):
    resized_image = resize_image(image1, (100, 100))
    assert resized_image.shape == (100, 100, 3)


def test_to_grayscale(image1: np.ndarray):
    grayscale_image = to_grayscale(image1)
    assert len(grayscale_image.shape) == 2


def test_align_images(image1: np.ndarray, image2: np.ndarray):
    aligned_image1, aligned_image2 = align_images(image1, image2)
    assert aligned_image1.shape == image1.shape
    assert aligned_image2.shape == image2.shape