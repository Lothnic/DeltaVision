
import numpy as np
import pytest
from src.change_detection import compute_difference
from src.image_processing import load_image


@pytest.fixture
def image1() -> np.ndarray:
    return load_image("data/image1.png")


@pytest.fixture
def image2() -> np.ndarray:
    return load_image("data/image2.png")


def test_compute_difference(image1: np.ndarray, image2: np.ndarray):
    diff = compute_difference(image1, image2)
    assert diff is not None
    assert isinstance(diff, np.ndarray)
    assert diff.shape == (image1.shape[0], image1.shape[1])
