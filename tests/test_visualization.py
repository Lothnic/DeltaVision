
import numpy as np
import pytest
from src.visualization import create_heatmap, save_image


@pytest.fixture
def diff() -> np.ndarray:
    return np.random.randint(0, 255, (100, 100), dtype=np.uint8)


def test_create_heatmap(diff: np.ndarray):
    heatmap = create_heatmap(diff)
    assert heatmap is not None
    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (100, 100, 3)


def test_save_image(diff: np.ndarray):
    save_image(diff, "test.png")
    import os

    assert os.path.exists("test.png")
    os.remove("test.png")
