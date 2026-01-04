import pytest
import numpy as np
import cv2
import os
from backend.utils.face_io import read_image

from synthetic_data_generator.generator import SyntheticDataGenerator

@pytest.fixture
def dummy_image():
    """Creates a dummy image file for testing."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img_path = "dummy_test_image.png"
    cv2.imwrite(img_path, img)
    yield img_path
    os.remove(img_path)

def test_generator_output(dummy_image):
    """Tests that the generator produces an image of the same size."""
    generator = SyntheticDataGenerator()
    original_image = read_image(dummy_image)
    synthetic_image = generator.generate(dummy_image)

    assert synthetic_image is not None
    assert synthetic_image.shape == original_image.shape
    # The image should have changed, so it shouldn't be all zeros anymore (most of the time)
    assert not np.all(synthetic_image == 0)
