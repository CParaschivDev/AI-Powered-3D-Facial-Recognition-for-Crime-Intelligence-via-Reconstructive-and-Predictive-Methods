import cv2
import numpy as np
import random

from .augmentations import CCTV_AUGMENTATIONS
from backend.utils.face_io import read_image

class SyntheticDataGenerator:
    """
    Generates synthetic CCTV-like data from clean input images.
    """
    def __init__(self, num_augmentations_range=(2, 5)):
        """
        Initializes the generator.

        Args:
            num_augmentations_range (tuple): A (min, max) tuple for the number of
                                             random augmentations to apply to each image.
        """
        self.augmentations = CCTV_AUGMENTATIONS
        self.num_augmentations_range = num_augmentations_range

    def generate(self, image_path: str) -> np.ndarray:
        """
        Applies a random sequence of augmentations to an image.

        Args:
            image_path: Path to the input image.

        Returns:
            The augmented image as a NumPy array.
        """
        image = read_image(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        num_to_apply = random.randint(*self.num_augmentations_range)
        augs_to_apply = random.sample(list(self.augmentations.values()), num_to_apply)

        for aug in augs_to_apply:
            image = aug(image)
        
        return image

    def save_image(self, image: np.ndarray, output_path: str):
        """Saves an image to the specified path."""
        cv2.imwrite(output_path, image)
