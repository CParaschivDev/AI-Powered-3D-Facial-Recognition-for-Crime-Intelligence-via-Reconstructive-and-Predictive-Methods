import io
import numpy as np
from PIL import Image
import cv2
from backend.database.dependencies import SessionLocal, get_db
import numpy as np

def read_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Reads an image from a byte stream and converts it to an OpenCV-compatible format (BGR).

    Args:
        image_bytes: The byte string of the image.

    Returns:
        A NumPy array representing the image in BGR format.
    """
    pil_image = Image.open(io.BytesIO(image_bytes))
    # Convert to RGB if it's not, then convert RGB to BGR for OpenCV
    return cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)

# Re-export get_db for backward compatibility
# Original function moved to backend.database.dependencies

def assess_blur(image: np.ndarray) -> float:
    """
    Assess blur using the variance of the Laplacian. Higher means sharper.
    Returns a float score (higher = less blurry).
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        score = float(lap.var())
        return score
    except Exception:
        return 0.0


def assess_illumination(image: np.ndarray) -> float:
    """
    Assess illumination by returning the mean brightness (0-255).
    """
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        return float(v.mean())
    except Exception:
        return 0.0


def assess_occlusion(image: np.ndarray) -> float:
    """
    Heuristic occlusion score: fraction of very dark pixels which may indicate occlusion.
    Returns value in [0,1] where higher means more occluded.
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dark = (gray < 30).sum()
        total = gray.size
        return float(dark) / float(total) if total else 0.0
    except Exception:
        return 0.0
