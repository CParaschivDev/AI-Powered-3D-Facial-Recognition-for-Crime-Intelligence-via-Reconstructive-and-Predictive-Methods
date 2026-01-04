from functools import lru_cache
import os

from backend.models.landmarks.inference import LandmarkModel
from backend.models.reconstruction.reconstruct import ReconstructionModel
from backend.models.recognition.recognize import RecognitionModel
from backend.database.dependencies import SessionLocal
from backend.database.models import ModelVersion

# In a real application, these paths would likely come from a config file
# or be managed more robustly.
# Use best models first, fallback to regular models
LANDMARK_MODEL_PATH = "./logs/landmarks/landmark_model_best.pth"
LANDMARK_MODEL_FALLBACK = "./logs/landmarks/landmark_model.pth"
RECONSTRUCTION_MODEL_PATH = "./logs/reconstruction/reconstruction_model_best.pth"
RECONSTRUCTION_MODEL_FALLBACK = "./logs/reconstruction/reconstruction_model.pth"
RECOGNITION_MODEL_PATH = "./logs/recognition/recognition_model_best.pth"
RECOGNITION_MODEL_FALLBACK = "./logs/recognition/recognition_model.pth"

def get_active_model_path(model_name: str, fallback_path: str) -> str:
    """
    Queries the database for the active model path.
    Returns a fallback path if no active model is found.
    """
    from backend.models.loader import active_model_path
    try:
        return active_model_path(model_name, fallback_path)
    except Exception as e:
        print(f"Failed to get active model path: {e}")
        return fallback_path

@lru_cache(maxsize=1)
def get_landmark_model() -> LandmarkModel:
    """
    Provides a singleton instance of the LandmarkModel.
    The @lru_cache decorator ensures the model is loaded only once.
    """
    print("INFO: Loading landmark detection model...")
    # Try best model first, fallback to regular model
    if os.path.exists(LANDMARK_MODEL_PATH):
        print(f"INFO: Using best landmark model: {LANDMARK_MODEL_PATH}")
        return LandmarkModel(model_path=LANDMARK_MODEL_PATH)
    elif os.path.exists(LANDMARK_MODEL_FALLBACK):
        print(f"INFO: Using fallback landmark model: {LANDMARK_MODEL_FALLBACK}")
        return LandmarkModel(model_path=LANDMARK_MODEL_FALLBACK)
    else:
        print("WARN: No landmark model found, using placeholder")
        return LandmarkModel(model_path="")

@lru_cache(maxsize=1)
def get_reconstruction_model() -> ReconstructionModel:
    """
    Provides an instance of the ReconstructionModel based on the active version in the registry.
    """
    print("INFO: Loading 3D reconstruction model...")
    # Try best model first, then fallback
    if os.path.exists(RECONSTRUCTION_MODEL_PATH):
        model_path = RECONSTRUCTION_MODEL_PATH
        print(f"INFO: Using best reconstruction model: {model_path}")
    elif os.path.exists(RECONSTRUCTION_MODEL_FALLBACK):
        model_path = RECONSTRUCTION_MODEL_FALLBACK
        print(f"INFO: Using fallback reconstruction model: {model_path}")
    else:
        model_path = get_active_model_path('reconstruction', RECONSTRUCTION_MODEL_FALLBACK)
        print(f"INFO: Using reconstruction model from registry: {model_path}")
    return ReconstructionModel(model_path=model_path)

@lru_cache(maxsize=1)
def get_recognition_model() -> RecognitionModel:
    """
    Provides an instance of the RecognitionModel based on the active version in the registry.
    """
    print("INFO: Loading recognition model...")
    # Try best model first, then fallback
    if os.path.exists(RECOGNITION_MODEL_PATH):
        model_path = RECOGNITION_MODEL_PATH
        print(f"INFO: Using best recognition model: {model_path}")
    elif os.path.exists(RECOGNITION_MODEL_FALLBACK):
        model_path = RECOGNITION_MODEL_FALLBACK
        print(f"INFO: Using fallback recognition model: {model_path}")
    else:
        model_path = get_active_model_path('recognition', RECOGNITION_MODEL_FALLBACK)
        print(f"INFO: Using recognition model from registry: {model_path}")
    
    # RecognitionModel expects arcface_path and gcn_path (or None).
    # If `model_path` is a directory containing model files, pass those paths.
    # If it's a single checkpoint file (.pth), pass it as arcface_path so the
    # RecognitionModel will attempt to load it as the local checkpoint.
    if os.path.isdir(model_path):
        arcface_path = os.path.join(model_path, 'arcface_model.pth')
        gcn_path = os.path.join(model_path, 'gcn_model.pth')
        return RecognitionModel(arcface_path=arcface_path, gcn_path=gcn_path)
    else:
        return RecognitionModel(arcface_path=model_path)
