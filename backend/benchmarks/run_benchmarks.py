import os
import sys
import time
import tracemalloc
import cv2
import numpy as np
import pandas as pd
import torch
import logging

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from backend.utils.face_io import read_image

from backend.models.reconstruction.reconstruct import ReconstructionModel
from backend.models.recognition.recognize import RecognitionModel
from backend.utils.augmentation import apply_gaussian_blur, add_gaussian_noise

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'reports')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
SAMPLE_IMAGE_PATH = os.path.join(DATA_DIR, 'sample_face.jpg')
GROUND_TRUTH_IDENTITY = "person_01"

# --- Helper Functions ---

def create_dummy_image():
    """Creates a dummy image if the sample doesn't exist."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(SAMPLE_IMAGE_PATH):
        logger.warning(f"Sample image not found. Creating a dummy image at {SAMPLE_IMAGE_PATH}")
        dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        cv2.imwrite(SAMPLE_IMAGE_PATH, dummy_img)

def get_image_qualities(base_image):
    """Generates a dictionary of images with different quality degradations."""
    qualities = {
        "clean": base_image,
        "blur_low": apply_gaussian_blur(base_image, kernel_size=(5, 5)),
        "blur_high": apply_gaussian_blur(base_image, kernel_size=(15, 15)),
        "noise_low": add_gaussian_noise(base_image, sigma=25),
        "noise_high": add_gaussian_noise(base_image, sigma=75),
    }
    return qualities

def benchmark_reconstruction(model, image):
    """Benchmarks the reconstruction model."""
    start_time = time.perf_counter()
    # Landmarks not needed for placeholder model, pass None
    model.reconstruct(image, landmarks=None)
    end_time = time.perf_counter()
    return end_time - start_time

def benchmark_recognition(model, image, gt_embedding):
    """Benchmarks the recognition model and calculates mock accuracy."""
    start_time = time.perf_counter()
    query_embedding = model.extract_fused_embedding(image)
    end_time = time.perf_counter()

    # Mock accuracy check: Compare cosine similarity to the ground truth embedding
    similarity = np.dot(query_embedding, gt_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(gt_embedding))
    # Let's say a similarity > 0.8 is a "correct" match for this mock test.
    accuracy = 1.0 if similarity > 0.8 else 0.0
    
    return end_time - start_time, accuracy

# --- Main Execution ---

def main():
    logger.info("Starting benchmark suite...")
    create_dummy_image()

    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)

    base_image = read_image(SAMPLE_IMAGE_PATH)
    if base_image is None:
        logger.error(f"Failed to load sample image from {SAMPLE_IMAGE_PATH}")
        return

    image_qualities = get_image_qualities(base_image)
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    else:
        logger.warning("CUDA not available. Running benchmarks on CPU only.")

    results = []

    # Generate a stable ground truth embedding for accuracy checks
    logger.info("Generating ground truth embedding for accuracy checks...")
    rec_model_cpu = RecognitionModel(arcface_path="", gcn_path="")
    rec_model_cpu.device = "cpu"
    gt_embedding = rec_model_cpu.extract_fused_embedding(image_qualities["clean"])

    for device in devices:
        logger.info(f"--- Benchmarking on device: {device.upper()} ---")
        
        # --- Reconstruction Benchmark ---
        logger.info("Loading Reconstruction Model...")
        recon_model = ReconstructionModel(model_path="")
        recon_model.device = device
        
        for quality_name, image in image_qualities.items():
            logger.info(f"Running Reconstruction benchmark for quality: {quality_name}")
            tracemalloc.start()
            
            runtime = benchmark_reconstruction(recon_model, image)
            
            _, mem_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            results.append({
                "task": "reconstruction",
                "device": device,
                "quality": quality_name,
                "runtime_sec": runtime,
                "peak_mem_mb": mem_peak / (1024 * 1024),
                "accuracy": None # Accuracy not applicable for reconstruction in this setup
            })

        # --- Recognition Benchmark ---
        logger.info("Loading Recognition Model...")
        rec_model = RecognitionModel(arcface_path="", gcn_path="")
        rec_model.device = device

        for quality_name, image in image_qualities.items():
            logger.info(f"Running Recognition benchmark for quality: {quality_name}")
            tracemalloc.start()
            
            runtime, accuracy = benchmark_recognition(rec_model, image, gt_embedding)
            
            _, mem_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            results.append({
                "task": "recognition",
                "device": device,
                "quality": quality_name,
                "runtime_sec": runtime,
                "peak_mem_mb": mem_peak / (1024 * 1024),
                "accuracy": accuracy
            })

    # --- Save Results ---
    df = pd.DataFrame(results)
    output_path = os.path.join(REPORTS_DIR, "benchmark_results.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"Benchmark results saved to {output_path}")
    
    # Print results to stdout for frontend display
    print("Performance Benchmark Results:")
    print("=" * 50)
    for result in results:
        print(f"Task: {result['task']}")
        print(f"Device: {result['device']}")
        print(f"Quality: {result['quality']}")
        print(".2f")
        print(".2f")
        print(".4f")
        print("-" * 30)

if __name__ == "__main__":
    main()
