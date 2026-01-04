#!/usr/bin/env python3
"""
Generate embeddings using Buffalo (InsightFace) model for comparison.
"""

import torch
from pathlib import Path
import sys
import os
import numpy as np
import cv2
from insightface.app import FaceAnalysis

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.utils.dataset_loader import get_recognition_dataset
from training.utils.universal_pipeline import DatasetSplitter
from torch.utils.data import DataLoader

def generate_buffalo_embeddings(data_path: str, output_dir: str = "./logs/buffalo"):
    """Generate embeddings using Buffalo model."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Buffalo model
    print("Loading Buffalo model...")
    buffalo_path = r"C:\Users\Paras\.insightface\models\buffalo_l"
    buffalo_model = FaceAnalysis(name='buffalo_l', root=buffalo_path, allowed_modules=['detection', 'recognition'])
    buffalo_model.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
    
    # Adjust detection threshold for better face detection
    buffalo_model.det_model.det_thresh = 0.1  # Lower threshold
    buffalo_model.det_model.nms_thresh = 0.3   # Adjust NMS threshold
    print("✓ Buffalo model loaded")

    # Load and split dataset
    train_path = os.path.join(data_path, 'train')
    if os.path.exists(train_path):
        data_path = train_path
    print(f"Loading data from: {data_path}")
    full_dataset = get_recognition_dataset(data_dir=data_path, split=None, fast_mode=False)

    # Split into train/val/test
    train_dataset, val_dataset, test_dataset = DatasetSplitter.split_dataset(
        full_dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42
    )

    print(f"✓ Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    print(f"✓ Classes: {len(full_dataset.classes)}")

    # Use test dataset
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)  # Smaller batch for buffalo

    print(f"✓ Test dataset loaded: {len(test_dataset)} samples")

    # Generate embeddings
    embeddings = []
    labels = []

    print("Generating Buffalo embeddings...")
    for inputs, targets in test_loader:
        # Process batch
        batch_embeddings = []
        for i in range(inputs.size(0)):
            # Get single image
            img_tensor = inputs[i]  # (3, H, W)
            # Convert to numpy BGR
            img_np = img_tensor.permute(1, 2, 0).numpy()  # (H, W, 3) RGB
            img_bgr = img_np[..., ::-1] * 255  # RGB -> BGR, scale to 0-255
            img_bgr = img_bgr.astype(np.uint8)
            
            # Get embedding - since these are pre-aligned face crops, skip detection
            # Get embedding using FaceAnalysis get method
            faces = buffalo_model.get(img_bgr)
            if faces and len(faces) > 0:
                embedding = faces[0].embedding.astype(np.float32)
                # Check for NaN values
                if np.isnan(embedding).any():
                    print(f"Warning: NaN embedding detected, using fallback")
                    embedding = np.random.normal(0, 0.1, 512).astype(np.float32)  # Small random vector
                # Normalize embedding
                embedding = embedding / (np.linalg.norm(embedding) + 1e-12)
            else:
                # Fallback: small random vector if no face detected
                print(f"Warning: No face detected in image, using random fallback")
                embedding = np.random.normal(0, 0.1, 512).astype(np.float32)
            
            batch_embeddings.append(embedding)
        
        embeddings.append(np.array(batch_embeddings))
        labels.append(targets.numpy())

    # Concatenate
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)

    print(f"✓ Generated embeddings: shape {embeddings.shape}")
    print(f"✓ Generated labels: shape {labels.shape}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save
    np.save(output_path / 'embeddings.npy', embeddings)
    np.save(output_path / 'labels.npy', labels)

    print(f"✓ Saved embeddings.npy to {output_path}")
    print(f"✓ Saved labels.npy to {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Buffalo embeddings for comparison")
    parser.add_argument("--data-path", type=str, default="../Data/recognition_faces",
                        help="Path to dataset directory")
    parser.add_argument("--output-dir", type=str, default="../logs/buffalo",
                        help="Directory to save embeddings")

    args = parser.parse_args()
    generate_buffalo_embeddings(args.data_path, args.output_dir)