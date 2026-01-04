#!/usr/bin/env python3
"""
Generate embeddings and labels from test dataset using trained recognition model.
"""

import torch
from pathlib import Path
import sys
import os
import numpy as np

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.utils.universal_pipeline import UniversalTransforms, DatasetSplitter
from training.recognition_train import RecognitionNet
from training.utils.dataset_loader import get_recognition_dataset
from torch.utils.data import DataLoader

def generate_embeddings(checkpoint_path: str, data_path: str, num_classes: int = 530, feature_dim: int = 256, output_dir: str = "./logs/recognition"):
    """Generate embeddings and labels from test dataset."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model architecture
    model = RecognitionNet(num_classes=num_classes, feature_dim=feature_dim, head_type='linear')
    model.to(device)

    # Load trained weights
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("✓ Model loaded successfully")

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
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    print(f"✓ Test dataset loaded: {len(test_dataset)} samples")

    # Generate embeddings
    embeddings = []
    labels = []

    print("Generating embeddings...")
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            # Get embeddings
            outputs = model(inputs, return_embedding=True)
            embeddings.append(outputs.cpu().numpy())
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
    np.save(output_path / 'test_embeddings.npy', embeddings)
    np.save(output_path / 'test_labels.npy', labels)

    print(f"✓ Saved test_embeddings.npy to {output_path}")
    print(f"✓ Saved test_labels.npy to {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate embeddings from test dataset")
    parser.add_argument("--checkpoint", type=str, default="./logs/recognition/recognition_best_model.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--data-path", type=str, default="./Data/recognition_faces",
                        help="Path to dataset directory")
    parser.add_argument("--num-classes", type=int, default=530,
                        help="Number of classes in the model")
    parser.add_argument("--feature-dim", type=int, default=256,
                        help="Feature dimension used in training")
    parser.add_argument("--output-dir", type=str, default="./logs/recognition",
                        help="Directory to save embeddings")

    args = parser.parse_args()
    generate_embeddings(args.checkpoint, args.data_path, args.num_classes, args.feature_dim, args.output_dir)