#!/usr/bin/env python3
"""
Extract embeddings from trained 3D face reconstruction model for recognition tasks.

This script loads the trained reconstruction model and extracts feature embeddings
from the backbone network (before the parameter prediction layer) for use in
face recognition and verification tasks.
"""

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from training.utils.dataset_loader import get_reconstruction_dataset
from training.utils.universal_pipeline import UniversalTransforms, DatasetSplitter


class ReconstructionFeatureExtractor(nn.Module):
    """
    Feature extractor that uses the backbone of the trained reconstruction model
    to extract embeddings for recognition tasks.
    """

    def __init__(self, reconstruction_model):
        super(ReconstructionFeatureExtractor, self).__init__()
        # Extract the backbone from the reconstruction model
        self.backbone = reconstruction_model.backbone
        self.adaptive_pool = reconstruction_model.adaptive_pool

        # Get the feature dimension
        self.feature_dim = 64 * 112 * 112  # Based on the backbone architecture

    def forward(self, x):
        """
        Extract features from input images.

        Args:
            x: Input images (B, 3, H, W)

        Returns:
            features: Feature embeddings (B, feature_dim)
        """
        features = self.backbone(x)
        features = self.adaptive_pool(features)
        features = features.view(features.size(0), -1)
        return features


def extract_embeddings(model_path, data_path, output_dir, batch_size=32, device='cuda'):
    """
    Extract embeddings from test dataset using trained reconstruction model.

    Args:
        model_path: Path to trained reconstruction model checkpoint
        data_path: Path to dataset directory
        output_dir: Directory to save embeddings and labels
        batch_size: Batch size for inference
        device: Device to run inference on
    """

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the trained reconstruction model
    print(f"Loading reconstruction model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # Recreate the model architecture
    # We need to determine the number of parameters from the checkpoint
    model_state = checkpoint['model_state_dict']

    # Find the output dimension from the param_predictor layer
    param_predictor_weight = model_state['param_predictor.weight']
    num_params = param_predictor_weight.shape[0]

    print(f"Detected {num_params} reconstruction parameters")

    # Create the model with the correct architecture
    class ReconstructionNet(nn.Module):
        def __init__(self, num_params):
            super(ReconstructionNet, self).__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=1, padding=3, bias=False),
                nn.ReLU(inplace=True)
            )
            self.adaptive_pool = nn.AdaptiveAvgPool2d((112, 112))
            self.param_predictor = nn.Linear(64 * 112 * 112, num_params)

        def forward(self, x):
            features = self.backbone(x)
            features = self.adaptive_pool(features)
            features = features.view(features.size(0), -1)
            params = self.param_predictor(features)
            return params

    model = ReconstructionNet(num_params=num_params).to(device)
    model.load_state_dict(model_state)
    model.eval()

    # Create feature extractor from the model
    feature_extractor = ReconstructionFeatureExtractor(model)
    feature_extractor.to(device)
    feature_extractor.eval()

    print(f"Feature dimension: {feature_extractor.feature_dim}")

    # Load dataset
    print(f"Loading dataset from: {data_path}")
    full_dataset = get_reconstruction_dataset(data_dir=data_path, split=None)

    # Split into train/val/test (70/15/15) - same as training
    train_dataset, val_dataset, test_dataset = DatasetSplitter.split_dataset(
        full_dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42
    )

    # Apply validation transforms (same as used during training)
    val_transform = UniversalTransforms.get_val_transforms()

    if hasattr(test_dataset, 'dataset'):
        test_dataset.dataset.transform = val_transform

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Test dataset size: {len(test_dataset)}")

    # Extract embeddings
    print("Extracting embeddings from test set...")
    all_embeddings = []
    all_labels = []
    all_params = []

    with torch.no_grad():
        for images, params in tqdm(test_loader, desc="Extracting embeddings"):
            images = images.to(device)

            # Extract features
            embeddings = feature_extractor(images)

            # Store results
            all_embeddings.append(embeddings.cpu().numpy())
            all_params.append(params.numpy())

            # For labels, we'll use the parameter values as pseudo-labels
            # In a real scenario, you'd have identity labels
            # For now, we'll create dummy identity labels based on parameter similarity
            all_labels.extend([f"sample_{i}" for i in range(len(images))])

    # Concatenate all batches
    embeddings = np.concatenate(all_embeddings, axis=0)
    params = np.concatenate(all_params, axis=0)
    labels = np.array(all_labels)

    print(f"Extracted {len(embeddings)} embeddings")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Parameter shape: {params.shape}")

    # Save embeddings and labels
    embeddings_path = output_dir / 'test_embeddings.npy'
    labels_path = output_dir / 'test_labels.npy'
    params_path = output_dir / 'test_params.npy'

    np.save(embeddings_path, embeddings)
    np.save(labels_path, labels)
    np.save(params_path, params)

    print(f"✓ Embeddings saved to: {embeddings_path}")
    print(f"✓ Labels saved to: {labels_path}")
    print(f"✓ Parameters saved to: {params_path}")

    # Save metadata
    metadata = {
        'model_path': str(model_path),
        'data_path': str(data_path),
        'num_samples': len(embeddings),
        'embedding_dim': embeddings.shape[1],
        'param_dim': params.shape[1],
        'batch_size': batch_size,
        'device': str(device)
    }

    import json
    metadata_path = output_dir / 'extraction_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Metadata saved to: {metadata_path}")

    # Print statistics
    print("\nEmbedding Statistics:")
    print(f"  Mean: {embeddings.mean():.6f}")
    print(f"  Std:  {embeddings.std():.6f}")
    print(f"  Min:  {embeddings.min():.6f}")
    print(f"  Max:  {embeddings.max():.6f}")

    print("\nParameter Statistics:")
    print(f"  Mean: {params.mean():.6f}")
    print(f"  Std:  {params.std():.6f}")
    print(f"  Min:  {params.min():.6f}")
    print(f"  Max:  {params.max():.6f}")

    return embeddings, labels, params


def main():
    parser = argparse.ArgumentParser(description='Extract embeddings from trained reconstruction model')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained reconstruction model checkpoint')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to AFLW2000 dataset directory')
    parser.add_argument('--output-dir', type=str, default='./embeddings',
                       help='Directory to save embeddings and labels')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run inference on')

    args = parser.parse_args()

    extract_embeddings(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device
    )


if __name__ == "__main__":
    main()