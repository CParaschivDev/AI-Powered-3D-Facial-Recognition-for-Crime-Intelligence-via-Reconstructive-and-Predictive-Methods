#!/usr/bin/env python3
"""
Extract embeddings from trained landmark detection model for recognition tasks.

This script loads the trained landmark model and extracts landmark coordinates
and/or intermediate features for use in face recognition and verification tasks.
"""

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from training.utils.dataset_loader import get_landmark_dataset
from training.utils.universal_pipeline import UniversalTransforms, DatasetSplitter


class LandmarkFeatureExtractor(nn.Module):
    """
    Feature extractor that uses the trained landmark detection model
    to extract landmark coordinates and/or intermediate features.
    """

    def __init__(self, landmark_model, extract_coordinates=True, extract_features=False):
        super(LandmarkFeatureExtractor, self).__init__()
        # Store the full model
        self.landmark_model = landmark_model

        # Configuration
        self.extract_coordinates = extract_coordinates
        self.extract_features = extract_features

        # Get dimensions
        self.num_landmarks = None
        self.feature_dim = None

        # Try to infer dimensions from the model
        if hasattr(landmark_model, 'fc1'):
            self.feature_dim = landmark_model.fc1.out_features

        # Extract number of landmarks from model structure
        if hasattr(landmark_model, 'output_layer') and landmark_model.output_layer is not None:
            self.num_landmarks = landmark_model.output_layer.out_features // 2
        elif hasattr(landmark_model, 'fc1'):
            self.num_landmarks = landmark_model.fc1.out_features // 2

    def forward(self, x):
        """
        Extract features and/or coordinates from input images.

        Args:
            x: Input images (B, 3, H, W)

        Returns:
            results: Dict containing 'coordinates' and/or 'features'
        """
        results = {}

        # Get intermediate features before final layers
        x = torch.relu(self.landmark_model.conv1(x))
        x = self.landmark_model.adaptive_pool(x)
        features = x.view(x.size(0), -1)

        if self.extract_features:
            results['features'] = features

        # Get final landmark predictions
        x = self.landmark_model.fc1(features)

        # Apply feature selection if configured
        if self.landmark_model.feature_selector_layer is not None:
            selected_features = self.landmark_model.feature_selector_layer(x)
            if self.extract_features and 'selected_features' not in results:
                results['selected_features'] = selected_features
            x = self.landmark_model.output_layer(selected_features)

        # Reshape to landmark coordinates
        coordinates = x.view(x.size(0), -1, 2)  # (B, num_landmarks, 2)

        if self.extract_coordinates:
            results['coordinates'] = coordinates

        return results


def extract_embeddings(model_path, data_path, output_dir, batch_size=32, device='cuda',
                      extract_coordinates=True, extract_features=False):
    """
    Extract embeddings from test dataset using trained landmark model.

    Args:
        model_path: Path to trained landmark model checkpoint
        data_path: Path to dataset directory
        output_dir: Directory to save embeddings and labels
        batch_size: Batch size for inference
        device: Device to run inference on
        extract_coordinates: Whether to extract landmark coordinates
        extract_features: Whether to extract intermediate features
    """

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the trained landmark model
    print(f"Loading landmark model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # Recreate the model architecture
    # We need to determine the architecture from the checkpoint
    model_state = checkpoint['model_state_dict']

    # Infer number of landmarks from the final layer
    if 'fc1.weight' in model_state:
        fc1_weight = model_state['fc1.weight']
        num_landmarks = fc1_weight.shape[0] // 2  # Divide by 2 for (x,y) coordinates
    else:
        raise ValueError("Could not determine number of landmarks from model checkpoint")

    print(f"Detected {num_landmarks} landmarks per face")

    # Check if feature selection was used
    has_feature_selection = 'feature_selector_layer.weight' in model_state
    feature_dim = None

    if has_feature_selection:
        feature_selector_weight = model_state['feature_selector_layer.weight']
        feature_dim = feature_selector_weight.shape[0]
        print(f"Model uses feature selection with {feature_dim} features")
    else:
        print("Model does not use feature selection")

    # Create the model with the correct architecture
    class LandmarkNet(nn.Module):
        def __init__(self, num_landmarks=4000, feature_dim=None):
            super(LandmarkNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.adaptive_pool = nn.AdaptiveAvgPool2d((56, 56))
            self.fc1 = nn.Linear(16 * 56 * 56, num_landmarks * 2)

            # Feature selection layer (optional)
            if feature_dim is not None and feature_dim < num_landmarks * 2:
                self.feature_selector_layer = nn.Linear(num_landmarks * 2, feature_dim)
                self.output_layer = nn.Linear(feature_dim, num_landmarks * 2)
            else:
                self.feature_selector_layer = None
                self.output_layer = None

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.adaptive_pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc1(x)

            # Apply feature selection if configured
            if self.feature_selector_layer is not None:
                features = self.feature_selector_layer(x)
                x = self.output_layer(features)

            return x.view(x.size(0), -1, 2)

    model = LandmarkNet(num_landmarks=num_landmarks, feature_dim=feature_dim).to(device)
    model.load_state_dict(model_state)
    model.eval()

    # Create feature extractor
    feature_extractor = LandmarkFeatureExtractor(
        model,
        extract_coordinates=extract_coordinates,
        extract_features=extract_features
    )
    feature_extractor.to(device)
    feature_extractor.eval()

    # Load dataset
    print(f"Loading dataset from: {data_path}")
    full_dataset = get_landmark_dataset(data_dir=data_path, split=None)

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
    all_coordinates = []
    all_features = []
    all_selected_features = []
    all_gt_landmarks = []

    with torch.no_grad():
        for images, gt_landmarks in tqdm(test_loader, desc="Extracting embeddings"):
            images = images.to(device)

            # Extract features
            results = feature_extractor(images)

            # Store results
            if extract_coordinates and 'coordinates' in results:
                all_coordinates.append(results['coordinates'].cpu().numpy())

            if extract_features and 'features' in results:
                all_features.append(results['features'].cpu().numpy())

            if extract_features and 'selected_features' in results:
                all_selected_features.append(results['selected_features'].cpu().numpy())

            all_gt_landmarks.append(gt_landmarks.numpy())

    # Concatenate all batches
    results_data = {}

    if all_coordinates:
        coordinates = np.concatenate(all_coordinates, axis=0)
        results_data['coordinates'] = coordinates
        print(f"Extracted {len(coordinates)} coordinate embeddings")
        print(f"Coordinate shape: {coordinates.shape}")

    if all_features:
        features = np.concatenate(all_features, axis=0)
        results_data['features'] = features
        print(f"Extracted {len(features)} feature embeddings")
        print(f"Feature shape: {features.shape}")

    if all_selected_features:
        selected_features = np.concatenate(all_selected_features, axis=0)
        results_data['selected_features'] = selected_features
        print(f"Extracted {len(selected_features)} selected feature embeddings")
        print(f"Selected feature shape: {selected_features.shape}")

    gt_landmarks = np.concatenate(all_gt_landmarks, axis=0)
    results_data['gt_landmarks'] = gt_landmarks
    print(f"Ground truth landmarks shape: {gt_landmarks.shape}")

    # Create labels (sample IDs for now)
    labels = np.array([f"sample_{i}" for i in range(len(gt_landmarks))])
    results_data['labels'] = labels

    # Save all results
    for key, data in results_data.items():
        output_path = output_dir / f'test_{key}.npy'
        np.save(output_path, data)
        print(f"✓ {key} saved to: {output_path}")

    # Save metadata
    metadata = {
        'model_path': str(model_path),
        'data_path': str(data_path),
        'num_samples': len(labels),
        'num_landmarks': num_landmarks,
        'has_feature_selection': has_feature_selection,
        'feature_dim': feature_dim,
        'extract_coordinates': extract_coordinates,
        'extract_features': extract_features,
        'batch_size': batch_size,
        'device': str(device)
    }

    # Add shape information
    for key, data in results_data.items():
        if key != 'labels':
            metadata[f'{key}_shape'] = list(data.shape)

    import json
    metadata_path = output_dir / 'extraction_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Metadata saved to: {metadata_path}")

    # Print statistics for each embedding type
    for key, data in results_data.items():
        if key != 'labels' and isinstance(data, np.ndarray):
            print(f"\n{key} Statistics:")
            print(f"  Mean: {data.mean():.6f}")
            print(f"  Std:  {data.std():.6f}")
            print(f"  Min:  {data.min():.6f}")
            print(f"  Max:  {data.max():.6f}")

    return results_data


def main():
    parser = argparse.ArgumentParser(description='Extract embeddings from trained landmark model')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained landmark model checkpoint')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to AFLW2000 dataset directory')
    parser.add_argument('--output-dir', type=str, default='./landmark_embeddings',
                       help='Directory to save embeddings and labels')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run inference on')
    parser.add_argument('--extract-coordinates', action='store_true', default=True,
                       help='Extract landmark coordinates as embeddings')
    parser.add_argument('--extract-features', action='store_true', default=False,
                       help='Extract intermediate features as embeddings')

    args = parser.parse_args()

    extract_embeddings(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        extract_coordinates=args.extract_coordinates,
        extract_features=args.extract_features
    )


if __name__ == "__main__":
    main()