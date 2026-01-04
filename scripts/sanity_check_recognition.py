#!/usr/bin/env python3
"""
Quick sanity check for recognition training with minimal data.
"""

import torch
from pathlib import Path
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader, Subset
import numpy as np
import os

from training.utils.dataset_loader import get_recognition_dataset
from training.utils.evaluation_metrics import evaluate_classification
from training.recognition_train import RecognitionNet


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load dataset from train split specifically
    data_path = "Data/recognition_faces/train"
    print(f"Loading dataset from: {data_path}")

    dataset = get_recognition_dataset(data_dir=data_path, split=None, fast_mode=True)

    # Select only first 5 classes (identities) for quick test
    classes_to_keep = sorted(list(dataset.classes))[:5]
    print(f"Testing with {len(classes_to_keep)} identities: {classes_to_keep}")

    # Filter dataset to only include samples from these classes
    indices_to_keep = []
    for i, label in enumerate(dataset.labels):
        class_name = dataset.classes[label]
        if class_name in classes_to_keep:
            indices_to_keep.append(i)

    # Limit to 50 samples per class for speed
    samples_per_class = {}
    final_indices = []
    for idx in indices_to_keep:
        label = dataset.labels[idx]
        class_name = dataset.classes[label]
        if class_name not in samples_per_class:
            samples_per_class[class_name] = 0

        if samples_per_class[class_name] < 50:  # max 50 samples per class
            final_indices.append(idx)
            samples_per_class[class_name] += 1

    print(f"Using {len(final_indices)} total samples")
    for class_name, count in samples_per_class.items():
        print(f"  {class_name}: {count} samples")

    # Create subset
    subset_dataset = Subset(dataset, final_indices)

    # Remap labels to 0,1,2,... for the subset
    class_to_idx = {cls: i for i, cls in enumerate(classes_to_keep)}
    num_classes = len(classes_to_keep)

    # Create a wrapper to remap labels
    class RemappedDataset(torch.utils.data.Dataset):
        def __init__(self, subset, class_to_idx):
            self.subset = subset
            self.class_to_idx = class_to_idx

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            img, label = self.subset[idx]
            original_class = subset_dataset.dataset.classes[label]
            remapped_label = self.class_to_idx[original_class]
            return img, remapped_label

    remapped_dataset = RemappedDataset(subset_dataset, class_to_idx)

    # Split into train/val/test
    train_size = int(0.7 * len(remapped_dataset))
    val_size = int(0.15 * len(remapped_dataset))
    test_size = len(remapped_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        remapped_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    print(f"Split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Create model
    model = RecognitionNet(num_classes=num_classes, embedding_size=128).to(device)

    # Quick training test
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print("\nTesting training loop...")
    model.train()
    for epoch in range(1):  # Just 1 epoch
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i >= 2:  # Just a few batches
                break

        print(f"  Epoch {epoch+1} - Avg Loss: {running_loss / (i+1):.4f}")
    # Test evaluation
    print("\nTesting evaluation...")
    try:
        val_metrics = evaluate_classification(model, val_loader, device, num_classes)
        print("✓ Evaluation successful!")
        print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  F1: {val_metrics['f1']:.4f}")
        print(f"  Keys available: {list(val_metrics.keys())}")

        # Test the aliases we added
        if 'predictions' in val_metrics and 'labels' in val_metrics and 'probabilities' in val_metrics:
            print("✓ Aliases working correctly!")
        else:
            print("⚠ Aliases not found")

    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        return False

    print("\n✓ Sanity check passed! The script should work with full data.")
    return True


if __name__ == "__main__":
    main()