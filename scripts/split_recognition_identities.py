#!/usr/bin/env python3
"""
Split Recognition Identities Script

This script preprocesses facial recognition data to create clean train/val splits
at the identity level, avoiding label collisions between different data sources.

Usage:
    python split_recognition_identities.py \
        --source-roots Data/actor_faces Data/actress_faces \
        --output-root Data/recognition_faces \
        --val-ratio 0.2 \
        --seed 42
"""

import os
import shutil
import argparse
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Set


def collect_identities(source_roots: List[str]) -> Dict[str, List[str]]:
    """
    Collect all identity names and their source directories.

    Args:
        source_roots: List of root directories containing identity subfolders

    Returns:
        Dictionary mapping identity names to list of source directories containing that identity
    """
    identities = defaultdict(list)

    for root in source_roots:
        root_path = Path(root)
        if not root_path.exists():
            print(f"Warning: Source root {root} does not exist, skipping")
            continue

        print(f"Scanning {root} for identities...")

        # Get all subdirectories (identities)
        for item in root_path.iterdir():
            if item.is_dir():
                identity_name = item.name
                identities[identity_name].append(str(item))

    return dict(identities)


def split_identities(identities: Dict[str, List[str]], val_ratio: float, seed: int) -> tuple:
    """
    Split identities into train and validation sets.

    Args:
        identities: Dictionary mapping identity names to source directories
        val_ratio: Fraction of identities to put in validation set
        seed: Random seed for reproducible splits

    Returns:
        Tuple of (train_identities, val_identities) dictionaries
    """
    identity_names = list(identities.keys())
    random.seed(seed)

    # Shuffle identities for random split
    random.shuffle(identity_names)

    # Calculate split point
    n_val = int(len(identity_names) * val_ratio)
    val_names = set(identity_names[:n_val])
    train_names = set(identity_names[n_val:])

    # Create split dictionaries
    train_identities = {name: identities[name] for name in train_names}
    val_identities = {name: identities[name] for name in val_names}

    return train_identities, val_identities


def copy_identity_images(identity_name: str, source_dirs: List[str], dest_dir: Path) -> int:
    """
    Copy all images for an identity from source directories to destination.

    Args:
        identity_name: Name of the identity
        source_dirs: List of source directories containing this identity
        dest_dir: Destination directory (train/ or val/)

    Returns:
        Number of images copied
    """
    identity_dest = dest_dir / identity_name
    identity_dest.mkdir(parents=True, exist_ok=True)

    total_images = 0
    used_filenames = set()

    # Copy images from all source directories for this identity
    for source_dir in source_dirs:
        source_path = Path(source_dir)

        if not source_path.exists():
            continue

        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        for file_path in source_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                # Handle filename collisions by adding suffix
                base_name = file_path.stem
                extension = file_path.suffix
                filename = f"{base_name}{extension}"
                counter = 1

                while filename in used_filenames:
                    filename = f"{base_name}_{counter}{extension}"
                    counter += 1

                used_filenames.add(filename)

                # Copy the file
                dest_file = identity_dest / filename
                shutil.copy2(file_path, dest_file)
                total_images += 1

    return total_images


def create_split_directories(train_identities: Dict[str, List[str]],
                           val_identities: Dict[str, List[str]],
                           output_root: str) -> None:
    """
    Create the train/val directory structure and copy images.

    Args:
        train_identities: Dictionary of identities for training
        val_identities: Dictionary of identities for validation
        output_root: Root directory for the split data
    """
    output_path = Path(output_root)

    # Create train and val directories
    train_dir = output_path / "train"
    val_dir = output_path / "val"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCreating training set in {train_dir}")
    total_train_images = 0
    for identity_name, source_dirs in train_identities.items():
        images_copied = copy_identity_images(identity_name, source_dirs, train_dir)
        total_train_images += images_copied
        print(f"  {identity_name}: {images_copied} images")

    print(f"\nCreating validation set in {val_dir}")
    total_val_images = 0
    for identity_name, source_dirs in val_identities.items():
        images_copied = copy_identity_images(identity_name, source_dirs, val_dir)
        total_val_images += images_copied
        print(f"  {identity_name}: {images_copied} images")

    print("\nSummary:")
    print(f"  Training identities: {len(train_identities)}")
    print(f"  Training images: {total_train_images}")
    print(f"  Validation identities: {len(val_identities)}")
    print(f"  Validation images: {total_val_images}")
    print(f"  Total identities: {len(train_identities) + len(val_identities)}")
    print(f"  Total images: {total_train_images + total_val_images}")


def main():
    parser = argparse.ArgumentParser(description="Split facial recognition identities for clean train/val splits")
    parser.add_argument("--source-roots", nargs="+", required=True,
                       help="Source root directories containing identity subfolders")
    parser.add_argument("--output-root", required=True,
                       help="Output root directory for train/val split")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                       help="Fraction of identities to put in validation set (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible splits (default: 42)")
    parser.add_argument("--force", action="store_true",
                       help="Overwrite existing output directory without confirmation")

    args = parser.parse_args()

    # Validate arguments
    if not (0 < args.val_ratio < 1):
        parser.error("val-ratio must be between 0 and 1")

    output_path = Path(args.output_root)

    # Check if output directory exists and handle it
    if output_path.exists() and any(output_path.iterdir()):
        if not args.force:
            response = input(f"Output directory {output_path} is not empty. Overwrite? (y/N): ")
            if response.lower() != 'y':
                print("Aborted.")
                return
        print(f"Cleaning existing output directory {output_path}")
        shutil.rmtree(output_path)

    print("Facial Recognition Identity Split Tool")
    print("=" * 50)
    print(f"Source roots: {args.source_roots}")
    print(f"Output root: {args.output_root}")
    print(f"Validation ratio: {args.val_ratio}")
    print(f"Random seed: {args.seed}")
    print()

    # Step 1: Collect all identities
    print("Step 1: Collecting identities from source directories...")
    identities = collect_identities(args.source_roots)

    if not identities:
        print("Error: No identities found in source directories")
        return

    print(f"Found {len(identities)} unique identities")

    # Check for identities that appear in multiple sources
    multi_source = {name: sources for name, sources in identities.items() if len(sources) > 1}
    if multi_source:
        print(f"Found {len(multi_source)} identities that appear in multiple sources:")
        for name, sources in multi_source.items():
            print(f"  {name}: {len(sources)} sources")
    else:
        print("All identities are unique across sources")

    # Step 2: Split identities
    print("\nStep 2: Splitting identities into train/val sets...")
    train_identities, val_identities = split_identities(identities, args.val_ratio, args.seed)

    print(f"Training identities: {len(train_identities)}")
    print(f"Validation identities: {len(val_identities)}")

    # Step 3: Create directory structure and copy images
    print("\nStep 3: Creating directory structure and copying images...")
    create_split_directories(train_identities, val_identities, args.output_root)

    print("\n✅ Identity split completed successfully!")
    print(f"Training data: {args.output_root}/train")
    print(f"Validation data: {args.output_root}/val")
    print("\nYou can now train with:")
    print(f"python -m training.recognition_train --data-path \"{args.output_root}\" --epochs 30 --batch-size 64 --lr 1e-3")


if __name__ == "__main__":
    main()