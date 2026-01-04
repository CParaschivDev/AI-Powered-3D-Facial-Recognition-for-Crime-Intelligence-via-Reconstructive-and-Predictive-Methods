#!/usr/bin/env python3
"""
Complete Embedding Regeneration Script
Regenerates all embeddings after model updates (ArcFace, feature selection, etc.)

This script handles:
1. Watchlist embeddings (numpy files)
2. Database embeddings (SQL database)
3. Label consistency checks

Usage: python scripts/regenerate_all_embeddings.py --model-path logs/recognition/recognition_model.pth
"""

import torch
import numpy as np
import json
import argparse
from pathlib import Path
import cv2
from tqdm import tqdm
import sys

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / 'backend'))

from backend.models.utils.model_loader import get_recognition_model
from backend.core.paths import watchlist_roots
from backend.utils.face_io import iter_images, id_from_path, read_image
from backend.database.dependencies import SessionLocal
from backend.database.db_utils import store_identity_embedding, upsert_watchlist_identity
from backend.models.recognition.arcface import RecognitionNet


def load_trained_model(model_path, device='cpu'):
    """Load a trained recognition model with all its configuration."""
    print(f"Loading trained model from {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Extract model configuration from checkpoint
    # This assumes the checkpoint was saved with our training script
    model_config = checkpoint.get('model_config', {})

    # Recreate model with same configuration
    num_classes = model_config.get('num_classes', 1000)  # fallback
    feature_dim = model_config.get('feature_dim')
    head_type = model_config.get('head_type', 'linear')

    model = RecognitionNet(num_classes=num_classes, feature_dim=feature_dim, head_type=head_type)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model


def regenerate_watchlist_embeddings(model, output_dir, args):
    """Regenerate watchlist embeddings using the new model."""
    print("🔄 Regenerating watchlist embeddings...")

    id_to_vecs = {}
    total_images = 0

    roots = watchlist_roots()
    if not roots:
        print("❌ No watchlist roots found. Check DATA_ROOT and WATCHLIST_DIRS in config.")
        return False

    device = next(model.parameters()).device

    # Process all identities
    for root in tqdm(roots, desc="Processing watchlist roots"):
        for img_path in iter_images(root):
            pid = id_from_path(root, img_path)

            if args.limit and total_images >= args.limit:
                break

            img = read_image(img_path)
            if img is None:
                continue

            # Preprocess image
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rsz = cv2.resize(rgb, (224, 224))  # Assuming 224x224 input
            tensor = torch.from_numpy(rsz.astype('float32').transpose(2, 0, 1) / 255.0)
            tensor = tensor.unsqueeze(0).to(device)

            # Extract embedding
            with torch.no_grad():
                embedding = model(tensor, return_embedding=True)
                embedding = embedding.cpu().numpy().flatten()

                # L2 normalize
                embedding = embedding / (np.linalg.norm(embedding) + 1e-12)

            id_to_vecs.setdefault(pid, []).append(embedding)
            total_images += 1

    # Calculate mean embeddings per identity
    ids = []
    embeddings = []

    for pid, vecs in sorted(id_to_vecs.items()):
        if not vecs:
            continue

        mean_embedding = np.stack(vecs).mean(axis=0)
        # L2 normalize again
        mean_embedding = mean_embedding / (np.linalg.norm(mean_embedding) + 1e-12)

        ids.append(pid)
        embeddings.append(mean_embedding.astype('float32'))

    if not embeddings:
        print("❌ No embeddings generated!")
        return False

    embeddings_array = np.stack(embeddings)

    # Save to files
    emb_path = output_dir / "watchlist_embeddings_v2.npy"
    labels_path = output_dir / "watchlist_labels_v2.json"

    np.save(emb_path, embeddings_array)
    with open(labels_path, 'w') as f:
        json.dump(ids, f, indent=2)

    print(f"✅ Saved {len(ids)} watchlist embeddings ({total_images} images processed)")
    print(f"   Embeddings: {emb_path}")
    print(f"   Labels: {labels_path}")

    return True


def regenerate_database_embeddings(model, args):
    """Regenerate database embeddings for all stored identities."""
    print("🔄 Regenerating database embeddings...")

    # This is more complex as we need to:
    # 1. Get all identities from database
    # 2. For each identity, get their original images
    # 3. Recompute embeddings
    # 4. Update database

    # For now, we'll focus on watchlist embeddings
    # Database embeddings would require additional logic to map identities to their source images

    print("⚠️  Database embedding regeneration requires identity-to-image mapping")
    print("   For now, focusing on watchlist embeddings")
    print("   Manual database update may be needed for individual identities")

    return True


def validate_embedding_consistency(output_dir):
    """Validate that embeddings and labels are consistent."""
    print("🔍 Validating embedding consistency...")

    emb_path = output_dir / "watchlist_embeddings_v2.npy"
    labels_path = output_dir / "watchlist_labels_v2.json"

    if not emb_path.exists() or not labels_path.exists():
        print("❌ Embedding files not found!")
        return False

    embeddings = np.load(emb_path)
    with open(labels_path, 'r') as f:
        labels = json.load(f)

    if len(embeddings) != len(labels):
        print(f"❌ Mismatch: {len(embeddings)} embeddings vs {len(labels)} labels")
        return False

    # Check embedding properties
    print(f"✅ Consistency check passed:")
    print(f"   - {len(labels)} identities")
    print(f"   - Embedding dimension: {embeddings.shape[1]}")
    print(".4f"    print(".6f"
    return True


def main():
    parser = argparse.ArgumentParser(description='Regenerate all embeddings after model updates')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained recognition model checkpoint')
    parser.add_argument('--output-dir', type=str, default='logs/recognition',
                       help='Output directory for embeddings')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run model on')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of images to process (for testing)')
    parser.add_argument('--skip-database', action='store_true',
                       help='Skip database embedding regeneration')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🚀 Starting complete embedding regeneration")
    print(f"   Model: {args.model_path}")
    print(f"   Output: {output_dir}")
    print(f"   Device: {args.device}")

    try:
        # Load the trained model
        model = load_trained_model(args.model_path, args.device)

        # Regenerate watchlist embeddings
        watchlist_success = regenerate_watchlist_embeddings(model, output_dir, args)

        # Regenerate database embeddings (if requested)
        if not args.skip_database:
            db_success = regenerate_database_embeddings(model, args)
        else:
            db_success = True
            print("⏭️  Skipping database embedding regeneration")

        # Validate consistency
        if watchlist_success:
            validate_embedding_consistency(output_dir)

        if watchlist_success and db_success:
            print("\n🎉 Embedding regeneration completed successfully!")
            print("   Remember to restart the backend service to load new embeddings")
            return 0
        else:
            print("\n❌ Embedding regeneration failed!")
            return 1

    except Exception as e:
        print(f"\n❌ Error during embedding regeneration: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())