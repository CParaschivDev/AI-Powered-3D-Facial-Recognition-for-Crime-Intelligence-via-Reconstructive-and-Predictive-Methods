#!/usr/bin/env python3
"""
Regenerate watchlist/gallery embeddings using the current recognition model.

Saves:
- logs/recognition/watchlist_embeddings_v2.npy
- logs/recognition/watchlist_labels_v2.json

Run as: python scripts/refresh_watchlist_embeddings.py
"""
from pathlib import Path
import json
import numpy as np
import cv2
import argparse
import math
import torch
import sys

from backend.models.utils.model_loader import get_recognition_model
from backend.core.paths import watchlist_roots
from backend.utils.face_io import iter_images, id_from_path, read_image


OUT_DIR = Path("logs") / "recognition"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EMB_PATH = OUT_DIR / "watchlist_embeddings_v2.npy"
LABELS_PATH = OUT_DIR / "watchlist_labels_v2.json"


def main():
    parser = argparse.ArgumentParser(description='Regenerate watchlist embeddings')
    parser.add_argument('--limit', type=int, default=None, help='Maximum number of images to process (for quick tests)')
    parser.add_argument('--per-identity-limit', type=int, default=None, help='Max images to use per identity (helps sample across identities)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for torch inference')
    parser.add_argument('--progress-every', type=int, default=50, help='Print progress every N processed images')
    args = parser.parse_args()

    rm = get_recognition_model()
    id_to_vecs = {}
    total_images = 0

    roots = watchlist_roots()
    if not roots:
        print("No watchlist roots found. Check DATA_ROOT and WATCHLIST_DIRS in config.")
        return 1

    # Stream identities one-by-one to avoid large memory use
    use_batch = getattr(rm, 'torch_model', None) is not None
    batch_size = max(1, int(args.batch_size))
    processed = 0
    progress_every = int(args.progress_every) if args.progress_every else 0
    print(f"Starting embedding run (streaming) (batch_size={batch_size})", flush=True)

    stop_requested = False
    for root in roots:
        # Group files by identity for this root only (keeps memory bounded)
        per_id = {}
        for p in iter_images(root):
            pid = id_from_path(root, p)
            per_id.setdefault(pid, []).append(p)

        # Process each identity's files in turn
        for pid, files in per_id.items():
            if args.per_identity_limit is not None and args.per_identity_limit > 0:
                files = files[: args.per_identity_limit]

            # Process files for this identity in batches
            i = 0
            while i < len(files):
                end = min(len(files), i + batch_size)
                batch_files = files[i:end]
                imgs = []
                pids = []
                for fp in batch_files:
                    if args.limit is not None and args.limit > 0 and total_images >= args.limit:
                        stop_requested = True
                        break
                    img = read_image(fp)
                    if img is None:
                        continue
                    imgs.append(img)
                    pids.append(pid)

                if stop_requested:
                    break

                if not imgs:
                    i = end
                    continue

                if use_batch:
                    try:
                        device = next(rm.torch_model.parameters()).device
                    except Exception:
                        device = torch.device('cpu')

                    tensors = []
                    for img in imgs:
                        rgb = img[..., ::-1]
                        rsz = cv2.resize(rgb, (112, 112))
                        t = torch.from_numpy(rsz.astype('float32').transpose(2, 0, 1) / 255.0)
                        tensors.append(t)

                    batch_t = torch.stack(tensors, dim=0).to(device)
                    with torch.no_grad():
                        emb_t = rm.torch_model(batch_t, return_embedding=True)
                    emb_np = emb_t.cpu().numpy()
                    norms = np.linalg.norm(emb_np, axis=1, keepdims=True) + 1e-12
                    emb_np = (emb_np / norms).astype('float32')

                    for emb in emb_np:
                        id_to_vecs.setdefault(pid, []).append(emb)
                        total_images += 1
                        processed += 1
                        if progress_every and (processed % progress_every) == 0:
                            print(f"Processed {processed} images; identities so far: {len(id_to_vecs)}", flush=True)
                else:
                    for img in imgs:
                        emb = rm.extract_fused_embedding(img)
                        if emb is None:
                            continue
                        id_to_vecs.setdefault(pid, []).append(emb.astype('float32'))
                        total_images += 1
                        processed += 1
                        if progress_every and (processed % progress_every) == 0:
                            print(f"Processed {processed} images; identities so far: {len(id_to_vecs)}", flush=True)

                i = end

            if stop_requested:
                break

        if stop_requested:
            break

    # Build means
    ids = []
    means = []
    for pid, vecs in sorted(id_to_vecs.items()):
        if not vecs:
            continue
        mean = np.stack(vecs).mean(axis=0)
        # L2 normalise
        mean = mean / (np.linalg.norm(mean) + 1e-12)
        ids.append(pid)
        means.append(mean.astype('float32'))

    if not means:
        print("No embeddings generated. Exiting.")
        return 1

    embeddings = np.stack(means)
    np.save(EMB_PATH, embeddings)
    LABELS_PATH.write_text(json.dumps(ids, indent=2))

    print(f"Saved embeddings for {len(ids)} identities ({total_images} images used) -> {EMB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
