from pathlib import Path
import numpy as np, cv2
from backend.core.paths import watchlist_roots
from backend.models.utils.model_loader import get_recognition_model
from backend.database.dependencies import SessionLocal
from backend.database.db_utils import upsert_watchlist_identity
from backend.utils.face_io import iter_images, id_from_path, read_image
import logging

logger = logging.getLogger(__name__)
import os
from backend.core.config import settings

def build_watchlist(max_per_id: int = 20) -> dict:
    # Check if DATA_ROOT is configured
    if not settings.DATA_ROOT:
        raise ValueError("DATA_ROOT not configured. Please set it in .env file.")
    
    rm = get_recognition_model()
    id_to_vecs = {}
    for root in watchlist_roots():
        # group paths per ID
        per_id = {}
        for p in iter_images(root):
            pid = id_from_path(root, p)
            per_id.setdefault(pid, []).append(p)
        # cap and embed
        for pid, files in per_id.items():
            files = files[:max_per_id]
            for fp in files:
                img = read_image(fp)
                if img is None:
                    logger.debug(f"Could not read image, skipping: {fp}")
                    continue
                e = rm.embed(img)
                if e is None:
                    logger.debug(f"Embedding failed for image, skipping: {fp}")
                    continue
                id_to_vecs.setdefault(pid, []).append(e)
    # mean & store
    db = SessionLocal()
    saved = 0
    for pid, vecs in id_to_vecs.items():
        if not vecs: continue
        mean = np.stack(vecs).mean(axis=0).astype("float32")
        upsert_watchlist_identity(db, pid, mean)
        saved += 1
    db.close()
    return {"identities": saved}
