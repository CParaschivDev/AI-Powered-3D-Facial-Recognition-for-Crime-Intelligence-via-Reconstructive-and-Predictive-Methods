from pathlib import Path
import numpy as np
import cv2
import random
import logging
from typing import Dict, List, Tuple
from backend.core.paths import mine_root, aflw_root, watchlist_roots
from backend.models.utils.model_loader import get_recognition_model
from backend.database.dependencies import SessionLocal
from backend.database.db_utils import load_watchlist_means
from backend.services.matcher import best_match, cosine
from backend.utils.face_io import iter_images, read_image

logger = logging.getLogger(__name__)

def _get_test_samples(max_watchlist: int = 5, max_aflw: int = 5, max_mine: int = 5) -> Dict[str, List[np.ndarray]]:
    """
    Get positive and negative test samples for threshold tuning.
    
    Args:
        max_watchlist: Maximum number of held-out positive samples per identity
        max_aflw: Maximum number of negative samples from AFLW dataset
        max_mine: Maximum number of negative samples from mine dataset
        
    Returns:
        Dictionary with 'positives' and 'negatives' lists of embeddings
    """
    rec_model = get_recognition_model()
    
    # Dictionary to store embeddings
    samples = {
        'positives': [],  # Embeddings of watchlist faces not used for mean calculation
        'negatives': []   # Embeddings of non-watchlist faces (mine + AFLW)
    }
    
    # Get watchlist identities to select held-out samples
    db = SessionLocal()
    watchlist_ids = list(load_watchlist_means(db).keys())
    db.close()
    
    if not watchlist_ids:
        logger.warning("No watchlist identities found. Build the watchlist first.")
        return samples
        
    # Collect positive (watchlist) samples for testing
    # These should be different from the ones used to build the mean vectors
    held_out = {}
    for root in watchlist_roots():
        for p in iter_images(root):
            identity = p.parts[-2]  # Assuming the identity is the parent folder name
            if identity in watchlist_ids:
                held_out.setdefault(identity, []).append(p)
    
    # For each identity, take some held-out samples
    for identity, paths in held_out.items():
        if len(paths) <= 20:  # If we have few samples, we probably used all for the mean
            continue
        
        # Use samples beyond the first 20 (which were likely used for mean calculation)
        test_paths = paths[20:20+max_watchlist]
        for path in test_paths:
            try:
                img = read_image(path)
                embedding = rec_model.embed(img)
                if embedding is not None:
                    samples['positives'].append(embedding)
            except Exception as e:
                logger.warning(f"Failed to process positive sample {path}: {e}")
    
    # Collect negative (mine) samples
    mine_paths = list(iter_images(mine_root()))
    random.shuffle(mine_paths)
    for path in mine_paths[:max_mine]:
        try:
            img = read_image(path)
            embedding = rec_model.embed(img)
            if embedding is not None:
                samples['negatives'].append(embedding)
        except Exception as e:
            logger.warning(f"Failed to process mine sample {path}: {e}")
    
    # Collect negative (AFLW) samples if available
    try:
        aflw_paths = list(iter_images(aflw_root()))
        random.shuffle(aflw_paths)
        for path in aflw_paths[:max_aflw]:
            try:
                img = read_image(path)
                embedding = rec_model.embed(img)
                if embedding is not None:
                    samples['negatives'].append(embedding)
            except Exception as e:
                logger.warning(f"Failed to process AFLW sample {path}: {e}")
    except Exception as e:
        logger.warning(f"Failed to access AFLW dataset: {e}")
    
    logger.info(f"Collected {len(samples['positives'])} positive and {len(samples['negatives'])} negative samples for threshold tuning")
    return samples

def tune_threshold(max_samples: int = 10) -> Dict:
    """
    Tune the threshold for face recognition using held-out samples.
    
    Args:
        max_samples: Maximum number of samples to use per category
        
    Returns:
        Dictionary with:
            - suggested_threshold: The recommended threshold value
            - min_positive: The minimum similarity score for positive samples
            - max_negative: The maximum similarity score for negative samples
            - pos_samples: Number of positive samples used
            - neg_samples: Number of negative samples used
            - gap: The gap between min_positive and max_negative
    """
    # Get test samples
    samples = _get_test_samples(max_watchlist=max_samples, max_aflw=max_samples, max_mine=max_samples)
    
    if not samples['positives'] or not samples['negatives']:
        logger.warning("Insufficient samples for threshold tuning")
        return {
            'suggested_threshold': 0.5,  # Default
            'min_positive': None,
            'max_negative': None,
            'pos_samples': len(samples['positives']),
            'neg_samples': len(samples['negatives']),
            'gap': None,
            'error': "Insufficient samples for tuning"
        }
    
    # Get watchlist means
    db = SessionLocal()
    means = load_watchlist_means(db)
    db.close()
    
    if not means:
        logger.warning("No watchlist means found. Build the watchlist first.")
        return {
            'suggested_threshold': 0.5,  # Default
            'min_positive': None,
            'max_negative': None,
            'pos_samples': len(samples['positives']),
            'neg_samples': len(samples['negatives']),
            'gap': None,
            'error': "Watchlist is empty"
        }
    
    # Calculate similarity scores for positive samples
    positive_scores = []
    for embedding in samples['positives']:
        pair = best_match(embedding)
        if pair:
            positive_scores.append(pair[1])  # The score from the best match
    
    # Calculate similarity scores for negative samples
    negative_scores = []
    for embedding in samples['negatives']:
        pair = best_match(embedding)
        if pair:
            negative_scores.append(pair[1])
    
    if not positive_scores or not negative_scores:
        logger.warning("Failed to calculate similarity scores")
        return {
            'suggested_threshold': 0.5,  # Default
            'min_positive': None,
            'max_negative': None,
            'pos_samples': len(samples['positives']),
            'neg_samples': len(samples['negatives']),
            'gap': None,
            'error': "Failed to calculate similarity scores"
        }
    
    # Find min positive and max negative scores
    min_positive = min(positive_scores)
    max_negative = max(negative_scores)
    
    # Calculate suggested threshold in the middle of the gap
    if min_positive > max_negative:
        # Ideal case: clear separation between positives and negatives
        suggested_threshold = (min_positive + max_negative) / 2
        gap = min_positive - max_negative
    else:
        # Overlap case: use a threshold that minimizes false positives
        # (prefer to miss some wanted persons than falsely identify)
        suggested_threshold = min_positive
        gap = 0.0
    
    return {
        'suggested_threshold': float(suggested_threshold),
        'min_positive': float(min_positive),
        'max_negative': float(max_negative),
        'pos_samples': len(positive_scores),
        'neg_samples': len(negative_scores),
        'gap': float(gap)
    }