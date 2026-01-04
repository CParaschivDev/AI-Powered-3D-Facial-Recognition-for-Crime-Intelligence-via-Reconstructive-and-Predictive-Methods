"""
Forensic scoring utilities for image quality and prediction confidence.
Part of GEN-4 AI Copilot system.
"""
import numpy as np
import cv2
from typing import List, Dict, Any


def compute_image_quality_score(image: np.ndarray) -> float:
    """
    Compute image quality score based on blur and brightness.
    
    Args:
        image: BGR image (H, W, 3)
    
    Returns:
        Quality score in range [0, 1] where 1 is best quality
    """
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 1. Laplacian variance (blur detection)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Normalize: typical good images have variance > 100, blurry < 50
        blur_score = min(1.0, laplacian_var / 100.0)
        
        # 2. Brightness deviation from optimal (around 128)
        mean_brightness = np.mean(gray)
        brightness_deviation = abs(mean_brightness - 128.0) / 128.0
        brightness_score = 1.0 - min(1.0, brightness_deviation)
        
        # Combined score (weighted average)
        quality_score = 0.7 * blur_score + 0.3 * brightness_score
        
        return float(np.clip(quality_score, 0.0, 1.0))
    
    except Exception as e:
        print(f"WARN: Image quality scoring failed: {e}")
        return 0.5  # Neutral score on failure


def compute_prediction_entropy(similarities: List[float], epsilon: float = 1e-10) -> float:
    """
    Compute Shannon entropy of similarity distribution.
    Lower entropy = more confident (one clear match)
    Higher entropy = uncertain (many similar matches)
    
    Args:
        similarities: List of similarity scores [0, 1]
        epsilon: Small constant to avoid log(0)
    
    Returns:
        Normalized entropy in range [0, 1] where 0 is most confident
    """
    try:
        if not similarities or len(similarities) == 0:
            return 1.0  # Maximum uncertainty
        
        # Convert to probability distribution
        similarities = np.array(similarities)
        similarities = np.clip(similarities, 0.0, 1.0)
        
        # Normalize to sum to 1
        total = np.sum(similarities) + epsilon
        probabilities = similarities / total
        
        # Shannon entropy: H = -Σ p(x) * log(p(x))
        entropy = -np.sum(probabilities * np.log(probabilities + epsilon))
        
        # Normalize by maximum possible entropy (uniform distribution)
        max_entropy = np.log(len(similarities))
        if max_entropy > 0:
            normalized_entropy = entropy / max_entropy
        else:
            normalized_entropy = 0.0
        
        return float(np.clip(normalized_entropy, 0.0, 1.0))
    
    except Exception as e:
        print(f"WARN: Entropy calculation failed: {e}")
        return 0.5  # Neutral score on failure


def annotate_wanted_status(matches: List[Dict[str, Any]], wanted_ids: set) -> List[Dict[str, Any]]:
    """
    Annotate match objects with wanted/not-wanted flag.
    
    Args:
        matches: List of match dictionaries with person_id
        wanted_ids: Set of person IDs that are wanted
    
    Returns:
        Updated matches with 'wanted' boolean field
    """
    for match in matches:
        person_id = match.get('person_id')
        match['wanted'] = person_id in wanted_ids if person_id is not None else False
    
    return matches
