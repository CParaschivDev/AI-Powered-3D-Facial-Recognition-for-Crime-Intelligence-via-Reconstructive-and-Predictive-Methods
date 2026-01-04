"""
Bias Monitoring Service

Computes fairness metrics across demographic groups to detect bias in facial recognition.
Uses FaceScrub actor/actress split as gender proxy for demographic analysis.
"""

import numpy as np
import warnings
import os
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime, timezone
import logging

# Suppress sklearn warnings about many classes in multi-class classification
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

logger = logging.getLogger(__name__)


class BiasMonitor:
    """
    Monitors and computes bias metrics for facial recognition system.
    """
    
    def __init__(self, threshold: float = 0.05):
        """
        Initialize bias monitor.
        
        Args:
            threshold: Acceptable bias threshold (default 5% accuracy gap)
        """
        self.threshold = threshold
    
    def compute_group_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        group_labels: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute accuracy, precision, recall, F1 for each demographic group.
        
        Args:
            predictions: Model predictions (N,)
            labels: Ground truth labels (N,)
            group_labels: Demographic group for each sample (N,) - e.g., 0=male, 1=female
            
        Returns:
            Dictionary with metrics per group
        """
        unique_groups = np.unique(group_labels)
        group_metrics = {}
        
        for group in unique_groups:
            mask = group_labels == group
            group_preds = predictions[mask]
            group_true = labels[mask]
            
            if len(group_true) == 0:
                continue
            
            group_name = f"group_{group}"
            group_metrics[group_name] = {
                "accuracy": float(accuracy_score(group_true, group_preds)),
                "precision": float(precision_score(group_true, group_preds, average='macro', zero_division=0)),
                "recall": float(recall_score(group_true, group_preds, average='macro', zero_division=0)),
                "f1": float(f1_score(group_true, group_preds, average='macro', zero_division=0)),
                "sample_count": int(np.sum(mask))
            }
        
        return group_metrics
    
    def compute_demographic_parity(
        self,
        predictions: np.ndarray,
        group_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute demographic parity: positive prediction rate should be similar across groups.
        
        Args:
            predictions: Binary predictions (N,)
            group_labels: Demographic group labels (N,)
            
        Returns:
            Dictionary with positive rates per group
        """
        unique_groups = np.unique(group_labels)
        parity_metrics = {}
        
        for group in unique_groups:
            mask = group_labels == group
            group_preds = predictions[mask]
            positive_rate = float(np.mean(group_preds > 0)) if len(group_preds) > 0 else 0.0
            parity_metrics[f"group_{group}_positive_rate"] = positive_rate
        
        return parity_metrics
    
    def compute_equalized_odds(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        group_labels: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute equalized odds: TPR and FPR should be similar across groups.
        
        Args:
            predictions: Model predictions (N,)
            labels: Ground truth labels (N,)
            group_labels: Demographic group labels (N,)
            
        Returns:
            Dictionary with TPR and FPR per group
        """
        unique_groups = np.unique(group_labels)
        odds_metrics = {}
        
        for group in unique_groups:
            mask = group_labels == group
            group_preds = predictions[mask]
            group_true = labels[mask]
            
            if len(group_true) == 0:
                continue
            
            # True Positive Rate (TPR) - for positive class
            tp = np.sum((group_preds == group_true) & (group_true == 1))
            fn = np.sum((group_preds != group_true) & (group_true == 1))
            tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            
            # False Positive Rate (FPR) - for negative class
            fp = np.sum((group_preds != group_true) & (group_true == 0))
            tn = np.sum((group_preds == group_true) & (group_true == 0))
            fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            
            odds_metrics[f"group_{group}"] = {
                "tpr": tpr,
                "fpr": fpr
            }
        
        return odds_metrics
    
    def compute_bias_score(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        group_labels: np.ndarray,
        metric: str = "accuracy"
    ) -> Tuple[float, bool, Dict]:
        """
        Compute overall bias score as maximum gap between groups.
        
        Args:
            predictions: Model predictions (N,)
            labels: Ground truth labels (N,)
            group_labels: Demographic group labels (N,)
            metric: Metric to use for bias computation (accuracy, precision, recall, f1)
            
        Returns:
            Tuple of (bias_score, is_flagged, detailed_metrics)
        """
        group_metrics = self.compute_group_metrics(predictions, labels, group_labels)
        
        if len(group_metrics) < 2:
            logger.warning("Less than 2 groups found, cannot compute bias score")
            return 0.0, False, group_metrics
        
        # Extract the chosen metric for each group
        metric_values = [metrics[metric] for metrics in group_metrics.values()]
        
        # Bias score is the maximum gap between any two groups
        bias_score = float(np.max(metric_values) - np.min(metric_values))
        is_flagged = bias_score > self.threshold
        
        detailed_metrics = {
            "bias_score": bias_score,
            "is_flagged": is_flagged,
            "threshold": self.threshold,
            "metric_used": metric,
            "group_metrics": group_metrics,
            "max_value": float(np.max(metric_values)),
            "min_value": float(np.min(metric_values))
        }
        
        return bias_score, is_flagged, detailed_metrics
    
    def generate_bias_report(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        group_labels: np.ndarray,
        model_name: str = "RecognitionNet",
        group_names: Optional[Dict[int, str]] = None
    ) -> Dict:
        """
        Generate comprehensive bias report.
        
        Args:
            predictions: Model predictions (N,)
            labels: Ground truth labels (N,)
            group_labels: Demographic group labels (N,)
            model_name: Name of the model being evaluated
            group_names: Optional mapping of group IDs to human-readable names
            
        Returns:
            Comprehensive bias report dictionary
        """
        # Compute all fairness metrics
        bias_score, is_flagged, detailed_metrics = self.compute_bias_score(
            predictions, labels, group_labels
        )
        
        demographic_parity = self.compute_demographic_parity(predictions, group_labels)
        equalized_odds = self.compute_equalized_odds(predictions, labels, group_labels)
        
        # Map group names if provided
        if group_names:
            for old_key in list(detailed_metrics["group_metrics"].keys()):
                group_id = int(old_key.split("_")[1])
                if group_id in group_names:
                    new_key = group_names[group_id]
                    detailed_metrics["group_metrics"][new_key] = detailed_metrics["group_metrics"].pop(old_key)
        
        report = {
            "model_name": model_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_bias_score": bias_score,
            "is_flagged": is_flagged,
            "threshold": self.threshold,
            "total_samples": len(predictions),
            "num_groups": len(np.unique(group_labels)),
            "fairness_metrics": {
                "accuracy_gap": detailed_metrics,
                "demographic_parity": demographic_parity,
                "equalized_odds": equalized_odds
            },
            "recommendation": self._get_recommendation(bias_score, is_flagged)
        }
        
        return report
    
    def _get_recommendation(self, bias_score: float, is_flagged: bool) -> str:
        """Generate recommendation based on bias score."""
        if not is_flagged:
            return "Model shows acceptable fairness across demographic groups. Continue monitoring."
        elif bias_score < 0.10:
            return "Model shows moderate bias. Consider re-balancing training data or applying fairness constraints."
        else:
            return "Model shows significant bias. Immediate review required. Consider retraining with balanced data or fairness-aware algorithms."


def _predict_from_embeddings(embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Fallback: Predict using nearest neighbor on embeddings.
    
    Args:
        embeddings: Feature embeddings (N, D)
        labels: Ground truth labels (N,)
        
    Returns:
        Predicted labels (N,)
    """
    unique_labels = np.unique(labels)
    class_prototypes = {}
    
    for label in unique_labels:
        class_mask = labels == label
        class_prototypes[label] = embeddings[class_mask].mean(axis=0)
    
    predictions = []
    for emb in embeddings:
        min_dist = float('inf')
        pred_label = unique_labels[0]
        
        for label, prototype in class_prototypes.items():
            dist = np.linalg.norm(emb - prototype)
            if dist < min_dist:
                min_dist = dist
                pred_label = label
        
        predictions.append(pred_label)
    
    return np.array(predictions)


def evaluate_facescrub_gender_bias(
    embeddings_path: str,
    labels_path: str,
    actor_indices_path: str,
    actress_indices_path: str,
    output_path: Optional[str] = None,
    model_path: Optional[str] = None,
    test_images_dir: Optional[str] = None
) -> Dict:
    """
    Evaluate gender bias using FaceScrub actor/actress split.
    
    Args:
        embeddings_path: Path to embeddings.npy (for fallback)
        labels_path: Path to labels.npy
        actor_indices_path: Path to file with actor sample indices
        actress_indices_path: Path to file with actress sample indices
        output_path: Optional path to save JSON report
        model_path: Path to trained recognition model (.pth)
        test_images_dir: Path to test images directory
        
    Returns:
        Bias report dictionary
    """
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from PIL import Image
    import glob
    
    # Load labels and indices
    labels = np.load(labels_path)
    actor_indices = np.load(actor_indices_path)
    actress_indices = np.load(actress_indices_path)
    
    # Create group labels: 0=actor (male), 1=actress (female)
    group_labels = np.zeros(len(labels), dtype=int)
    group_labels[actress_indices] = 1
    
    # Get predictions from model
    if model_path and os.path.exists(model_path):
        print(f"  ✓ Loading trained model: {model_path}")
        logger.info(f"Loading model from {model_path}")
        
        # Define RecognitionNet (must match training architecture)
        class RecognitionNet(nn.Module):
            def __init__(self, num_classes, embedding_size=512):
                super(RecognitionNet, self).__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 64, 7, stride=1, padding=3, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
                )
                self.adaptive_pool = nn.AdaptiveAvgPool2d((112, 112))
                self.embedding_layer = nn.Linear(64 * 112 * 112, embedding_size)
                self.classifier = nn.Linear(embedding_size, num_classes)
            
            def forward(self, x):
                x = self.backbone(x)
                x = self.adaptive_pool(x)
                x = x.view(x.size(0), -1)
                embedding = self.embedding_layer(x)
                logits = self.classifier(embedding)
                return logits
        
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        num_classes = checkpoint.get('num_classes', 530)
        
        model = RecognitionNet(num_classes=num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"  ✓ Model loaded with {num_classes} classes")
        print(f"  ✓ Using model's classifier on pre-computed embeddings")
        
        # Load embeddings and pass through classifier layer
        embeddings = np.load(embeddings_path)
        embeddings_tensor = torch.from_numpy(embeddings).float()
        
        predictions = []
        with torch.no_grad():
            # Process in batches for efficiency
            batch_size = 128
            for i in range(0, len(embeddings_tensor), batch_size):
                if i % 500 == 0:
                    print(f"  Processing batch {i}/{len(embeddings_tensor)}...")
                batch = embeddings_tensor[i:i+batch_size]
                logits = model.classifier(batch)
                preds = torch.argmax(logits, dim=1).numpy()
                predictions.extend(preds)
        
        print(f"  ✓ Generated {len(predictions)} predictions from model classifier")
        predictions = np.array(predictions)
    else:
        # Fallback to nearest neighbor on embeddings
        print(f"  ⚠️  Using nearest neighbor fallback on embeddings")
        print(f"     Model path exists: {model_path and os.path.exists(model_path) if model_path else False}")
        print(f"     Test images dir exists: {test_images_dir and os.path.exists(test_images_dir) if test_images_dir else False}")
        logger.info("Model not available, using nearest neighbor on embeddings")
        embeddings = np.load(embeddings_path)
        predictions = _predict_from_embeddings(embeddings, labels)
    
    # Initialize bias monitor
    monitor = BiasMonitor(threshold=0.05)
    
    # Generate report
    group_names = {0: "Actors (Male)", 1: "Actresses (Female)"}
    report = monitor.generate_bias_report(
        predictions=predictions,
        labels=labels,
        group_labels=group_labels,
        model_name="RecognitionNet",
        group_names=group_names
    )
    
    # Save report if output path provided
    if output_path:
        import json
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Bias report saved to {output_path}")
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute bias metrics for facial recognition")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to embeddings.npy")
    parser.add_argument("--labels", type=str, required=True, help="Path to labels.npy")
    parser.add_argument("--actor-indices", type=str, required=True, help="Path to actor indices")
    parser.add_argument("--actress-indices", type=str, required=True, help="Path to actress indices")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    
    args = parser.parse_args()
    
    report = evaluate_facescrub_gender_bias(
        embeddings_path=args.embeddings,
        labels_path=args.labels,
        actor_indices_path=args.actor_indices,
        actress_indices_path=args.actress_indices,
        output_path=args.output
    )
    
    print("\n=== BIAS MONITORING REPORT ===")
    print(f"Model: {report['model_name']}")
    print(f"Bias Score: {report['overall_bias_score']:.4f}")
    print(f"Flagged: {report['is_flagged']}")
    print(f"\nRecommendation: {report['recommendation']}")
