#!/usr/bin/env python3
"""
Comprehensive Model Comparison: Classifier vs Embedding Evaluation
Compares Recognition Model and Buffalo using both evaluation methods
"""

import numpy as np
import torch
import json
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from evaluation.evaluate_recognition import evaluate_all_metrics
from training.recognition_train import RecognitionNet
from training.utils.dataset_loader import get_recognition_dataset
from torch.utils.data import DataLoader
from training.utils.universal_pipeline import DatasetSplitter

def load_recognition_model():
    """Load the trained Recognition Model for classifier evaluation"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint first to infer shapes
    checkpoint = torch.load('logs/recognition/recognition_model_best.pth', map_location=device)
    sd = checkpoint.get('model_state_dict', checkpoint)

    # Infer num_classes from classifier.weight shape
    if 'classifier.weight' in sd:
        num_classes_ckpt = sd['classifier.weight'].shape[0]
    else:
        num_classes_ckpt = 530  # fallback

    # Infer feature_dim if feature_selector is present
    feature_dim = None
    if 'feature_selector_layer.weight' in sd:
        feature_dim = sd['feature_selector_layer.weight'].shape[0]

    # Load model architecture with inferred params
    model = RecognitionNet(num_classes=num_classes_ckpt, feature_dim=feature_dim, head_type='linear')
    model.to(device)

    # Load state dict
    model.load_state_dict(sd, strict=False)
    model.eval()

    return model, device

def evaluate_with_classifier(model, device, data_source):
    """Evaluate using the trained classifier (cross-entropy approach).

    `data_source` may be either a path (str) to a dataset folder or a PyTorch
    `Dataset` instance. This allows passing a pre-split test Dataset when the
    data lives under a `train/` folder and no explicit `test/` split exists on disk.
    """
    # Load or use provided dataset
    if hasattr(data_source, '__len__') and hasattr(data_source, '__getitem__'):
        test_dataset = data_source
    else:
        test_dataset = get_recognition_dataset(data_dir=data_source, split=None)

    # Handle empty dataset gracefully
    if len(test_dataset) == 0:
        print(f"No test images found in '{data_source}'. Skipping classifier evaluation.")
        # Return metrics with None to indicate unavailable
        return {
            'accuracy': None,
            'precision_macro': None,
            'recall_macro': None,
            'f1_macro': None,
            'precision_micro': None,
            'recall_micro': None,
            'f1_micro': None,
            'balanced_accuracy': None,
            'kappa': None,
            'mcc': None
        }

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)  # Get logits from classifier
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(targets.numpy())

    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score
    )

    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision_macro': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall_macro': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'precision_micro': precision_score(all_labels, all_preds, average='micro', zero_division=0),
        'recall_micro': recall_score(all_labels, all_preds, average='micro', zero_division=0),
        'f1_micro': f1_score(all_labels, all_preds, average='micro', zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(all_labels, all_preds),
        'kappa': cohen_kappa_score(all_labels, all_preds),
        'mcc': matthews_corrcoef(all_labels, all_preds)
    }

    return metrics

def main():
    print("="*80)
    print("COMPREHENSIVE MODEL COMPARISON: CLASSIFIER vs EMBEDDING EVALUATION")
    print("="*80)

    # Load embeddings for k-NN evaluation
    print("\nLoading embeddings for k-NN evaluation...")
    recognition_embeddings = np.load('logs/recognition/test_embeddings.npy')
    recognition_labels = np.load('logs/recognition/test_labels.npy')

    buffalo_embeddings = np.load('logs/buffalo/embeddings.npy')
    buffalo_labels = np.load('logs/buffalo/labels.npy')

    print(f"Recognition Model: {recognition_embeddings.shape} embeddings")
    print(f"Buffalo Model: {buffalo_embeddings.shape} embeddings")

    # Method 1: k-NN on Embeddings
    print("\n" + "-"*60)
    print("METHOD 1: k-NN CLASSIFICATION ON EMBEDDINGS")
    print("-"*60)

    print("Evaluating Recognition Model (k-NN)...")
    recognition_knn_metrics = evaluate_all_metrics(recognition_embeddings, recognition_labels)

    print("Evaluating Buffalo Model (k-NN)...")
    buffalo_knn_metrics = evaluate_all_metrics(buffalo_embeddings, buffalo_labels)

    print("\nk-NN Comparison:")
    for key in recognition_knn_metrics:
        if key in buffalo_knn_metrics and isinstance(recognition_knn_metrics[key], (int, float)):
            diff = recognition_knn_metrics[key] - buffalo_knn_metrics[key]
            print(f"{key}: Recognition {recognition_knn_metrics[key]:.4f} vs Buffalo {buffalo_knn_metrics[key]:.4f} (diff: {diff:+.4f})")

    # Method 2: Trained Classifier (for Recognition Model only)
    print("\n" + "-"*60)
    print("METHOD 2: TRAINED CLASSIFIER EVALUATION")
    print("-"*60)

    print("Loading Recognition Model for classifier evaluation...")
    try:
        model, device = load_recognition_model()
        # Pick and prepare the proper dataset for evaluation. Prefer an on-disk
        # `test/` folder; if not present but `train/` exists, create a test split
        # from the `train/` data using the same splitter used during training.
        base_dir = "Data/recognition_faces"
        test_dir = os.path.join(base_dir, 'test')
        if os.path.isdir(test_dir):
            recognition_classifier_metrics = evaluate_with_classifier(model, device, test_dir)
        else:
            # Fallback: if there's a train subfolder, split it to get a test set
            train_dir = os.path.join(base_dir, 'train')
            if os.path.isdir(train_dir):
                print(f"No explicit test folder found. Creating test split from '{train_dir}' using DatasetSplitter.")
                combined_ds = get_recognition_dataset(data_dir=train_dir, split=None)
                # Use same ratios and seed as training pipeline
                _, _, test_dataset = DatasetSplitter.split_dataset(combined_ds, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
                recognition_classifier_metrics = evaluate_with_classifier(model, device, test_dataset)
            else:
                # Last resort: try base_dir directly
                recognition_classifier_metrics = evaluate_with_classifier(model, device, base_dir)

        print("Recognition Model (Classifier):")
        for key, value in recognition_classifier_metrics.items():
            if value is None:
                print(f"  {key}: n/a")
            else:
                print(f"  {key}: {value:.4f}")

        print("\nClassifier vs k-NN Comparison for Recognition Model:")
        for key in recognition_classifier_metrics:
            if key in recognition_knn_metrics:
                knn_val = recognition_knn_metrics[key]
                clf_val = recognition_classifier_metrics[key]
                if clf_val is None:
                    print(f"{key}: k-NN {knn_val:.4f} vs Classifier n/a (diff: n/a)")
                else:
                    diff = knn_val - clf_val
                    print(f"{key}: k-NN {knn_val:.4f} vs Classifier {clf_val:.4f} (diff: {diff:+.4f})")

    except Exception as e:
        print(f"Error loading classifier model: {e}")
        print("Skipping classifier evaluation...")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print("Key Findings:")
    print(f"• Recognition Model k-NN accuracy: {recognition_knn_metrics['accuracy']:.1%}")
    if 'recognition_classifier_metrics' in locals():
        acc = recognition_classifier_metrics.get('accuracy')
        if acc is None:
            print("• Recognition Model classifier accuracy: n/a")
            print("• Embeddings outperform classifier by: n/a")
        else:
            print(f"• Recognition Model classifier accuracy: {acc:.1%}")
            print(f"• Embeddings outperform classifier by: {(recognition_knn_metrics['accuracy'] - acc)*100:.1f} percentage points")
    print(f"• Buffalo Model k-NN accuracy: {buffalo_knn_metrics['accuracy']:.1%}")
    print(f"• Recognition Model outperforms Buffalo by: {(recognition_knn_metrics['accuracy'] - buffalo_knn_metrics['accuracy'])*100:.1f} percentage points")

    print("\nInterpretation:")
    print("• k-NN measures embedding quality (feature learning)")
    print("• Classifier measures end-to-end performance (feature learning + classification)")
    print("• Higher k-NN scores indicate better embeddings")
    print("• Gap between k-NN and classifier shows classifier head limitations")

    # Save results
    results = {
        'recognition_model': {
            'knn_metrics': recognition_knn_metrics,
            'embedding_dim': recognition_embeddings.shape[1]
        },
        'buffalo_model': {
            'knn_metrics': buffalo_knn_metrics,
            'embedding_dim': buffalo_embeddings.shape[1]
        }
    }

    if 'recognition_classifier_metrics' in locals():
        results['recognition_model']['classifier_metrics'] = recognition_classifier_metrics

    os.makedirs('logs/comparison', exist_ok=True)
    with open('logs/comparison/comprehensive_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("✓ Saved comprehensive results to logs/comparison/comprehensive_comparison.json")
if __name__ == "__main__":
    main()