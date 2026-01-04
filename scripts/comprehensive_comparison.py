#!/usr/bin/env python3
"""
Comprehensive model comparison: classifier vs embedding (k-NN) evaluation.

- Loads recognition embeddings from `logs/recognition/test_embeddings.npy` and `test_labels.npy`
- Loads Buffalo embeddings from `logs/buffalo/embeddings.npy` and `labels.npy`
- Runs k-NN evaluation using Leave-One-Out (LOO) by default (safe fallback to StratifiedKFold)
- Loads the trained recognition classifier checkpoint and runs `evaluate_classification` on the test split
- Prints side-by-side comparison and saves results to `logs/comparison/comprehensive_comparison.json`
"""

import argparse
import os
from pathlib import Path
import numpy as np
import json
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef
)
import torch
import sys

# Add repo root to path so local imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.utils.dataset_loader import get_recognition_dataset
from training.utils.universal_pipeline import DatasetSplitter
from training.recognition_train import RecognitionNet
from training.utils.evaluation_metrics import evaluate_classification


def compute_metrics(y_true, y_pred):
    metrics = {}
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['recall_macro'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['precision_micro'] = float(precision_score(y_true, y_pred, average='micro', zero_division=0))
    metrics['recall_micro'] = float(recall_score(y_true, y_pred, average='micro', zero_division=0))
    metrics['f1_micro'] = float(f1_score(y_true, y_pred, average='micro', zero_division=0))
    metrics['balanced_accuracy'] = float(balanced_accuracy_score(y_true, y_pred))
    metrics['kappa'] = float(cohen_kappa_score(y_true, y_pred))
    metrics['mcc'] = float(matthews_corrcoef(y_true, y_pred))
    return metrics


def run_knn_evaluation(embeddings, labels, n_neighbors=1, cv_method='loo'):
    # Use 1-NN by default. Use Leave-One-Out cross-validation so each sample is predicted without itself.
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean')
    labels = np.array(labels)

    if cv_method == 'loo':
        cv = LeaveOneOut()
    elif cv_method == 'skf5':
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        raise ValueError('Unknown cv_method')

    print(f"Running k-NN (k={n_neighbors}) with cv={cv_method}. This may take a while for LOO on large sets.")
    try:
        y_pred = cross_val_predict(clf, embeddings, labels, cv=cv, n_jobs=-1)
    except Exception as e:
        print(f"LOO failed or too slow ({e}), retrying with 5-fold stratified CV")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_pred = cross_val_predict(clf, embeddings, labels, cv=cv, n_jobs=-1)

    metrics = compute_metrics(labels, y_pred)
    return metrics


def run_classifier_evaluation(checkpoint_path, data_path, num_classes, feature_dim, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for classifier evaluation: {device}")

    # Create model the same way embeddings were generated: keep embedding_size default
    # and optionally apply a feature selector if `feature_dim` was used during training.
    model = RecognitionNet(num_classes=num_classes, feature_dim=feature_dim, head_type='linear')
    model.to(device)

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    # Support different checkpoint key names and allow mismatches (strict=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"Warning: loading state_dict with strict=False raised: {e}")
        # Last resort: try to load key-by-key where shapes match
        own_state = model.state_dict()
        for k, v in list(state_dict.items()):
            if k in own_state and own_state[k].shape == v.shape:
                own_state[k] = v
        model.load_state_dict(own_state)
    model.eval()

    # Load dataset and split
    train_path = os.path.join(data_path, 'train')
    if os.path.exists(train_path):
        data_path = train_path
    full_dataset = get_recognition_dataset(data_dir=data_path, split=None, fast_mode=False)
    train_dataset, val_dataset, test_dataset = DatasetSplitter.split_dataset(full_dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)

    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    metrics = evaluate_classification(model, test_loader, device, num_classes)
    # Remap metrics names to match k-NN output naming used above
    out = {
        'accuracy': float(metrics.get('accuracy', 0.0)),
        'precision_macro': float(metrics.get('precision', 0.0) or 0.0),
        'recall_macro': float(metrics.get('recall', 0.0) or 0.0),
        'f1_macro': float(metrics.get('f1', 0.0) or 0.0),
        'precision_micro': None,
        'recall_micro': None,
        'f1_micro': None,
        'balanced_accuracy': float(metrics.get('accuracy', 0.0)),
        'kappa': None,
        'mcc': None
    }
    # If evaluate_classification returned sampled preds & labels, compute micro metrics
    if 'predictions' in metrics and 'labels' in metrics and len(metrics['predictions']) > 0:
        y_true = np.array(metrics['labels'])
        y_pred = np.array(metrics['predictions'])
        out['precision_micro'] = float(precision_score(y_true, y_pred, average='micro', zero_division=0))
        out['recall_micro'] = float(recall_score(y_true, y_pred, average='micro', zero_division=0))
        out['f1_micro'] = float(f1_score(y_true, y_pred, average='micro', zero_division=0))
        out['balanced_accuracy'] = float(balanced_accuracy_score(y_true, y_pred))
        out['kappa'] = float(cohen_kappa_score(y_true, y_pred))
        out['mcc'] = float(matthews_corrcoef(y_true, y_pred))

    return out


def pretty_print_metrics(title, metrics):
    print('\n' + title)
    print('-' * len(title))
    keys = [
        'accuracy','precision_macro','recall_macro','f1_macro',
        'precision_micro','recall_micro','f1_micro',
        'balanced_accuracy','kappa','mcc'
    ]
    for k in keys:
        v = metrics.get(k)
        if v is None:
            s = 'N/A'
        else:
            s = f"{v:.4f}"
        print(f"{k}: {s}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive model comparison: classifier vs embeddings (k-NN).")
    parser.add_argument('--rec-emb', type=str, default='logs/recognition/test_embeddings.npy')
    parser.add_argument('--rec-lab', type=str, default='logs/recognition/test_labels.npy')
    parser.add_argument('--buf-emb', type=str, default='logs/buffalo/embeddings.npy')
    parser.add_argument('--buf-lab', type=str, default='logs/buffalo/labels.npy')
    parser.add_argument('--checkpoint', type=str, default='logs/recognition/recognition_model_best.pth')
    parser.add_argument('--data-path', type=str, default='Data/recognition_faces')
    parser.add_argument('--num-classes', type=int, default=530)
    parser.add_argument('--feature-dim', type=int, default=256)
    parser.add_argument('--knn-cv', type=str, default='loo', choices=['loo','skf5'])
    parser.add_argument('--output-dir', type=str, default='logs/comparison')
    args = parser.parse_args()

    # Load embeddings
    rec_emb = np.load(args.rec_emb) if Path(args.rec_emb).exists() else None
    rec_lab = np.load(args.rec_lab) if Path(args.rec_lab).exists() else None
    buf_emb = np.load(args.buf_emb) if Path(args.buf_emb).exists() else None
    buf_lab = np.load(args.buf_lab) if Path(args.buf_lab).exists() else None

    if rec_emb is None or rec_lab is None:
        print(f"Recognition embeddings or labels not found at {args.rec_emb} / {args.rec_lab}")
        return
    if buf_emb is None or buf_lab is None:
        print(f"Buffalo embeddings or labels not found at {args.buf_emb} / {args.buf_lab}")
        return

    print(f"Loading embeddings for k-NN evaluation...")
    print(f"Recognition Model: {rec_emb.shape} embeddings")
    print(f"Buffalo Model: {buf_emb.shape} embeddings")

    # METHOD 1: k-NN classification on embeddings
    print('\nMETHOD 1: k-NN CLASSIFICATION ON EMBEDDINGS')
    print('Evaluating Recognition Model (k-NN)...')
    rec_knn_metrics = run_knn_evaluation(rec_emb, rec_lab, n_neighbors=1, cv_method=args.knn_cv)
    print('Evaluating Buffalo Model (k-NN)...')
    buf_knn_metrics = run_knn_evaluation(buf_emb, buf_lab, n_neighbors=1, cv_method=args.knn_cv)

    print('\nk-NN Comparison:')
    for k in ['accuracy','precision_macro','recall_macro','f1_macro','precision_micro','recall_micro','f1_micro','balanced_accuracy','kappa','mcc']:
        r = rec_knn_metrics.get(k)
        b = buf_knn_metrics.get(k)
        diff = None
        if r is not None and b is not None:
            diff = r - b
        print(f"{k}: Recognition {r:.4f} vs Buffalo {b:.4f} (diff: {diff:+.4f})")

    # METHOD 2: Trained classifier evaluation (Recognition model)
    print('\nMETHOD 2: TRAINED CLASSIFIER EVALUATION')
    print('Loading Recognition Model for classifier evaluation...')
    try:
        classifier_metrics = run_classifier_evaluation(args.checkpoint, args.data_path, args.num_classes, args.feature_dim)
    except Exception as e:
        print(f"Classifier evaluation failed: {e}")
        classifier_metrics = None

    if classifier_metrics is not None:
        print('\nRecognition Model (Classifier):')
        pretty_print_metrics('', classifier_metrics)

        print('\nClassifier vs k-NN Comparison for Recognition Model:')
        for k in ['accuracy','precision_macro','recall_macro','f1_macro','precision_micro','recall_micro','f1_micro','balanced_accuracy','kappa','mcc']:
            kn = rec_knn_metrics.get(k)
            cl = classifier_metrics.get(k)
            try:
                diff = kn - cl if kn is not None and cl is not None else None
            except Exception:
                diff = None
            print(f"{k}: k-NN {kn:.4f} vs Classifier {cl if cl is not None else 'N/A'} (diff: {diff:+.4f} if applicable)")

    # Save summary
    out = {
        'rec_knn': rec_knn_metrics,
        'buf_knn': buf_knn_metrics,
        'classifier': classifier_metrics
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'comprehensive_comparison.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved comprehensive results to {out_path}")


if __name__ == '__main__':
    main()
