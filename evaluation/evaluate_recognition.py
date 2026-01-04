
import numpy as np
import argparse
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score,
    classification_report
)

def evaluate_all_metrics(embeddings, labels):
    # For multiclass, use nearest class center or argmax classifier
    # Here, we assume embeddings are logits for each class (N, C) or feature vectors (N, D)
    # If embeddings are feature vectors, use nearest class mean; if logits, use argmax
    unique_labels = np.unique(labels)
    if embeddings.ndim == 2 and embeddings.shape[1] == len(unique_labels):
        # Looks like logits -> predicted index per class
        preds_idx = np.argmax(embeddings, axis=1)
        preds = unique_labels[preds_idx]
    else:
        # Feature vectors: use nearest class mean
        class_means = []
        unique_labels = np.unique(labels)
        for c in unique_labels:
            class_means.append(embeddings[labels == c].mean(axis=0))
        class_means = np.stack(class_means)
        # Compute distances efficiently to avoid memory issues
        preds = []
        for emb in embeddings:
            dists = np.linalg.norm(class_means - emb, axis=1)
            preds.append(np.argmin(dists))
        preds = np.array(preds)
        # preds currently are indices into class_means (0..K-1) - map back to original label values
        preds = unique_labels[preds]

    acc = accuracy_score(labels, preds)
    prec_macro = precision_score(labels, preds, average='macro', zero_division=0)
    rec_macro = recall_score(labels, preds, average='macro', zero_division=0)
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    prec_micro = precision_score(labels, preds, average='micro', zero_division=0)
    rec_micro = recall_score(labels, preds, average='micro', zero_division=0)
    f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
    cm = confusion_matrix(labels, preds)
    cm_list = cm.tolist()
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    bal_acc = balanced_accuracy_score(labels, preds)
    kappa = cohen_kappa_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_micro": prec_micro,
        "recall_micro": rec_micro,
        "f1_micro": f1_micro,
        "balanced_accuracy": bal_acc,
        "kappa": kappa,
        "mcc": mcc,
        "confusion_matrix": cm_list,
        "classification_report": report,
    }

def main(args):
    # Load embeddings and labels
    scores = np.load(args.embeddings_path)
    labels = np.load(args.labels_path)
    # Compute all metrics
    metrics = evaluate_all_metrics(scores, labels)
    # Output the JSON
    json_output = json.dumps(metrics)
    if args.output_path:
        with open(args.output_path, 'w') as f:
            f.write(json_output)
    else:
        print(json_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Recognition Performance")
    parser.add_argument("--embeddings-path", type=str, required=True, help="Path to pre-computed embeddings.")
    parser.add_argument("--labels-path", type=str, required=True, help="Path to evaluation pair labels.")
    parser.add_argument("--output-path", type=str, default=None, help="Path to save the JSON output.")
    args = parser.parse_args()
    main(args)
