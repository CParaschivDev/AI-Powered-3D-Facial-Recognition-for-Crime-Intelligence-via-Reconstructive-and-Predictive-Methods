#!/usr/bin/env python3
import numpy as np
import sys
import os
# Ensure project root is on path so local imports work when run as a script
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from evaluation.evaluate_recognition import evaluate_all_metrics

def load(path):
    return np.load(path)

def l2_normalize(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norm

def mean_intra_inter_distances(embeddings, labels):
    labels = np.array(labels)
    unique = np.unique(labels)
    intra = []
    inter = []
    for c in unique:
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            continue
        embs = embeddings[idx]
        # pairwise distances
        dists = np.linalg.norm(embs[:, None, :] - embs[None, :, :], axis=2)
        # take upper triangle
        iu = np.triu_indices_from(dists, k=1)
        intra.extend(dists[iu].tolist())
    # inter-class: sample random pairs from different classes
    rng = np.random.default_rng(0)
    n = 10000
    for _ in range(n):
        a = rng.integers(0, len(labels))
        b = rng.integers(0, len(labels))
        if labels[a] != labels[b]:
            inter.append(np.linalg.norm(embeddings[a] - embeddings[b]))
    return np.mean(intra) if intra else float('nan'), np.mean(inter)

def per_class_accuracy(embeddings, labels):
    labels = np.array(labels)
    unique = np.unique(labels)
    class_means = {c: embeddings[labels==c].mean(axis=0) for c in unique}
    preds = []
    for emb in embeddings:
        dists = [np.linalg.norm(emb - class_means[c]) for c in unique]
        preds.append(unique[np.argmin(dists)])
    preds = np.array(preds)
    accs = {}
    for c in unique:
        idx = labels==c
        accs[c] = np.mean(preds[idx] == labels[idx])
    accs_list = np.array(list(accs.values()))
    return accs_list.mean(), accs_list.std(), np.percentile(accs_list, [5,25,50,75,95])

def run():
    base = 'logs/recognition'
    rec_emb = load(os.path.join(base, 'test_embeddings.npy'))
    rec_lab = load(os.path.join(base, 'test_labels.npy'))

    baseb = 'logs/buffalo'
    buf_emb = load(os.path.join(baseb, 'embeddings.npy'))
    buf_lab = load(os.path.join(baseb, 'labels.npy'))

    print('Recognition embeddings:', rec_emb.shape)
    print('Buffalo embeddings:', buf_emb.shape)

    print('\nEvaluating original embeddings (nearest class mean)')
    rec_metrics = evaluate_all_metrics(rec_emb, rec_lab)
    buf_metrics = evaluate_all_metrics(buf_emb, buf_lab)
    print('Recognition accuracy:', rec_metrics['accuracy'])
    print('Buffalo accuracy:', buf_metrics['accuracy'])

    print('\nDiagnostic distances (original)')
    rec_intra, rec_inter = mean_intra_inter_distances(rec_emb, rec_lab)
    buf_intra, buf_inter = mean_intra_inter_distances(buf_emb, buf_lab)
    print(f'Recognition mean intra: {rec_intra:.4f}, inter: {rec_inter:.4f}')
    print(f'Buffalo   mean intra: {buf_intra:.4f}, inter: {buf_inter:.4f}')

    print('\nPer-class accuracy summary (original embeddings)')
    rec_mean, rec_std, rec_pct = per_class_accuracy(rec_emb, rec_lab)
    buf_mean, buf_std, buf_pct = per_class_accuracy(buf_emb, buf_lab)
    print(f'Recognition per-class mean: {rec_mean:.4f} std: {rec_std:.4f} pctiles: {rec_pct}')
    print(f'Buffalo   per-class mean: {buf_mean:.4f} std: {buf_std:.4f} pctiles: {buf_pct}')

    # Now try L2-normalized embeddings
    rec_norm = l2_normalize(rec_emb)
    buf_norm = l2_normalize(buf_emb)
    print('\nEvaluating L2-normalized embeddings')
    rec_metrics_n = evaluate_all_metrics(rec_norm, rec_lab)
    buf_metrics_n = evaluate_all_metrics(buf_norm, buf_lab)
    print('Recognition accuracy (L2):', rec_metrics_n['accuracy'])
    print('Buffalo accuracy   (L2):', buf_metrics_n['accuracy'])

if __name__ == "__main__":
    run()
