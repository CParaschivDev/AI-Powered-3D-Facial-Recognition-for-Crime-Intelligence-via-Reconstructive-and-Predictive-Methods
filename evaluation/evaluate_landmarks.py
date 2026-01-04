import numpy as np
import argparse
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compute_nme(preds, gts):
    # NME: Normalized Mean Error (per sample, per landmark)
    n_samples = preds.shape[0]
    nme_list = []
    for i in range(n_samples):
        pred = preds[i]
        gt = gts[i]
        # Assume shape (num_landmarks, 2)
        norm = np.linalg.norm(np.max(gt, axis=0) - np.min(gt, axis=0))
        nme = np.mean(np.linalg.norm(pred - gt, axis=1)) / (norm + 1e-8)
        nme_list.append(float(nme))
    return float(np.mean(nme_list)), nme_list

def main(args):
    preds = np.load(args.preds_path)
    gts = np.load(args.gt_path)
    mse = float(mean_squared_error(gts.reshape(-1, 2), preds.reshape(-1, 2)))
    mae = float(mean_absolute_error(gts.reshape(-1, 2), preds.reshape(-1, 2)))
    r2 = float(r2_score(gts.reshape(-1, 2), preds.reshape(-1, 2)))
    nme_mean, nme_list = compute_nme(preds, gts)
    nme_list = [float(x) for x in nme_list]
    metrics = {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "nme_mean": nme_mean,
        "nme_per_sample": nme_list,
        "num_samples": int(preds.shape[0]),
        "num_landmarks": int(preds.shape[1]) if preds.ndim > 1 else None
    }
    print(json.dumps(metrics))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Landmark Detection Performance")
    parser.add_argument("--preds-path", type=str, required=True, help="Path to predicted landmarks .npy file.")
    parser.add_argument("--gt-path", type=str, required=True, help="Path to ground truth landmarks .npy file.")
    args = parser.parse_args()
    main(args)
