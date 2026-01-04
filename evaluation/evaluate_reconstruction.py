
import numpy as np
import argparse
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def main(args):
    print("--- 3D Reconstruction Parameter Evaluation ---")
    print(f"Predicted params path: {args.preds_path}")
    print(f"Ground truth params path: {args.gt_path}")

    # Load predictions and ground truth
    preds = np.load(args.preds_path)
    gt = np.load(args.gt_path)

    if preds.shape != gt.shape:
        raise ValueError(f"Shape mismatch: preds {preds.shape}, gt {gt.shape}")

    # Compute metrics
    mse = float(mean_squared_error(gt, preds))
    mae = float(mean_absolute_error(gt, preds))
    r2 = float(r2_score(gt, preds))
    per_sample_mse = np.mean((gt - preds) ** 2, axis=1)
    per_sample_mae = np.mean(np.abs(gt - preds), axis=1)

    # Convert numpy types to python types for JSON serialization
    per_sample_mse = [float(x) for x in per_sample_mse]
    per_sample_mae = [float(x) for x in per_sample_mae]

    metrics = {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "per_sample_mse": per_sample_mse,
        "per_sample_mae": per_sample_mae,
        "num_samples": int(preds.shape[0]),
        "num_params": int(preds.shape[1])
    }
    print(json.dumps(metrics))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 3D Reconstruction Parameter Performance")
    parser.add_argument("--preds-path", type=str, required=True, help="Path to predicted parameter .npy file.")
    parser.add_argument("--gt-path", type=str, required=True, help="Path to ground truth parameter .npy file.")
    args = parser.parse_args()
    main(args)
