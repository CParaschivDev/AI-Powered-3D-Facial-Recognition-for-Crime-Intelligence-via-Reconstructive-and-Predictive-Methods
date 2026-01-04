import torch
import numpy as np
import os
from backend.core.safety import BUF_LIMIT
from tqdm import tqdm
from pathlib import Path
import argparse

from training.utils.dataset_loader import get_reconstruction_dataset
from training.reconstruction_train import ReconstructionNet

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load test dataset (use all data if no split)
    test_dataset = get_reconstruction_dataset(data_dir=args.data_path, split=None)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Infer number of params from first sample
    _, sample_params = test_dataset[0]
    num_params = sample_params.shape[0]
    print(f"Detected {num_params} 3DMM parameters per face.")

    # Load model
    model = ReconstructionNet(num_params=num_params).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Stream batches to disk to avoid unbounded memory growth
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(args.output_dir) / "_tmp_batches"
    tmp_dir.mkdir(exist_ok=True)

    batch_files = []
    buf_pred = []
    buf_gt = []
    buf_bytes = 0
    # BUF_LIMIT imported from `backend.core.safety`

    def _flush_buffer_to_files(buf_pred, buf_gt, pred_path, gt_path):
        total_rows = sum(a.shape[0] for a in buf_pred)
        if total_rows == 0:
            np.save(pred_path, np.zeros((0, 0)))
            np.save(gt_path, np.zeros((0, 0)))
            return 0
        pred_dim = buf_pred[0].shape[1]
        dtype = buf_pred[0].dtype
        mem = np.lib.format.open_memmap(str(pred_path), mode='w+', dtype=dtype, shape=(total_rows, pred_dim))
        pos = 0
        for a in buf_pred:
            r = a.shape[0]
            mem[pos: pos + r] = a
            pos += r
        del mem
        # Write ground-truth arrays sequentially to avoid concatenation in memory
        if buf_gt:
            gt_total = sum(a.shape[0] for a in buf_gt)
            gt_mem = np.lib.format.open_memmap(str(gt_path), mode='w+', dtype=dtype, shape=(gt_total, pred_dim))
            gpos = 0
            for ga in buf_gt:
                grow = ga.shape[0]
                gt_mem[gpos: gpos + grow] = ga
                gpos += grow
            del gt_mem
        else:
            np.save(str(gt_path), np.zeros((0, pred_dim), dtype=dtype))
        return total_rows

    with torch.no_grad():
        batch_idx = 0
        for inputs, gt_params in tqdm(test_loader, desc="Generating predictions"):
            inputs = inputs.to(device)
            preds = model(inputs)
            arr_p = preds.cpu().numpy()
            arr_g = gt_params.cpu().numpy()
            buf_pred.append(arr_p)
            buf_gt.append(arr_g)
            buf_bytes += arr_p.nbytes + arr_g.nbytes

            if buf_bytes >= BUF_LIMIT:
                ppath = tmp_dir / f"pred_batch_{batch_idx}.npy"
                gpath = tmp_dir / f"gt_batch_{batch_idx}.npy"
                rows = _flush_buffer_to_files(buf_pred, buf_gt, ppath, gpath)
                batch_files.append((str(ppath), str(gpath), rows))
                batch_idx += 1
                buf_pred = []
                buf_gt = []
                buf_bytes = 0

        if buf_pred:
            ppath = tmp_dir / f"pred_batch_{batch_idx}.npy"
            gpath = tmp_dir / f"gt_batch_{batch_idx}.npy"
            rows = _flush_buffer_to_files(buf_pred, buf_gt, ppath, gpath)
            batch_files.append((str(ppath), str(gpath), rows))
            batch_idx += 1

    if not batch_files:
        raise RuntimeError("No predictions produced; check dataset and model")

    total_rows = sum(r for _, _, r in batch_files)
    sample = np.load(batch_files[0][0], mmap_mode='r')
    pred_dim = sample.shape[1]
    pred_dtype = sample.dtype

    out_preds = Path(args.output_dir) / "test_preds.npy"
    out_gt = Path(args.output_dir) / "test_gt.npy"

    memmap = np.lib.format.open_memmap(str(out_preds), mode='w+', dtype=pred_dtype, shape=(total_rows, pred_dim))
    gt_arr = np.zeros((total_rows, sample.shape[1]), dtype=sample.dtype)

    pos = 0
    for ppath, gpath, rows in batch_files:
        bpred = np.load(ppath)
        bgt = np.load(gpath)
        memmap[pos: pos + rows] = bpred
        gt_arr[pos: pos + rows] = bgt
        pos += rows
        try:
            os.remove(ppath)
            os.remove(gpath)
        except Exception:
            pass

    del memmap
    np.save(str(out_gt), gt_arr)
    try:
        tmp_dir.rmdir()
    except Exception:
        pass

    print(f"Saved predictions to {out_preds}")
    print(f"Saved ground truth to {out_gt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predictions for 3D reconstruction evaluation.")
    parser.add_argument("--data-path", type=str, default="Data/AFLW2000", help="Path to data folder.")
    parser.add_argument("--model-path", type=str, default="./logs/reconstruction/reconstruction_model.pth", help="Path to trained model checkpoint.")
    parser.add_argument("--output-dir", type=str, default="./logs/reconstruction", help="Directory to save .npy files.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference.")
    args = parser.parse_args()
    main(args)
