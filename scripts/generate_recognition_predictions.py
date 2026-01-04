
import torch
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
import os
from backend.core.safety import BUF_LIMIT

from training.recognition_train import RecognitionNet
from training.utils.dataset_loader import get_recognition_dataset

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading recognition model from {args.model_path}")

    # Load dataset
    test_dataset = get_recognition_dataset(data_dir=args.data_path, split=None)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    class_names = test_dataset.classes
    print(f"Detected {len(class_names)} classes: {class_names}")

    # Model setup
    num_classes = len(class_names)
    model = RecognitionNet(num_classes=num_classes).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    # Load checkpoint weights robustly: adapt classifier if shapes differ, otherwise copy matching params
    ckpt_state = checkpoint.get('model_state_dict', checkpoint)
    model_state = model.state_dict()
    for k, v in ckpt_state.items():
        if k not in model_state:
            print(f"Skipping parameter {k} (missing in model)")
            continue
        tgt = model_state[k]
        if v.size() == tgt.size():
            model_state[k] = v
            continue
        # Handle classifier adaptation specifically
        if k.endswith('classifier.weight') or k == 'classifier.weight':
            # v: [ckpt_classes, embed_dim], tgt: [model_classes, embed_dim]
            ck_rows, ck_cols = v.size()
            tgt_rows, tgt_cols = tgt.size()
            if ck_cols != tgt_cols:
                print(f"Classifier embedding dim mismatch: checkpoint {ck_cols} vs model {tgt_cols}, skipping {k}")
                continue
            if ck_rows >= tgt_rows:
                print(f"Slicing classifier weights from {ck_rows} -> {tgt_rows} rows for {k}")
                model_state[k] = v[:tgt_rows, :].clone()
            else:
                print(f"Extending classifier weights from {ck_rows} -> {tgt_rows} rows for {k}")
                new_w = tgt.clone()
                new_w[:ck_rows, :] = v
                # keep remaining rows as initialized in model (already in tgt)
                model_state[k] = new_w
            continue
        if k.endswith('classifier.bias') or k == 'classifier.bias':
            ck_len = v.size(0)
            tgt_len = tgt.size(0)
            if ck_len >= tgt_len:
                print(f"Slicing classifier bias from {ck_len} -> {tgt_len} for {k}")
                model_state[k] = v[:tgt_len].clone()
            else:
                print(f"Extending classifier bias from {ck_len} -> {tgt_len} for {k}")
                new_b = tgt.clone()
                new_b[:ck_len] = v
                model_state[k] = new_b
            continue
        # Fallback: sizes differ and not classifier -> skip
        print(f"Skipping parameter {k} (shape mismatch: checkpoint {v.size()} vs model {tgt.size()})")
    model.load_state_dict(model_state)
    model.eval()

    # Stream batches to temporary files to avoid unbounded memory growth on large datasets.
    os.makedirs(args.output_dir, exist_ok=True)
    tmp_dir = os.path.join(args.output_dir, "_tmp_batches")
    os.makedirs(tmp_dir, exist_ok=True)

    batch_files = []  # list of (emb_file, labels_file, rows)
    buf_emb = []
    buf_lbl = []
    buf_bytes = 0
    # BUF_LIMIT imported from `backend.core.safety`

    def _flush_buffer_to_files(buf_emb, buf_lbl, emb_path, lbl_path):
        # Write buffered arrays to a single .npy file without concatenating in-memory
        total_rows = sum(a.shape[0] for a in buf_emb)
        if total_rows == 0:
            np.save(emb_path, np.zeros((0, 0)))
            np.save(lbl_path, np.array([], dtype=np.int64))
            return 0
        emb_dim = buf_emb[0].shape[1]
        emb_dtype = buf_emb[0].dtype
        mem = np.lib.format.open_memmap(emb_path, mode='w+', dtype=emb_dtype, shape=(total_rows, emb_dim))
        pos = 0
        for a in buf_emb:
            rows = a.shape[0]
            mem[pos: pos + rows] = a
            pos += rows
        del mem
        # Write labels sequentially to avoid concatenating all label arrays in memory
        if buf_lbl:
            lbl_total = sum(a.shape[0] for a in buf_lbl)
            lbl_mem = np.lib.format.open_memmap(lbl_path, mode='w+', dtype=np.int64, shape=(lbl_total,))
            lpos = 0
            for la in buf_lbl:
                lrows = la.shape[0]
                lbl_mem[lpos: lpos + lrows] = la
                lpos += lrows
            del lbl_mem
        else:
            np.save(lbl_path, np.array([], dtype=np.int64))
        return total_rows

    with torch.no_grad():
        batch_idx = 0
        for inputs, lbls in tqdm(test_loader, desc="Extracting embeddings"):
            inputs = inputs.to(device)
            embs = model(inputs, return_embedding=True)
            arr_e = embs.cpu().numpy()
            arr_l = lbls.numpy()
            buf_emb.append(arr_e)
            buf_lbl.append(arr_l)
            buf_bytes += arr_e.nbytes + arr_l.nbytes

            if buf_bytes >= BUF_LIMIT:
                # flush buffer to disk without building one large array in memory
                emb_path = os.path.join(tmp_dir, f"emb_batch_{batch_idx}.npy")
                lbl_path = os.path.join(tmp_dir, f"lbl_batch_{batch_idx}.npy")
                rows = _flush_buffer_to_files(buf_emb, buf_lbl, emb_path, lbl_path)
                batch_files.append((emb_path, lbl_path, rows))
                batch_idx += 1
                # clear buffer
                buf_emb = []
                buf_lbl = []
                buf_bytes = 0

        # flush remaining buffer
        if buf_emb:
            emb_path = os.path.join(tmp_dir, f"emb_batch_{batch_idx}.npy")
            lbl_path = os.path.join(tmp_dir, f"lbl_batch_{batch_idx}.npy")
            rows = _flush_buffer_to_files(buf_emb, buf_lbl, emb_path, lbl_path)
            batch_files.append((emb_path, lbl_path, rows))
            batch_idx += 1

    # Combine batch files into final memmap to avoid loading everything at once
    if not batch_files:
        raise RuntimeError("No embeddings produced; check dataset and model")

    total_rows = sum(r for _, _, r in batch_files)
    # Load first batch to get embedding dim and dtype
    sample = np.load(batch_files[0][0], mmap_mode='r')
    emb_dim = sample.shape[1]
    emb_dtype = sample.dtype

    out_emb_path = os.path.join(args.output_dir, "test_embeddings.npy")
    out_lbl_path = os.path.join(args.output_dir, "test_labels.npy")

    # Create memmap and write batches sequentially
    memmap = np.lib.format.open_memmap(out_emb_path, mode='w+', dtype=emb_dtype, shape=(total_rows, emb_dim))
    labels_arr = np.zeros((total_rows,), dtype=np.int64)

    pos = 0
    for emb_path, lbl_path, rows in batch_files:
        bemb = np.load(emb_path)
        blbl = np.load(lbl_path)
        memmap[pos: pos + rows] = bemb
        labels_arr[pos: pos + rows] = blbl
        pos += rows
        # remove temp files
        try:
            os.remove(emb_path)
            os.remove(lbl_path)
        except Exception:
            pass

    # finalize
    del memmap
    np.save(out_lbl_path, labels_arr)
    try:
        os.rmdir(tmp_dir)
    except Exception:
        pass

    print(f"Saved embeddings to {out_emb_path}")
    print(f"Saved labels to {out_lbl_path}")
    print(f"Saved embeddings to {os.path.join(args.output_dir, 'test_embeddings.npy')}")
    print(f"Saved labels to {os.path.join(args.output_dir, 'test_labels.npy')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate recognition embeddings and labels for evaluation (PyTorch model).")
    parser.add_argument("--data-path", type=str, required=True, help="Path to data directory (should contain class subfolders)")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained recognition model .pth file")
    parser.add_argument("--output-dir", type=str, default="./logs/recognition", help="Directory to save embeddings and labels")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    args = parser.parse_args()
    main(args)
