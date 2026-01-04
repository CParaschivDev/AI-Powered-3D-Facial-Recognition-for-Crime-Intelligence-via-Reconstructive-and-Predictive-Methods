from pathlib import Path
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shutil
from backend.core.paths import uk_police_root
from backend.core.config import settings

REQUIRED_COLUMNS = ["Month", "Crime type", "Latitude", "Longitude", "Outcome type", "Falls within", "LSOA name", "LSOA code"]

def _estimate_total_size(csv_paths):
    total = 0
    for p in csv_paths:
        try:
            total += os.path.getsize(p)
        except OSError:
            pass
    return total

def build_parquet(out_path: Path, force_filter: str | None = None, progress_callback=None):
    """Stream all UK police CSVs into a single parquet file with compression.

    - Reads CSVs in chunks to avoid large memory use.
    - Adds any missing required columns.
    - Applies optional force filter (case-insensitive substring match on 'Falls within').
    - Checks available disk space before starting and warns/errors early if insufficient.
    - Writes to a temp file then atomically renames on success.
    Returns the final parquet path as string or None if no data.
    """
    if not settings.DATA_ROOT:
        raise ValueError("DATA_ROOT not configured. Please set it in .env file.")

    csv_paths = [p for p in uk_police_root().rglob("*.csv") if "street" in p.name.lower()]
    if not csv_paths:
        return None
    if progress_callback:
        progress_callback({"files_total": len(csv_paths)})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_suffix(".tmp.parquet")
    if temp_path.exists():
        try:
            temp_path.unlink()
        except OSError:
            pass

    # Disk space check (rough estimate: compressed parquet ~30-60% of raw CSV size)
    estimated_raw = _estimate_total_size(csv_paths)
    est_parquet = int(estimated_raw * 0.5)  # heuristic
    usage = shutil.disk_usage(out_path.parent)
    free_bytes = usage.free
    if free_bytes < est_parquet * 1.1:  # require a 10% headroom
        raise OSError(
            f"Insufficient disk space: need ~{est_parquet/1e6:.1f} MB, have {free_bytes/1e6:.1f} MB in {out_path.parent}"
        )

    compression = os.getenv("CRIME_PARQUET_COMPRESSION", "snappy")
    writer = None
    total_rows = 0
    try:
        for idx, csv_file in enumerate(csv_paths, start=1):
            try:
                # Read in manageable chunks
                for chunk in pd.read_csv(csv_file, chunksize=50_000):
                    # Ensure required columns exist
                    for col in REQUIRED_COLUMNS:
                        if col not in chunk.columns:
                            chunk[col] = None
                    if force_filter:
                        mask = chunk["Falls within"].fillna("").str.lower().str.contains(force_filter.lower())
                        chunk = chunk[mask]
                        if chunk.empty:
                            continue
                    # Normalize Month to YYYY-MM (keep original for now; evaluation layer will parse)
                    if "Month" in chunk.columns:
                        # Avoid expensive parsing repeatedly; just ensure string
                        chunk["Month"] = chunk["Month"].astype(str)
                    table = pa.Table.from_pandas(chunk, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(str(temp_path), table.schema, compression=compression)
                    writer.write_table(table)
                    total_rows += table.num_rows
                    if progress_callback:
                        progress_callback({
                            "files_processed": idx,
                            "rows_written": total_rows,
                            "last_file": str(csv_file),
                        })
            except Exception:
                # Skip corrupt CSVs but continue
                continue
        if writer is None:
            return None
    finally:
        if writer is not None:
            writer.close()

    # Finalize
    if temp_path.exists():
        # Optionally write a small sidecar manifest with metadata
        os.replace(temp_path, out_path)
        return str(out_path)
    return None
