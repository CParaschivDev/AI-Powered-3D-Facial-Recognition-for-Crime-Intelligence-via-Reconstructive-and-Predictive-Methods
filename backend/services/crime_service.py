import os
import json
from datetime import datetime
from functools import lru_cache
from typing import Optional, List, Dict, Any

import polars as pl

from backend.core.config import settings

# Cache configuration fallbacks. Use settings if provided, otherwise disable cache safely.
CACHE_ENABLED = getattr(settings, "CACHE_ENABLED", False)
CACHE_EXPIRATION_SECONDS = getattr(settings, "CACHE_EXPIRATION_SECONDS", 3600)
redis_client = None
if CACHE_ENABLED:
    try:
        import redis as _redis

        redis_url = getattr(settings, "REDIS_URL", "redis://localhost:6379/0")
        # create a client with decoded responses for easier JSON handling
        redis_client = _redis.Redis.from_url(redis_url, decode_responses=True)
    except Exception as _e:
        # If redis is not available or configuration fails, disable cache and continue
        print(f"WARN: Redis caching disabled due to error: {_e}")
        CACHE_ENABLED = False

def _compute_summary(df: pl.DataFrame) -> Dict[str, Any]:
    if df.height == 0:
        return {"rows": 0, "min_month": None, "max_month": None, "forces": 0, "lsoas": 0}
    min_month = df.select(pl.min("Month")).item()
    max_month = df.select(pl.max("Month")).item()
    forces = df.select(pl.col("Falls within").n_unique()).item()
    lsoas = df.select(pl.col("LSOA name").n_unique()).item() if "LSOA name" in df.columns else 0
    return {
        "rows": int(df.height),
        "min_month": str(min_month),
        "max_month": str(max_month),
        "forces": int(forces),
        "lsoas": int(lsoas),
    }

def get_crime_summary() -> Dict[str, Any]:
    df = get_crime_dataframe()
    if df is None:
        return {"rows": 0, "min_month": None, "max_month": None, "forces": 0, "lsoas": 0}
    summary = _compute_summary(df)
    return summary

def clear_crime_caches():
    try:
        get_crime_dataframe.cache_clear()
    except Exception:
        pass


@lru_cache(maxsize=1)
def get_crime_dataframe() -> Optional[pl.DataFrame]:
    """
    Loads the crime data from the parquet file.
    Handles the fallback path logic.
    Caches the loaded DataFrame in memory using lru_cache.
    """
    processed_path = os.path.join(settings.DATA_ROOT, "processed", "crime_full.parquet") if settings.DATA_ROOT else os.path.join("data", "processed", "crime_full.parquet")
    fallback_path = os.path.join("reports", "crime", "crime_full.parquet")

    file_path = None
    if os.path.exists(processed_path):
        file_path = processed_path
    elif os.path.exists(fallback_path):
        file_path = fallback_path
    else:
        print(f"ERROR: Crime data not found at {processed_path} or {fallback_path}")
        return None

    print(f"Loading crime data from: {file_path}")
    try:
        df = pl.read_parquet(file_path)
        print(f"Crime parquet loaded: rows={df.height}, columns={df.columns}")
        # Try to detect a Month-like column if 'Month' is not present (case-insensitive or variant names)
        if "Month" not in df.columns:
            candidate = None
            for c in df.columns:
                lc = c.lower()
                if lc == "month":
                    candidate = c
                    break
            if candidate is None:
                for c in df.columns:
                    lc = c.lower()
                    if "month" in lc or "date" in lc:
                        candidate = c
                        break
            if candidate:
                try:
                    print(f"Renaming detected column '{candidate}' to 'Month' for normalization")
                    df = df.rename({candidate: "Month"})
                except Exception:
                    print(f"Could not rename column {candidate} to Month; proceeding without rename")
        # Robust Month normalization: convert variety of string forms to first-of-month Date.
        if "Month" not in df.columns:
            print("ERROR: 'Month' column missing in crime parquet after detection attempts")
            return None
        month_col = df["Month"].dtype
        if month_col != pl.Date:
            # Normalize to string, trim, ensure we only keep first token (avoid accidental time)
            # Normalize: cast to Utf8, trim whitespace (older Polars uses .str.strip_chars, newer .str.strip)
            # Capture some sample raw values (best-effort) for debugging
            try:
                sample_raw = df.select(pl.col("Month").cast(pl.Utf8)).head(10).to_series().to_list()
                print(f"Sample raw Month values (first 10): {sample_raw}")
            except Exception:
                pass

            month_utf8 = pl.col("Month").cast(pl.Utf8)
            try:
                month_utf8 = month_utf8.str.strip()  # recent Polars
            except AttributeError:
                month_utf8 = month_utf8.str.strip_chars()  # fallback for older Polars
            df = df.with_columns(
                month_utf8.str.slice(0, 19).alias("Month_str")  # truncate any timestamp
            )
            # Add -01 for year-month only values
            df = df.with_columns(
                pl.when(pl.col("Month_str").str.len_chars() == 7)
                  .then(pl.col("Month_str") + "-01")
                  .otherwise(pl.col("Month_str"))
                  .alias("Month_norm")
            )
            parsed = pl.col("Month_norm").str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
            df = df.with_columns(Month=parsed)
            # Fallback: if parse failed (all null), try alternative patterns directly
            if df.select(pl.col("Month").is_null().sum()).item() == df.height:
                alt = pl.coalesce([
                    pl.col("Month_norm").str.strptime(pl.Date, format="%d/%m/%Y", strict=False),
                    pl.col("Month_norm").str.strptime(pl.Date, format="%m/%d/%Y", strict=False),
                ])
                df = df.with_columns(Month=alt)
            # If still null, drop rows with null Month but provide helpful debug info
            null_months = df.select(pl.col("Month").is_null().sum()).item()
            if null_months:
                print(f"WARN: {null_months} rows have unparseable Month values")
                # If all rows would be dropped, print samples and abort to avoid silent data loss
                if null_months == df.height:
                    try:
                        raw_samples = df.select(pl.col("Month_str").cast(pl.Utf8)).head(20).to_series().to_list()
                        print(f"ERROR: All Month values parsed as null. Sample raw Month_str values: {raw_samples}")
                    except Exception:
                        print("ERROR: All Month values parsed as null and sample extraction failed")
                    return None
                else:
                    print("Dropping rows with null Month and continuing")
                    df = df.filter(pl.col("Month").is_not_null())
            df = df.drop([c for c in ["Month_str", "Month_norm"] if c in df.columns])
        # Ensure Month is Date finally
        if df["Month"].dtype == pl.Datetime:
            df = df.with_columns(pl.col("Month").cast(pl.Date))
        return df
    except Exception as e:
        print(f"Error loading or processing crime parquet file: {e}")
        return None

def ensure_month_date(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure the Month column is type Date (first of month). Accepts values in 'YYYY-MM' or 'YYYY-MM-DD'."""
    try:
        if "Month" not in df.columns:
            return df
        if df["Month"].dtype == pl.Date:
            return df
        return df.with_columns(
            pl.when(pl.col("Month").cast(pl.Utf8).str.len_chars() == 7)
              .then(pl.col("Month").cast(pl.Utf8) + "-01")
              .otherwise(pl.col("Month").cast(pl.Utf8))
              .str.slice(0,10)
              .str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
              .alias("Month")
        )
    except Exception as e:
        print(f"Error normalizing Month column: {e}")
        return df


def get_monthly_trends_by_force(from_date: str, to_date: str) -> List[Dict[str, Any]]:
    """
    Computes monthly crime trends aggregated by police force.
    """
    df = get_crime_dataframe()
    if df is None:
        return []

    # Interpret from/to as first-of-month inclusive; end boundary exclusive first-of-next-month
    start_dt = datetime.strptime(from_date + "-01", "%Y-%m-%d").date()
    to_year, to_month = map(int, to_date.split('-'))
    if to_month == 12:
        end_boundary_dt = datetime(to_year + 1, 1, 1).date()
    else:
        end_boundary_dt = datetime(to_year, to_month + 1, 1).date()

    df = ensure_month_date(df)
    result_df = df.filter((pl.col("Month") >= start_dt) & (pl.col("Month") < end_boundary_dt)) \
        .group_by(["Falls within", "Month"]) \
        .agg(pl.count().alias("crime_count")) \
        .sort(["Falls within", "Month"])

    result = result_df.to_dicts()
    # Convert dates to strings for JSON serialization
    for row in result:
        row["Month"] = row["Month"].strftime("%Y-%m-%d")
    return result


def get_latest_hotspots(force: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Computes the latest crime hotspots for a given force.
    Hotspots are defined as LSOAs with the highest crime counts in the most recent month.
    """
    # cache_key = f"crime:hotspots:{force or 'ALL'}"
    # if CACHE_ENABLED:
    #     cached_raw = redis_client.get(cache_key)
    #     if cached_raw is not None:
    #         try:
    #             cached_data = json.loads(cached_raw)
    #             if cached_data:  # recompute if empty list
    #                 return cached_data
    #         except Exception:
    #             pass

    df = get_crime_dataframe()
    if df is None:
        return []

    # Determine target month: prefer month with most data for the force, else global latest
    if force:
        # Find month with most records for the force
        force_months = (
            df.filter(pl.col("Falls within") == force)
              .group_by("Month")
              .agg(pl.count().alias("count"))
              .sort("count", descending=True)
              .limit(1)
        )
        if force_months.height > 0:
            target_month = force_months.select("Month").item()
        else:
            target_month = df.select(pl.max("Month")).item()
    else:
        target_month = df.select(pl.max("Month")).item()

    # Ensure target_month comparable (cast to date if needed)
    df = ensure_month_date(df)
    # Build filter: by force if provided, else global
    base_filter = (pl.col("Month") == target_month) & pl.col("LSOA name").is_not_null()
    if force:
        base_filter = base_filter & (pl.col("Falls within") == force)

    result_df = (
        df.filter(base_filter)
        .group_by(["LSOA name", "Latitude", "Longitude"])
        .agg(pl.count().alias("crime_count"))
        .sort("crime_count", descending=True)
        .limit(50)
    )

    result = result_df.to_dicts()

    # if CACHE_ENABLED:
    #     redis_client.set(cache_key, json.dumps(result), ex=CACHE_EXPIRATION_SECONDS)

    return result


def get_lsoa_series(lsoa: str) -> List[Dict[str, Any]]:
    """
    Computes a time series of crime counts for a specific LSOA.
    """
    cache_key = f"crime:lsoa:{lsoa}"
    if CACHE_ENABLED:
        cached_raw = redis_client.get(cache_key)
        if cached_raw is not None:
            try:
                cached_data = json.loads(cached_raw)
                if cached_data:
                    return cached_data
            except Exception:
                pass

    df = get_crime_dataframe()
    if df is None:
        return []

    df = ensure_month_date(df)
    lsoa_df = df.filter(pl.col("LSOA name") == lsoa)
    # If 'Crime type' is entirely null or missing, aggregate without it and label as 'ALL'
    if "Crime type" not in lsoa_df.columns or lsoa_df.select(pl.col("Crime type").is_null().sum()).item() == lsoa_df.height:
        result_df = (
            lsoa_df.group_by(["Month"]).agg(pl.count().alias("crime_count")).with_columns(
                pl.lit("ALL").alias("Crime type")
            ).sort("Month")
        )
    else:
        result_df = (
            lsoa_df.group_by(["Month", "Crime type"]).agg(pl.count().alias("crime_count")).sort("Month")
        )

    result = result_df.to_dicts()

    if CACHE_ENABLED:
        for row in result:
            row["Month"] = row["Month"].strftime("%Y-%m-%d")
        redis_client.set(cache_key, json.dumps(result), ex=CACHE_EXPIRATION_SECONDS)

    return result


def get_monthly_trends_aggregated() -> List[Dict[str, Any]]:
    """Pre-computed total crime counts per force per month (no filtering)."""
    cache_key = "crime:trends:aggregated"
    if CACHE_ENABLED and (cached_result := redis_client.get(cache_key)):
        return json.loads(cached_result)
    df = get_crime_dataframe()
    if df is None:
        return []
    result_df = (
        df.group_by(["Falls within", "Month"]).agg(pl.count().alias("crime_count")).sort(["Falls within", "Month"])
    )
    result = result_df.to_dicts()
    if CACHE_ENABLED:
        # Serialize dates
        for row in result:
            row["Month"] = row["Month"].strftime("%Y-%m-%d")
        redis_client.set(cache_key, json.dumps(result), ex=CACHE_EXPIRATION_SECONDS)
    return result


def get_crime_diagnostics() -> Dict[str, Any]:
    """Return lightweight diagnostics about the crime parquet for local debugging.

    This is intended for development use only. It provides rows, columns, a small
    sample of raw Month strings, and counts of nulls for important columns.
    """
    try:
        df = get_crime_dataframe()
        if df is None:
            return {"ok": False, "reason": "dataframe_load_failed"}

        # Basic stats
        rows = int(df.height)
        cols = list(df.columns)

        # Sample raw Month values (up to 20)
        month_samples = []
        if "Month" in df.columns:
            try:
                # cast to string for readable samples
                month_samples = df.select(pl.col("Month").cast(pl.Utf8)).head(20).to_series().to_list()
            except Exception:
                month_samples = []

        # Null counts for a few important columns
        nulls = {}
        for c in ["Month", "Falls within", "LSOA name", "Crime type"]:
            if c in df.columns:
                try:
                    nulls[c] = int(df.select(pl.col(c).is_null().sum()).item())
                except Exception:
                    nulls[c] = None
            else:
                nulls[c] = None

        # Distinct counts for Month, Falls within, LSOA name, Crime type
        distincts = {}
        try:
            distincts["Month"] = int(df.select(pl.col("Month").n_unique()).item()) if "Month" in df.columns else 0
        except Exception:
            distincts["Month"] = None
        for c in ["Falls within", "LSOA name", "Crime type"]:
            if c in df.columns:
                try:
                    distincts[c] = int(df.select(pl.col(c).n_unique()).item())
                except Exception:
                    distincts[c] = None
            else:
                distincts[c] = None

        return {
            "ok": True,
            "rows": rows,
            "columns": cols,
            "month_samples": month_samples,
            "null_counts": nulls,
            "distinct_counts": distincts,
        }
    except Exception as e:
        return {"ok": False, "reason": str(e)}