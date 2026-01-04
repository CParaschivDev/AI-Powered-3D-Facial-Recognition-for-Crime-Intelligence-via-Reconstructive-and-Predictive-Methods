import logging
import re
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query
import polars as pl

from backend.api.models import schemas
from backend.services import crime_service

router = APIRouter()
logger = logging.getLogger(__name__)

# Regex to validate YYYY-MM format
YYYY_MM_REGEX = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


@router.get(
    "/crime/forces/monthly",
    response_model=List[Dict[str, Any]],
    summary="Get monthly crime trends by police force",
)
async def get_monthly_trends(
    from_date: str = Query(..., alias="from", description="Start month in YYYY-MM format"),
    to_date: str = Query(..., alias="to", description="End month in YYYY-MM format"),
):
    """
    Retrieves monthly aggregated crime counts for all police forces within a given date range.
    """
    if not YYYY_MM_REGEX.match(from_date) or not YYYY_MM_REGEX.match(to_date):
        raise HTTPException(
            status_code=400, detail="Invalid date format. Please use YYYY-MM."
        )

    try:
        data = crime_service.get_monthly_trends_by_force(from_date, to_date)
        return data
    except Exception as e:
        logger.exception("Failed to get monthly crime trends.")
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/crime/summary",
    summary="Get dataset summary (min/max month, counts)",
)
async def get_crime_summary():
    try:
        return crime_service.get_crime_summary()
    except Exception as e:
        logger.exception("Failed to get crime summary")
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/crime/forces/monthly/all",
    summary="Get pre-computed aggregated monthly trends (all data)",
)
async def get_monthly_trends_all():
    try:
        return crime_service.get_monthly_trends_aggregated()
    except Exception as e:
        logger.exception("Failed to get aggregated monthly trends")
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/crime/debug/sample",
    summary="Debug: return first 10 rows (sanitized)",
)
async def crime_debug_sample():
    try:
        df = crime_service.get_crime_dataframe()
        if df is None:
            return {"rows": []}
        show_cols = [c for c in ["Month","Falls within","LSOA name","Crime type","Latitude","Longitude"] if c in df.columns]
        sample = df.select(show_cols).head(10).to_dicts()
        return {"rows": sample, "columns": show_cols}
    except Exception as e:
        logger.exception("Debug sample failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/crime/debug/diagnostics",
    summary="Debug: return quick diagnostics about loaded crime parquet",
)
async def crime_debug_diagnostics():
    """Unauthenticated diagnostics endpoint for local development only."""
    try:
        data = crime_service.get_crime_diagnostics()
        return data
    except Exception as e:
        logger.exception("Diagnostics failed")
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/crime/debug/distincts",
    summary="Debug: distinct force & LSOA names (limited)",
)
async def crime_debug_distincts(limit: int = 50):
    try:
        df = crime_service.get_crime_dataframe()
        if df is None:
            return {"forces": [], "lsoas": []}
        df = crime_service.ensure_month_date(df)
        forces = (df.select("Falls within").unique().head(limit)["Falls within"].to_list() if "Falls within" in df.columns else [])
        lsoas = []
        if "LSOA name" in df.columns:
            lsoas = df.filter(pl.col("LSOA name").is_not_null()).select("LSOA name").unique().head(limit)["LSOA name"].to_list()
        return {"forces": forces, "lsoas": lsoas}
    except Exception as e:
        logger.exception("Debug distincts failed")
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/crime/debug/clear-cache",
    summary="Debug: clear crime data caches",
)
async def crime_debug_clear_cache():
    try:
        crime_service.clear_crime_caches()
        return {"ok": True}
    except Exception as e:
        logger.exception("Cache clear failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/crime/hotspots/latest",
    response_model=List[Dict[str, Any]],
    summary="Get latest crime hotspots for a force",
)
async def get_latest_hotspots(
    force: str | None = Query(None, description="The police force to query, e.g., 'City of London Police' (omit for all forces)")
):
    """
    Retrieves the top 50 LSOAs with the highest crime counts for the most recent month
    for a specific police force.
    """
    try:
        data = crime_service.get_latest_hotspots(force)
        return data
    except Exception as e:
        logger.exception(f"Failed to get hotspots for force: {force}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/crime/lsoa/series",
    response_model=List[Dict[str, Any]],
    summary="Get crime time series for an LSOA",
)
async def get_lsoa_series(
    lsoa: str = Query(..., description="The LSOA name to query, e.g., 'City of London 001A'")
):
    """
    Retrieves a monthly time series of crime counts, broken down by crime type,
    for a specific LSOA (Lower Layer Super Output Area).
    """
    try:
        data = crime_service.get_lsoa_series(lsoa)
        return data
    except Exception as e:
        logger.exception(f"Failed to get series for LSOA: {lsoa}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/crime/forces",
    summary="List distinct police forces",
)
async def list_forces(limit: int = 500):
    try:
        df = crime_service.get_crime_dataframe()
        if df is None or "Falls within" not in df.columns:
            return []
        forces = (
            df.select("Falls within").unique().sort("Falls within").head(limit)["Falls within"].to_list()
        )
        return forces
    except Exception as e:
        logger.exception("Failed to list forces")
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/crime/lsoas",
    summary="List distinct LSOA names (optionally filtered by force)",
)
async def list_lsoas(force: str | None = None, limit: int = 2000):
    try:
        logger.info(f"list_lsoas called with force={force}, limit={limit}")
        df = crime_service.get_crime_dataframe()
        if df is None or "LSOA name" not in df.columns:
            logger.warning("DataFrame is None or missing 'LSOA name' column")
            return []
        
        logger.info(f"DataFrame loaded: {df.height} rows, columns: {df.columns}")
        q = df.filter(pl.col("LSOA name").is_not_null())
        logger.info(f"After filtering null LSOAs: {q.height} rows")
        
        if force and "Falls within" in q.columns:
            logger.info(f"Filtering by force: {force}")
            q = q.filter(pl.col("Falls within") == force)
            logger.info(f"After filtering by force '{force}': {q.height} rows")
        elif force:
            logger.warning(f"Force filter requested ({force}) but 'Falls within' column not found")
        
        lsoas = q.select("LSOA name").unique().head(limit).sort("LSOA name")["LSOA name"].to_list()
        logger.info(f"Returning {len(lsoas)} unique LSOAs")
        return lsoas
    except Exception as e:
        logger.exception("Failed to list lsoas")
        raise HTTPException(status_code=500, detail=str(e))




