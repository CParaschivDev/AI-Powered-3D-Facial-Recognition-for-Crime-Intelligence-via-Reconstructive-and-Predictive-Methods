import logging
from pathlib import Path

from backend.core.config import settings
from backend.services.crime_context import build_parquet

logger = logging.getLogger(__name__)

# Note: Celery task removed - analytics now run synchronously
def refresh_crime_parquet_sync(force_filter: str | None = None) -> str:
    """
    Build the consolidated crime parquet into the preferred processed location:
    <DATA_ROOT>/processed/crime.parquet
    """
    if not settings.DATA_ROOT:
        msg = "DATA_ROOT not configured; cannot refresh crime parquet."
        logger.error(msg)
        return msg

    out = Path(settings.DATA_ROOT).resolve() / "processed" / "crime.parquet"
    try:
        path = build_parquet(out, force_filter=force_filter)
        logger.info("Crime parquet refreshed at %s", path)
        return str(path)
    except Exception as e:
        logger.exception("Failed to refresh crime parquet")
        raise
