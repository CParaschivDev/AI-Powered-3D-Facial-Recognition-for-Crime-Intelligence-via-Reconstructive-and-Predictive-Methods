from pathlib import Path
from typing import List
from backend.core.config import settings

def _p(*parts) -> Path:
    return Path(settings.DATA_ROOT, *parts).resolve()

def watchlist_roots() -> List[Path]:
    return [ _p(d) for d in settings.WATCHLIST_DIRS if _p(d).exists() ]

def mine_root() -> Path:
    return _p(settings.MINE_DIR)

def aflw_root() -> Path:
    return _p(settings.AFLW2K3D_DIR)

def uk_police_root() -> Path:
    return _p(settings.UK_POLICE_DIR)

# Path to reports directory (used for overlays, benchmarks, etc.)
REPORTS_PATH = Path("reports").resolve()
