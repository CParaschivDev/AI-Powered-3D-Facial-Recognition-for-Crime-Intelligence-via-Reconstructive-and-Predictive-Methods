import numpy as np
from backend.database.dependencies import SessionLocal
from backend.database.db_utils import load_watchlist_means

def cosine(a,b): 
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

def best_match(e: np.ndarray) -> tuple[str, float] | None:
    db = SessionLocal()
    means = load_watchlist_means(db)  # dict[str, np.ndarray]
    db.close()
    if not means: return None
    ids = list(means.keys())
    M = np.stack([means[i] for i in ids]).astype("float32")
    sims = (M @ e) / (np.linalg.norm(M,axis=1)*np.linalg.norm(e)+1e-12)
    j = int(np.argmax(sims)); return ids[j], float(sims[j])
