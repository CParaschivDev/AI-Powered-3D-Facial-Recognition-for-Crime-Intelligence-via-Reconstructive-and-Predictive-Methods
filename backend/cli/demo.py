def run_demo(threshold=0.5):
    from backend.services.watchlist_builder import build_watchlist
    from backend.database.dependencies import SessionLocal
    from backend.database.db_utils import load_watchlist_means
    from backend.core.paths import watchlist_roots, mine_root
    import glob
    from backend.utils.face_io import read_image
    from backend.models.utils.model_loader import get_recognition_model
    from backend.services.matcher import best_match

    build_watchlist(max_per_id=20)

    # pick one watchlist sample and one 'mine' sample
    wl = []
    for root in watchlist_roots():
        wl.extend(glob.glob(str((root/"**/*.jpg").resolve()), recursive=True))
    me = glob.glob(str((mine_root()/"**/*.jpg").resolve()), recursive=True)

    rec = get_recognition_model()

    # wanted
    img = read_image(wl[0]); e = rec.embed(img); pid, s = best_match(e)
    print({"probe":"watchlist", "best_id":pid, "score":s, "verdict":"WANTED" if s>=threshold else "NOT WANTED"})

    # not wanted
    img = read_image(me[0]); e = rec.embed(img); pid, s = best_match(e)
    print({"probe":"mine", "best_id":pid, "score":s, "verdict":"WANTED" if s>=threshold else "NOT WANTED"})