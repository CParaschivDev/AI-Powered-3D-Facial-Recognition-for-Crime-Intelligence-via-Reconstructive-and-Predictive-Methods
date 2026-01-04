from pathlib import Path
import cv2
import numpy as np
import contextlib
import os
import sys
from typing import Iterable, Tuple


def iter_images(root: Path):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    for p in root.rglob("*"):
        if p.suffix.lower() in exts:
            yield p


def id_from_path(root: Path, img: Path) -> str:
    rel = img.relative_to(root)
    # top folder under watchlist root is the identity
    return rel.parts[0] if len(rel.parts) > 1 else img.stem


def save_aligned_rgb112(aligned_rgb, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), aligned_rgb[:, :, ::-1])  # RGB->BGR for OpenCV


def read_image(path: Path):
    """Read image in a Windows/unicode-safe way and return an OpenCV BGR image or None.

    Uses numpy.fromfile + cv2.imdecode which avoids issues with OpenCV not supporting
    wide/unicode paths on some Windows builds. Accepts a Path or string.
    """
    p = str(path)
    try:
        arr = np.fromfile(p, dtype=np.uint8)
        if arr.size == 0:
            return None

        # libpng prints warnings (e.g. "iCCP: known incorrect sRGB profile")
        # at the C-level to stderr (fd 2). Python's redirect_stderr won't
        # capture those; temporarily redirect the OS-level stderr fd to
        # os.devnull while calling cv2.imdecode to fully silence them.
        @contextlib.contextmanager
        def _suppress_c_stderr():
            try:
                err_fd = sys.stderr.fileno()
            except Exception:
                # sys.stderr not a real fd, fallback
                yield
                return
            devnull_fd = os.open(os.devnull, os.O_RDWR)
            saved_fd = os.dup(err_fd)
            try:
                os.dup2(devnull_fd, err_fd)
                yield
            finally:
                os.dup2(saved_fd, err_fd)
                os.close(saved_fd)
                os.close(devnull_fd)

        with _suppress_c_stderr():
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        # fallback to regular imread for other platforms
        try:
            return cv2.imread(p)
        except Exception:
            return None
