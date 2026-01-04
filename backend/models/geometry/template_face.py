"""Template face mesh and landmark fitting utilities.

Behavior changes:
- If a FLAME OBJ is present in `Data/data/head_template_mesh.obj` or
    `Data/head_template_mesh.obj` then we do not attempt to load a cached
    `Data/template_face.npz`. This avoids silently overwriting the FLAME
    driven template with an on-disk cache.
"""
import os
import math
import logging
import numpy as np

logger = logging.getLogger(__name__)


TEMPLATE_PATH = os.path.join("Data", "template_face.npz")
FLAME_PATHS = [
        os.path.join("Data", "data", "head_template_mesh.obj"),
        os.path.join("Data", "head_template_mesh.obj"),
]


def fit_similarity_transform(src: np.ndarray, dst: np.ndarray):
    """
    Compute similarity transform (R, s, t) that maps src -> dst.

    Args:
        src: (N, 3) array of 3D points
        dst: (N, 3) array of 3D points

    Returns:
        R: (3, 3) rotation matrix
        s: scalar scale
        t: (3,) translation vector
    """
    assert src.shape == dst.shape
    assert src.shape[1] == 3

    src_mean = src.mean(axis=0, keepdims=True)
    dst_mean = dst.mean(axis=0, keepdims=True)

    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    # Compute covariance
    H = src_centered.T @ dst_centered / src.shape[0]

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Reflection handling
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1.0
        R = Vt.T @ U.T

    # Scale
    var_src = np.sum(src_centered ** 2) / src.shape[0]
    s = np.sum(S) / (var_src + 1e-8)

    t = dst_mean.reshape(3) - s * (R @ src_mean.reshape(3))

    return R.astype(np.float32), float(s), t.astype(np.float32)


class TemplateFace:
    """Template face mesh with canonical landmark positions."""
    
    def __init__(self):
        # If a FLAME OBJ is present we should not reuse a cached NPZ. The
        # ReconstructionModel drives the template geometry from FLAME and will
        # overwrite `self.vertices` and `self.faces` later. Only load a cached
        # TEMPLATE_PATH when no FLAME OBJ is present.
        flame_exists = any(os.path.exists(p) for p in FLAME_PATHS)

        if flame_exists:
            # Create a sensible, minimal template (not saved) that will be
            # replaced by the FLAME geometry by ReconstructionModel.
            self.vertices, self.faces, self.landmarks = self._create_default_head(full_head=False)
            logger.debug("FLAME OBJ detected; skipping cached template load and using ephemeral default template")
            return

        # Load existing template if available (only when no FLAME OBJ present)
        if os.path.exists(TEMPLATE_PATH):
            data = np.load(TEMPLATE_PATH)
            self.vertices = data["vertices"].astype(np.float32)  # (V, 3)
            self.faces = data["faces"].astype(np.int32)          # (F, 3)
            self.landmarks = data["landmarks"].astype(np.float32)  # (N, 3)

            # Basic sanity checks: non-empty, no NaNs, reasonable vertex count,
            # and reasonable radius. If any check fails, regenerate a full head
            # template and overwrite the NPZ to avoid silently reusing garbage.
            valid = True
            if self.vertices.size == 0:
                valid = False
            if np.isnan(self.vertices).any() or np.isinf(self.vertices).any():
                valid = False
            if self.vertices.shape[0] < 3000:
                valid = False
            try:
                centered = self.vertices - self.vertices.mean(axis=0, keepdims=True)
                max_radius = float(np.max(np.linalg.norm(centered, axis=1)))
                if not (5.0 <= max_radius <= 200.0):
                    valid = False
            except Exception:
                valid = False

            if not valid:
                # Downgrade this message to debug — a regenerated template is
                # non-fatal and happens in a limited set of environments.
                logger.debug("Existing template invalid or too small; regenerating full head template")
                self.vertices, self.faces, self.landmarks = self._create_default_head(full_head=True)
                os.makedirs(os.path.dirname(TEMPLATE_PATH) if os.path.dirname(TEMPLATE_PATH) else ".", exist_ok=True)
                np.savez(
                    TEMPLATE_PATH,
                    vertices=self.vertices,
                    faces=self.faces,
                    landmarks=self.landmarks,
                )
        else:
            # Create a full head template and save it so future runs load from disk.
            self.vertices, self.faces, self.landmarks = self._create_default_head(full_head=True)
            os.makedirs(os.path.dirname(TEMPLATE_PATH) if os.path.dirname(TEMPLATE_PATH) else ".", exist_ok=True)
            np.savez(
                TEMPLATE_PATH,
                vertices=self.vertices,
                faces=self.faces,
                landmarks=self.landmarks,
            )

    def _create_default_head(self, full_head: bool = False):
        """
        Build a neutral humanoid head from an ellipsoid and define
        canonical landmark points on it.
        """
        import numpy as _np
        import math as _math

        # Resolution of the template surface
        lat_res = 64   # vertical
        lon_res = 96 if full_head else 48   # horizontal (wider when full head)

        vertices = _np.zeros((lat_res * lon_res, 3), dtype=_np.float32)

        # -------- 1) Build a more human-like head shape --------
        for i in range(lat_res):
            # v: 0 (chin) -> 1 (top of head)
            v = i / (lat_res - 1)

            # vertical position: stretch a bit so chin is lower
            y = (v - 0.45) * 2.2  # shift down slightly

            # base radius in the horizontal plane
            base_radius = 1.0 - 0.20 * (v - 0.5) ** 2  # wider mid-face, slimmer top/bottom

            # cheeks: widest around mid–lower face (stronger)
            cheek_boost = 1.0 + 0.45 * math.exp(-((v - 0.55) ** 2) / 0.02)

            # jaw: narrow more strongly near the chin (stronger taper)
            if v < 0.25:
                jaw_taper = 0.35 + 2.2 * v  # 0.35 at chin → ~0.9 at v=0.25
            else:
                jaw_taper = 1.0

            # final width scaling
            width_scale = base_radius * cheek_boost * jaw_taper

            for j in range(lon_res):
                # t: full wrap when full_head -> -pi .. +pi, otherwise front-facing
                if full_head:
                    t = (j / (lon_res - 1) - 0.5) * 2.0 * _math.pi
                else:
                    t = (j / (lon_res - 1) - 0.5) * _math.pi

                # base ellipse in x–z plane
                x = _math.cos(t) * width_scale
                z = _math.sin(t) * base_radius * 0.9  # slightly compressed in depth

                # ---- Nose ridge: push central area forward (stronger) ----
                nose_band = _math.exp(-((v - 0.55) ** 2) / 0.015)  # vertical band
                nose_dir = _math.exp(-(t / 0.6) ** 2)              # around center horizontally
                z += 0.55 * nose_band * nose_dir

                # ---- Chin: protrude and narrow at very bottom (stronger) ----
                if v < 0.18:
                    chin_amt = (0.18 - v) / 0.18  # 1 at chin, 0 at 0.18
                    z += 0.65 * chin_amt
                    x *= 0.6 + 0.3 * (1.0 - chin_amt)  # narrower at very bottom

                # ---- Eye sockets: indent slightly above nose (deeper) ----
                eye_band = _math.exp(-((v - 0.70) ** 2) / 0.02)
                left_eye = _math.exp(-((t + 0.45) ** 2) / 0.08)
                right_eye = _math.exp(-((t - 0.45) ** 2) / 0.08)
                z -= 0.45 * eye_band * (left_eye + right_eye)

                # ---- Forehead: gently recede backward (slightly stronger) ----
                if v > 0.75:
                    forehead = (v - 0.75) / 0.25
                    z -= 0.35 * forehead

                # ---- Back of head rounding when full_head is True ----
                if full_head:
                    # Slight taper towards the back to form skull curvature
                    back_boost = _math.cos(t) * 0.15  # pushes rear area slightly
                    z -= back_boost

                idx = i * lon_res + j
                vertices[idx] = (x, y, z)

        # -------- 2) Build faces (grid triangles) --------
        faces = []
        if full_head:
            # wrap longitude (connect last column back to first)
            for i in range(lat_res - 1):
                for j in range(lon_res):
                    v0 = i * lon_res + j
                    v1 = i * lon_res + ((j + 1) % lon_res)
                    v2 = (i + 1) * lon_res + j
                    v3 = (i + 1) * lon_res + ((j + 1) % lon_res)
                    faces.append([v0, v2, v1])
                    faces.append([v1, v2, v3])
        else:
            for i in range(lat_res - 1):
                for j in range(lon_res - 1):
                    v0 = i * lon_res + j
                    v1 = i * lon_res + (j + 1)
                    v2 = (i + 1) * lon_res + j
                    v3 = (i + 1) * lon_res + (j + 1)
                    faces.append([v0, v2, v1])
                    faces.append([v1, v2, v3])

        faces = _np.asarray(faces, dtype=_np.int32)

        # -------- 3) Define 102 canonical landmarks on the template --------
        landmarks = []

        def vid(ii, jj):
            ii = int(_np.clip(ii, 0, lat_res - 1))
            jj = int(_np.clip(jj, 0, lon_res - 1))
            return ii * lon_res + jj

        mid_lat = int(lat_res * 0.55)
        eye_lat = int(lat_res * 0.70)
        brow_lat = int(lat_res * 0.78)
        chin_lat = int(lat_res * 0.02)

        mid_lon = lon_res // 2
        off3 = 3
        off4 = 4
        off6 = 6
        off8 = 8

        # Jawline / chin region
        landmarks.append(vertices[vid(chin_lat, mid_lon)])            # chin center
        landmarks.append(vertices[vid(chin_lat + 2, mid_lon - off6)]) # jaw left
        landmarks.append(vertices[vid(chin_lat + 2, mid_lon + off6)]) # jaw right
        landmarks.append(vertices[vid(chin_lat + 4, mid_lon - off8)]) # jaw far left
        landmarks.append(vertices[vid(chin_lat + 4, mid_lon + off8)]) # jaw far right

        # Nose
        landmarks.append(vertices[vid(mid_lat, mid_lon)])             # nose tip
        landmarks.append(vertices[vid(mid_lat - 2, mid_lon)])         # nose bridge
        landmarks.append(vertices[vid(mid_lat, mid_lon - off4)])      # nose left
        landmarks.append(vertices[vid(mid_lat, mid_lon + off4)])      # nose right

        # Eyes
        landmarks.append(vertices[vid(eye_lat, mid_lon - off6)])      # left eye outer
        landmarks.append(vertices[vid(eye_lat, mid_lon - off3)])      # left eye inner
        landmarks.append(vertices[vid(eye_lat, mid_lon + off6)])      # right eye outer

        # Eyebrows
        landmarks.append(vertices[vid(brow_lat, mid_lon - off6)])     # left brow
        landmarks.append(vertices[vid(brow_lat, mid_lon + off6)])     # right brow

        # Cheeks
        cheek_lat = int(lat_res * 0.48)
        landmarks.append(vertices[vid(cheek_lat, mid_lon - off8)])    # left cheek
        landmarks.append(vertices[vid(cheek_lat, mid_lon + off8)])    # right cheek

        # Fill the rest systematically over the face area to reach 102
        for ii in range(4, lat_res - 4, 4):
            for jj in range(4, lon_res - 4, 4):
                if len(landmarks) >= 102:
                    break
                # When generating a full head, prefer front-facing longitudes
                # around mid_lon to populate meaningful facial points.
                jj_sel = mid_lon + (jj - mid_lon)
                landmarks.append(vertices[vid(ii, jj_sel)])
            if len(landmarks) >= 102:
                break

        landmarks = _np.asarray(landmarks[:102], dtype=_np.float32)

        return vertices.astype(_np.float32), faces.astype(_np.int32), landmarks
