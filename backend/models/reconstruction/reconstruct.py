import numpy as np
import torch
import os
import logging
import cv2
from backend.models.geometry.template_face import TemplateFace, fit_similarity_transform

logger = logging.getLogger(__name__)


def lift_landmarks_2d_to_3d(lm2d: np.ndarray) -> np.ndarray:
    """
    Convert 2D landmarks (N,2) to pseudo-3D (N,3) using a simple depth heuristic.
    This assigns a structured depth curve so template fitting can recover
    reasonable 3D pose and proportions from 2D detections.

    The function expects 2D landmarks in normalized coordinates (e.g. in [-1, 1]).
    Returns landmarks in shape (N,3) with z values assigned by region.
    """
    lm2d = np.asarray(lm2d, dtype=np.float32)
    N = lm2d.shape[0]
    lm3d = np.zeros((N, 3), dtype=np.float32)

    # preserve x,y
    lm3d[:, :2] = lm2d

    # Canonical, skull-like depth anchors for stable fitting.
    # These are relative depth 'bumps' in arbitrary units; we'll normalize
    # and scale to template size below. Indices assume the landmark
    # detector ordering used elsewhere in the repo (e.g. 68/102-point sets).
    CANONICAL_DEPTHS = {
        30: 15,   # nose tip
        27: 8,    # nose bridge
        8:  5,    # chin area
        33: 5,    # upper lip
        57: 3,    # lower lip
        36: 2,    # left eye corner
        45: 2,    # right eye corner
        3: -2,    # jaw
        5: -2,    # jaw
        13: -3,   # jaw side
        14: -3,   # jaw side
    }

    # Apply canonical depths where available (ignore indices out of range)
    for idx, depth in CANONICAL_DEPTHS.items():
        if 0 <= idx < N:
            lm3d[idx, 2] = float(depth)

    # Default depth for other points remains zero; normalize and scale
    # so values are in a stable template-relative range.
    lm3d[:, 2] = lm3d[:, 2] - lm3d[:, 2].mean()
    max_abs = np.abs(lm3d[:, 2]).max() + 1e-6
    lm3d[:, 2] = lm3d[:, 2] / max_abs
    lm3d[:, 2] = lm3d[:, 2] * 12.0

    return lm3d

def overlay_on_rgb112(img112_rgb, reconstruction_model=None, landmark_model=None) -> np.ndarray:
    """
    Generate an overlay on the RGB face image.
    Uses the reconstruction model to get 3D vertices and projects them to 2D for wireframe overlay.

    Args:
        img112_rgb: RGB image of size 112x112
        reconstruction_model: Optional ReconstructionModel instance
        landmark_model: Optional LandmarkModel instance (for real landmark detection)

    Returns:
        Image with overlay applied
    """
    img = img112_rgb.copy()
    h, w, _ = img.shape

    try:
        if reconstruction_model is not None:
            # Get real landmarks if model available
            if landmark_model is not None:
                landmarks = landmark_model.predict(img112_rgb)
            else:
                # Fallback: use center point
                landmarks = np.array([[56, 56]])
            
            vertices, faces = reconstruction_model.reconstruct(img112_rgb, landmarks)

            # Project 3D to 2D using a stable orthographic-style projection.
            # Perspective division can explode when many z values are near zero
            # (which happens after centering the mesh). Use a scale based on
            # the mesh XY extent so the overlay is robust across inputs.
            center = np.array([w/2.0, h/2.0])

            verts_xy = vertices[:, :2]
            # compute extents in x,y
            min_xy = verts_xy.min(axis=0)
            max_xy = verts_xy.max(axis=0)
            range_xy = max_xy - min_xy
            max_range = max(range_xy.max(), 1e-6)

            # Occupy roughly 80% of the smaller image dimension (FLAME templates can be larger)
            target_px = min(w, h) * 0.8
            scale = float(target_px / max_range)

            vertices_2d = []
            for v in vertices:
                x, y = v[0], v[1]
                proj_x = center[0] + x * scale
                proj_y = center[1] - y * scale  # invert y for image coords
                # clamp to image bounds
                proj_x = max(0, min(w - 1, int(round(proj_x))))
                proj_y = max(0, min(h - 1, int(round(proj_y))))
                vertices_2d.append((proj_x, proj_y))

            # Draw wireframe: connect vertices according to faces
            for face in faces:
                for i in range(len(face)):
                    pt1 = vertices_2d[face[i]]
                    pt2 = vertices_2d[face[(i+1) % len(face)]]
                    cv2.line(img, pt1, pt2, (0, 255, 0), 1)
        else:
            # Fallback: draw a simple grid
            for x in range(0, w, 8):
                cv2.line(img, (x, 0), (x, h), (0, 255, 0), 1)
            for y in range(0, h, 8):
                cv2.line(img, (0, y), (w, y), (0, 255, 0), 1)

        return img
    except Exception as e:
        print(f"Warning: Could not generate proper overlay: {e}")
        return img112_rgb.copy()  # Return original image if overlay fails


class ReconstructionModel:
    """
    3D Face Reconstruction Model using simplified 3DMM approach.
    Reconstructs 3D face mesh from image using learned 3DMM parameters.
    """
    def __init__(self, model_path: str):
        """
        Initializes the reconstruction model.

        Args:
            model_path: Path to the trained model checkpoint.
        """
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Prefer FLAME template if provided in Data/data/
        # Prefer Data/data/... but also accept Data/... as alternative
        flame_obj = os.path.join('Data', 'data', 'head_template_mesh.obj')
        if not os.path.exists(flame_obj):
            flame_obj = os.path.join('Data', 'head_template_mesh.obj')
        flame_lm = os.path.join('Data', 'data', 'landmark_embedding.npy')
        if not os.path.exists(flame_lm):
            flame_lm = os.path.join('Data', 'landmark_embedding.npy')
        loaded_flame = False

        def _load_obj(path):
            verts = []
            faces = []
            with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
                for ln in fh:
                    ln = ln.strip()
                    if ln.startswith('v '):
                        parts = ln.split()
                        if len(parts) >= 4:
                            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                            verts.append([x, y, z])
                            # Safety guard: protect against extremely large OBJ files
                            if len(verts) > 500000:
                                raise ValueError('OBJ file contains too many vertices')
                    elif ln.startswith('f '):
                        parts = ln.split()[1:]
                        idxs = []
                        for p in parts:
                            # faces can be like v, v/t, v/t/n
                            v = p.split('/')[0]
                            try:
                                vi = int(v) - 1
                            except Exception:
                                vi = None
                            if vi is not None:
                                idxs.append(vi)
                        if len(idxs) >= 3:
                            # triangulate if necessary (assume triangles or quads)
                            if len(idxs) == 3:
                                faces.append(idxs)
                            else:
                                # fan triangulation
                                for i in range(1, len(idxs)-1):
                                    faces.append([idxs[0], idxs[i], idxs[i+1]])
            return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)

        if os.path.exists(flame_obj):
            try:
                bv, bf = _load_obj(flame_obj)
                if bv.size > 0 and bf.size > 0:
                    self.base_vertices = bv
                    self.base_faces = bf
                    # In FLAME mode we DO NOT generate random PCA bases.
                    # Signal that we are in canonical FLAME mode by clearing bases.
                    self.shape_basis = None
                    self.exp_basis = None
                    loaded_flame = True
                    logger.info(f'[ReconstructionModel] Loaded FLAME template from {flame_obj} (V={self.base_vertices.shape[0]}, F={len(self.base_faces)})')
            except Exception as e:
                logger.warning(f'Could not load FLAME OBJ {flame_obj}: {e}')

        if not loaded_flame:
            # Define a simplified face mesh topology (more realistic than a cube)
            self._create_base_face_mesh()

        # Try to locate a trained checkpoint
        self.checkpoint_paths = [
            os.path.join("logs", "reconstruction", "reconstruction_model.pth"),
            os.path.join("backend", "models", "reconstruction", "prnet_model.pth"),
            model_path,
        ]
        self.trained_model = None
        self._load_trained_model()
        
        # Load template face for landmark-based fitting
        self.template = TemplateFace()
        # If FLAME template was loaded above, override template geometry
        try:
            if hasattr(self, 'base_vertices') and hasattr(self, 'base_faces'):
                self.template.vertices = self.base_vertices.copy()
                self.template.faces = self.base_faces.copy()
        except Exception:
            pass

        # If user provided a landmark embedding for FLAME, load it
        try:
            if os.path.exists(flame_lm):
                lm_raw = np.load(flame_lm, allow_pickle=True)
                # If embedding is indices, convert to 3D landmarks
                if lm_raw.ndim == 1:
                    idxs = lm_raw.astype(int)
                    if hasattr(self, 'base_vertices') and len(self.base_vertices) > 0:
                        # guard indices
                        valid = idxs[(idxs >= 0) & (idxs < len(self.base_vertices))]
                        lm3 = self.base_vertices[valid]
                        self.template.landmarks = lm3
                        logger.info(f'[ReconstructionModel] Loaded FLAME landmark indices from {flame_lm} (N={lm3.shape[0]})')
                elif lm_raw.ndim == 2 and lm_raw.shape[1] == 3:
                        self.template.landmarks = lm_raw.astype(np.float32)
                        logger.info(f'[ReconstructionModel] Loaded FLAME landmark coordinates from {flame_lm}')
        except Exception as e:
                    logger.warning(f'Could not load landmark embedding {flame_lm}: {e}')

        # Expose these for easy debug access and for the temporary hard-reset
        # mode: `template_vertices` and `template_faces` are direct copies
        # of the neutral template and should not be mutated by debug runs.
        self.template_vertices = self.template.vertices.copy()
        self.template_faces = self.template.faces.copy()

        # Expose a flag for whether PCA-like bases are available (ellipsoid fallback)
        self.has_bases = (hasattr(self, 'shape_basis') and self.shape_basis is not None and
                          hasattr(self, 'exp_basis') and self.exp_basis is not None)
        logger.info(f'[ReconstructionModel] has_bases={self.has_bases}')

    def _create_base_face_mesh(self):
        """Create a high-density realistic face mesh topology."""
        # Generate a much denser mesh for smooth appearance
        
        # Parameters for mesh density - increase for smoother face
        lat_res = 40  # Latitude resolution (vertical)
        lon_res = 40  # Longitude resolution (horizontal)
        
        vertices = []
        
        # Generate vertices in a face-like ellipsoid shape
        for i in range(lat_res):
            lat = (i / (lat_res - 1) - 0.5) * np.pi * 0.9  # -0.45pi to 0.45pi (front hemisphere)
            for j in range(lon_res):
                lon = (j / (lon_res - 1) - 0.5) * np.pi * 1.1  # -0.55pi to 0.55pi (front face)
                
                # Simple smooth ellipsoid - no feature deformations
                x = 35 * np.cos(lat) * np.sin(lon)  # Width
                y = 45 * np.sin(lat)                 # Height (elongated vertically)
                z = 45 * np.cos(lat) * np.cos(lon)  # Depth
                
                vertices.append([x, y, z])
        
        self.base_vertices = np.array(vertices, dtype=np.float32)
        
        # Generate face indices for smooth triangulation
        faces = []
        for i in range(lat_res - 1):
            for j in range(lon_res - 1):
                # Two triangles per quad
                v0 = i * lon_res + j
                v1 = i * lon_res + (j + 1)
                v2 = (i + 1) * lon_res + j
                v3 = (i + 1) * lon_res + (j + 1)
                
                faces.append([v0, v1, v2])
                faces.append([v1, v3, v2])
        
        self.base_faces = np.array(faces, dtype=np.int32)
        
        # Create shape and expression bases
        num_vertices = len(self.base_vertices)
        num_shape_params = 199
        num_exp_params = 29

        # Create smooth, low-magnitude deformation bases
        np.random.seed(42)
        self.shape_basis = np.random.randn(num_shape_params, num_vertices * 3).astype(np.float32) * 0.03
        self.exp_basis = np.random.randn(num_exp_params, num_vertices * 3).astype(np.float32) * 0.02

        self.shape_basis = self.shape_basis.reshape(num_shape_params, num_vertices, 3)
        self.exp_basis = self.exp_basis.reshape(num_exp_params, num_vertices, 3)

    def _load_trained_model(self):
        """Load the trained reconstruction model."""
        try:
            for ck in self.checkpoint_paths:
                if ck and os.path.exists(ck):
                    logger.info(f"[ReconstructionModel] Found checkpoint at: {ck}")
                    ck_data = torch.load(ck, map_location=self.device)

                    # Handle different checkpoint formats
                    state_dict = ck_data.get('model_state_dict', ck_data) if isinstance(ck_data, dict) else ck_data

                    # Infer number of params from state_dict
                    weight_key = None
                    for k in state_dict.keys():
                        if k.endswith('param_predictor.weight'):
                            weight_key = k
                            break

                    if weight_key is None:
                        for k, v in state_dict.items():
                            if 'weight' in k and hasattr(v, 'shape') and len(v.shape) == 2:
                                weight_key = k
                                break

                    if weight_key is None:
                        logger.warning("Could not infer param predictor size from checkpoint")
                        continue

                    out_dim = state_dict[weight_key].shape[0]
                    num_params = out_dim

                    # Build the network architecture
                    class _ReconstructionNet(torch.nn.Module):
                        def __init__(self, num_params, predictor_in_features=None):
                            super().__init__()
                            # Simple backbone used during training/inference in this repo.
                            # Add a 2x pooling so a 224x224 input yields a 112x112 spatial map
                            # (matching many training pipelines that subsample once).
                            self.backbone = torch.nn.Sequential(
                                torch.nn.Conv2d(3, 64, 7, stride=1, padding=3, bias=False),
                                torch.nn.ReLU(inplace=True),
                                torch.nn.AvgPool2d(kernel_size=2, stride=2)
                            )

                            # With the 2x pooling above, a 224x224 input produces 64 x 112 x 112
                            # flattened features.
                            self._backbone_out = 64 * 112 * 112

                            # If the checkpoint's predictor expected a different in_features,
                            # we'll create a small adapter layer to map backbone output -> predictor_in_features.
                            if predictor_in_features is None:
                                predictor_in_features = self._backbone_out

                            if predictor_in_features != self._backbone_out:
                                self.adapter = torch.nn.Linear(self._backbone_out, predictor_in_features)
                            else:
                                self.adapter = None

                            self.param_predictor = torch.nn.Linear(predictor_in_features, num_params)

                        def forward(self, x):
                            feats = self.backbone(x)
                            feats = feats.view(feats.size(0), -1)
                            if self.adapter is not None:
                                feats = self.adapter(feats)
                            return self.param_predictor(feats)

                    # Determine predictor input features from checkpoint if present
                    predictor_in = None
                    try:
                        # weight_key typically like 'param_predictor.weight'
                        weight = state_dict[weight_key]
                        if hasattr(weight, 'shape') and len(weight.shape) == 2:
                            # shape = (out_dim, in_features)
                            predictor_in = int(weight.shape[1])
                            logger.info(f"[ReconstructionModel] Checkpoint predictor expects in_features={predictor_in}")
                    except Exception:
                        predictor_in = None

                    net = _ReconstructionNet(num_params, predictor_in).to(self.device)

                    try:
                        # Load state dict non-strictly so we accept missing/mismatched backbone keys.
                        net.load_state_dict(state_dict, strict=False)
                        self.trained_model = net.eval()
                        logger.info(f"[ReconstructionModel] Loaded trained net with {num_params} parameters (predictor_in={predictor_in})")
                        break
                    except Exception as e:
                        logger.warning(f"Could not load state_dict: {e}")
                        continue

            if self.trained_model is None:
                logger.info(f"[ReconstructionModel] No trained checkpoint found - using random deformation")
        except Exception as e:
            logger.warning(f"[ReconstructionModel] Error loading model: {e}. Using random deformation.")

    def reconstruct(self, image: np.ndarray, landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct mesh from an input image (or preprocessed tensor) and optional landmarks.

        This will use a loaded `trained_model` when available to predict 3DMM
        parameters and convert them into vertex offsets using the
        `shape_basis` and `exp_basis`. Several fallback heuristics are used
        so the method can operate with different checkpoint parameterizations:
          - If model predicts (ns + ne) coefficients, treat them as shape+exp coeffs
          - If model predicts (V*3) values, treat them as per-vertex offsets
          - Otherwise return the neutral template (safe fallback)

        Args:
            image: either a numpy HxWx3 uint8/float image or a torch.Tensor CxHxW
            landmarks: optional landmarks in image or normalized coordinates

        Returns:
            vertices (V,3), faces (F,3)
        """

        # Quick fallbacks: if no trained model, return neutral template (but sanitized)
        if self.trained_model is None:
            vertices = self.template_vertices.copy()
            faces = self.template_faces.copy()
            vertices = self._sanitize_mesh(vertices)
            return vertices, faces

        # Prepare input tensor for the trained model
        try:
            import torch as _torch

            if isinstance(image, _torch.Tensor):
                # assume (C,H,W) normalized tensor as used in training
                inp = image.unsqueeze(0).to(self.device)
            else:
                # assume numpy HxWx3 in range 0..255 or 0..1
                img = np.asarray(image)
                if img.dtype != np.float32:
                    img = img.astype(np.float32) / 255.0
                # Resize/normalize roughly as the training transform expects 224x224
                try:
                    import cv2 as _cv2
                    h, w = img.shape[:2]
                    if (h, w) != (224, 224):
                        img = _cv2.resize(img, (224, 224), interpolation=_cv2.INTER_LINEAR)
                except Exception:
                    pass
                # from HWC to CHW and normalize with ImageNet stats
                img_chw = np.transpose(img, (2, 0, 1))
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3,1,1)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3,1,1)
                img_chw = (img_chw - mean) / (std + 1e-8)
                inp = _torch.from_numpy(img_chw).unsqueeze(0).to(self.device)

            # Predict parameters
            with _torch.no_grad():
                pred = self.trained_model(inp)
            pred = pred.detach().cpu().numpy()
            if pred.ndim == 2 and pred.shape[0] == 1:
                pred = pred[0]

            # Map predicted parameters to vertex offsets
            V = self.base_vertices.shape[0]
            has_bases = getattr(self, 'has_bases', False)

            if has_bases:
                # Legacy/ellipsoid mode: use PCA-like bases
                ns = self.shape_basis.shape[0]
                ne = self.exp_basis.shape[0]

                # Case A: (ns + ne) coefficients
                if pred.size == (ns + ne):
                    shape_coeffs = pred[:ns].astype(np.float32)
                    exp_coeffs = pred[ns:ns+ne].astype(np.float32)

                    shape_offset = np.tensordot(shape_coeffs, self.shape_basis, axes=(0,0))
                    exp_offset = np.tensordot(exp_coeffs, self.exp_basis, axes=(0,0))
                    vertices = self.base_vertices.copy() + shape_offset + exp_offset
                    logger.debug('ReconstructionModel: Ellipsoid mode -> used shape+exp coefficients')

                # Case A2: only shape coefficients predicted (ns)
                elif pred.size == ns:
                    shape_coeffs = pred[:ns].astype(np.float32)
                    exp_coeffs = np.zeros((ne,), dtype=np.float32)

                    shape_offset = np.tensordot(shape_coeffs, self.shape_basis, axes=(0,0))
                    exp_offset = np.zeros_like(shape_offset)
                    vertices = self.base_vertices.copy() + shape_offset + exp_offset
                    logger.debug('ReconstructionModel: Ellipsoid mode -> used shape-only coefficients')

                # Case B: direct per-vertex offsets flattened (V*3)
                elif pred.size == V * 3:
                    offsets = pred.reshape(V, 3).astype(np.float32)
                    vertices = self.base_vertices.copy() + offsets
                    logger.debug('ReconstructionModel: Ellipsoid mode -> used direct per-vertex offsets')

                else:
                    # Unknown parameterization -> fallback to neutral template
                    logger.warning(f"Unrecognized reconstruction parameter size: {pred.size}. Falling back to neutral template.")
                    vertices = self.template_vertices.copy()

            else:
                # FLAME canonical mode: do NOT use random PCA bases. Only accept direct per-vertex offsets.
                if pred.size == V * 3:
                    offsets = pred.reshape(V, 3).astype(np.float32)
                    vertices = self.base_vertices.copy() + offsets
                    logger.debug('ReconstructionModel: FLAME mode -> applied direct per-vertex offsets')
                else:
                    # Use neutral FLAME template (landmark-based alignment/refinement will still apply)
                    vertices = self.base_vertices.copy()
                    logger.debug('ReconstructionModel: FLAME mode -> using neutral template (no offsets)')

            # Optional: if landmarks provided, try to align template landmarks to image
            try:
                if landmarks is not None and hasattr(self.template, 'landmarks'):
                    # If landmarks look 2D, lift to 3D pseudo-depth for alignment
                    lm = np.asarray(landmarks)
                    if lm.ndim == 2 and lm.shape[1] == 2:
                        lm3 = lift_landmarks_2d_to_3d(lm)
                    else:
                        lm3 = lm

                    # Use template landmarks (3D) to fit similarity transform
                    tpl_lm = self.template.landmarks
                    if tpl_lm.shape[0] == lm3.shape[0]:
                        R, s, t = fit_similarity_transform(tpl_lm, lm3)
                        vertices = (s * (R @ vertices.T)).T + t.reshape(1, 3)
            except Exception:
                pass

            # Sanitize and lightly refine
            vertices = self._sanitize_mesh(vertices)
            try:
                # If landmarks are available and match template size, refine non-rigidly
                if landmarks is not None and self.template.landmarks.shape[0] == np.asarray(landmarks).reshape(-1,2).shape[0]:
                    tgt_lm3 = lift_landmarks_2d_to_3d(np.asarray(landmarks).reshape(-1,2))
                    tpl_lm3 = self.template.landmarks
                    vertices = self._nonrigid_refine(vertices, tpl_lm3, tgt_lm3, radius=10.0, strength=0.6, iterations=2)
            except Exception:
                pass

            # Choose face topology matching the vertex set when possible
            if vertices.shape[0] == getattr(self, 'base_vertices', np.array([])).shape[0]:
                faces = self.base_faces.copy()
            else:
                faces = self.template_faces.copy()
            return vertices, faces

        except Exception as e:
            logger.warning(f"Reconstruction failed during inference: {e}")
            vertices = self.template_vertices.copy()
            faces = self.template_faces.copy()
            return vertices, faces
    
    def _sanitize_mesh(self, vertices: np.ndarray) -> np.ndarray:
        """Sanitize mesh: remove NaN/Inf, center, normalize scale."""
        # Remove NaN and Inf
        vertices = np.nan_to_num(vertices, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Center on origin
        centroid = np.mean(vertices, axis=0)
        vertices = vertices - centroid
        
        # Normalize to fixed radius (prevents explosion)
        max_radius = np.max(np.linalg.norm(vertices, axis=1))
        if max_radius > 0:
            vertices = vertices * (60.0 / max_radius)  # Normalize to radius 60
        
        return vertices

    def _nonrigid_refine(self, vertices: np.ndarray, template_lm: np.ndarray, target_lm: np.ndarray,
                         radius: float = 0.15, strength: float = 0.8, iterations: int = 3) -> np.ndarray:
        """Apply smooth, local RBF-style deformations to move vertices toward target landmarks.

        Args:
            vertices: (V,3) array of current vertex positions.
            template_lm: (N,3) template landmark positions (in same coords as vertices).
            target_lm: (N,3) target landmark positions to reach.
            radius: influence radius (in same units as vertices).
            strength: global multiplier for displacement (0..1).
            iterations: number of smoothing iterations.

        Returns:
            Deformed vertices (V,3)
        """
        verts = vertices.copy().astype(np.float32)
        V = verts.shape[0]
        if template_lm.shape[0] == 0 or target_lm.shape[0] == 0:
            return verts

        # Precompute vertex positions as array for fast distance computation
        for it in range(max(1, iterations)):
            disp = np.zeros_like(verts)
            weight_sum = np.zeros((V, 1), dtype=np.float32)

            for (src, dst) in zip(template_lm, target_lm):
                delta = (dst - src).astype(np.float32)
                if np.linalg.norm(delta) < 1e-6:
                    continue

                # distances to all vertices
                dists = np.linalg.norm(verts - src.reshape(1, 3), axis=1)
                mask = dists <= radius
                if not np.any(mask):
                    continue
                # Gaussian-like weight
                w = np.exp(- (dists[mask] / radius) ** 2)
                w = w.reshape(-1, 1)

                disp[mask] += delta.reshape(1, 3) * w
                weight_sum[mask] += w

            # Normalize by weights and apply strength
            nonzero = weight_sum.squeeze() > 0
            if np.any(nonzero):
                disp[nonzero] = disp[nonzero] / weight_sum[nonzero]
                verts[nonzero] += disp[nonzero] * float(strength)

            # Gentle Laplacian smoothing to keep mesh smooth
            try:
                # build simple adjacency from spatial proximity (fast approx)
                from scipy.spatial import cKDTree
                tree = cKDTree(verts)
                neigh = tree.query_ball_tree(tree, r=radius * 0.6)
                smooth_disp = np.zeros_like(verts)
                for i, nbrs in enumerate(neigh):
                    if len(nbrs) <= 1:
                        continue
                    nbrs = np.array(nbrs, dtype=int)
                    smooth_disp[i] = np.mean(verts[nbrs], axis=0) - verts[i]
                verts += 0.25 * smooth_disp
            except Exception:
                # if scipy not available, skip smoothing
                pass

        return verts

    def _nonrigid_refine_xy(self, vertices: np.ndarray, template_lm: np.ndarray, target_lm: np.ndarray,
                            radius: float = 0.15, strength: float = 0.6, iterations: int = 2) -> np.ndarray:
        """Apply smooth, local deformations but only adjust X and Y components.

        This keeps depth stable while allowing observable facial silhouette
        changes in the image plane.
        """
        verts = vertices.copy().astype(np.float32)
        V = verts.shape[0]
        if template_lm.shape[0] == 0 or target_lm.shape[0] == 0:
            return verts

        for it in range(max(1, iterations)):
            disp = np.zeros_like(verts)
            weight_sum = np.zeros((V, 1), dtype=np.float32)

            for (src, dst) in zip(template_lm, target_lm):
                delta = (dst - src).astype(np.float32)
                # only use x,y components of the target displacement
                delta_xy = delta.copy()
                delta_xy[2] = 0.0
                if np.linalg.norm(delta_xy) < 1e-6:
                    continue

                dists = np.linalg.norm(verts - src.reshape(1, 3), axis=1)
                mask = dists <= radius
                if not np.any(mask):
                    continue

                w = np.exp(- (dists[mask] / radius) ** 2).reshape(-1, 1)
                disp[mask] += delta_xy.reshape(1, 3) * w
                weight_sum[mask] += w

            nonzero = weight_sum.squeeze() > 0
            if np.any(nonzero):
                disp[nonzero] = disp[nonzero] / weight_sum[nonzero]
                # apply only x,y adjustments with strength
                verts[nonzero, :2] += disp[nonzero, :2] * float(strength)

            # Gentle smoothing using proximity
            try:
                from scipy.spatial import cKDTree
                tree = cKDTree(verts)
                neigh = tree.query_ball_tree(tree, r=radius * 0.6)
                smooth_disp = np.zeros_like(verts)
                for i, nbrs in enumerate(neigh):
                    if len(nbrs) <= 1:
                        continue
                    nbrs = np.array(nbrs, dtype=int)
                    smooth_disp[i, :2] = np.mean(verts[nbrs, :2], axis=0) - verts[i, :2]
                verts[:, :2] += 0.25 * smooth_disp[:, :2]
            except Exception:
                pass

        return verts
