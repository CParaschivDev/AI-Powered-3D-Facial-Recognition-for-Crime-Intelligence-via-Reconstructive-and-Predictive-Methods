"""
Render FLAME template mesh with supervised landmarks overlay.
Saves output to `docs/figures/flame_template_landmarks.png`.

This script attempts to be robust to a few common NPZ/NPY layouts.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
npz_path = os.path.join(ROOT, 'template_face.npz')
emb_path = os.path.join(ROOT, 'landmark_embeddings.npy')
output_dir = os.path.join(ROOT, 'docs', 'figures')
output_path = os.path.join(output_dir, 'flame_template_landmarks.png')

os.makedirs(output_dir, exist_ok=True)

print('Looking for template npz at', npz_path)
# If not present, try to locate the files anywhere under the workspace root.
if not os.path.exists(npz_path):
    found = None
    for dirpath, dirnames, filenames in os.walk(ROOT):
        if 'template_face.npz' in filenames:
            found = os.path.join(dirpath, 'template_face.npz')
            break
    if found:
        print('Found template_face.npz at', found)
        npz_path = found
    else:
        print('ERROR: template_face.npz not found at', npz_path)
        sys.exit(2)

data = np.load(npz_path, allow_pickle=True)
print('NPZ keys:', list(data.keys()))

# Heuristics to find vertices and faces
vertices = None
faces = None
for k in data.keys():
    lk = k.lower()
    if lk in ('v', 'verts', 'vertices', 'vertices3d', 'pos'):
        vertices = data[k]
    if lk in ('f', 'faces', 'tri', 'triangles'):
        faces = data[k]

# Fallback: try common names
if vertices is None:
    for k in data.keys():
        val = data[k]
        if isinstance(val, np.ndarray) and val.ndim == 2 and val.shape[1] in (3,):
            vertices = val
            break

if faces is None:
    for k in data.keys():
        val = data[k]
        if isinstance(val, np.ndarray) and val.ndim == 2 and val.shape[1] in (3,):
            # be conservative: only pick as faces if values look like ints
            try:
                if np.issubdtype(val.dtype, np.integer) or (val.max() < (len(vertices) if vertices is not None else 1)):
                    faces = val
                    break
            except Exception:
                pass

if vertices is None:
    print('ERROR: Could not locate vertex array inside npz. Keys:', list(data.keys()))
    sys.exit(3)

vertices = np.asarray(vertices)
print('Vertices shape:', vertices.shape)
if faces is not None:
    faces = np.asarray(faces, dtype=int)
    print('Faces shape:', faces.shape)
else:
    print('No faces found; will render vertices only.')

# Load landmark embeddings (try to locate if not at repo root)
if not os.path.exists(emb_path):
    found_emb = None
    for dirpath, dirnames, filenames in os.walk(ROOT):
        if 'landmark_embeddings.npy' in filenames:
            found_emb = os.path.join(dirpath, 'landmark_embeddings.npy')
            break
    if found_emb:
        print('Found landmark_embeddings.npy at', found_emb)
        emb_path = found_emb
    else:
        print('WARNING: landmark_embeddings.npy not found at', emb_path)
        landmark_points = None
        emb = None

if emb is None and os.path.exists(emb_path):
    emb = np.load(emb_path, allow_pickle=True)
    print('Loaded landmark_embeddings shape/dtype:', getattr(emb, 'shape', None), getattr(emb, 'dtype', None))
    # Attempt to resolve landmark points robustly
    pts = []
    try:
        if np.issubdtype(emb.dtype, np.integer):
            idx = np.asarray(emb, dtype=int)
            pts = vertices[idx]
        else:
            # object or float: iterate
            for e in emb:
                if isinstance(e, (np.integer, int)):
                    pts.append(vertices[int(e)])
                else:
                    a = np.asarray(e)
                    if a.ndim == 1 and a.size == 1:
                        pts.append(vertices[int(a[0])])
                    elif a.ndim == 1 and a.size == 3 and np.issubdtype(a.dtype, np.integer):
                        # possibly indices triangle -> pick first
                        pts.append(vertices[int(a[0])])
                    elif a.ndim == 1 and a.size == 3 and np.issubdtype(a.dtype, np.floating):
                        # might be barycentric weights without face indices: fallback to first vertex
                        pts.append(vertices[0])
                    else:
                        # try first item
                        try:
                            pts.append(vertices[int(np.round(a.flat[0]))])
                        except Exception:
                            pts.append(vertices[0])
        landmark_points = np.asarray(pts)
        if landmark_points.size == 0:
            landmark_points = None
    except Exception as exc:
        print('Failed to decode landmark embeddings:', exc)
        landmark_points = None

print('Landmark points:', None if landmark_points is None else landmark_points.shape)

# Create figure
fig = plt.figure(figsize=(6, 6), dpi=150)
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect((1,1,1))
ax.axis('off')

# Draw mesh
if faces is not None:
    mesh_verts = vertices
    tri_verts = mesh_verts[faces]
    collection = Poly3DCollection(tri_verts, facecolor=(0.8,0.8,0.8,0.95), edgecolor=(0.4,0.4,0.4,0.2), linewidths=0.1)
    ax.add_collection3d(collection)
else:
    ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], s=0.5, color='lightgray')

# Plot landmarks
if landmark_points is not None:
    ax.scatter(landmark_points[:,0], landmark_points[:,1], landmark_points[:,2], color='red', s=20)
    try:
        for i, p in enumerate(landmark_points):
            ax.text(p[0], p[1], p[2], str(i), color='black', fontsize=6)
    except Exception:
        pass

# set view
x, y, z = vertices[:,0], vertices[:,1], vertices[:,2]
max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
mid_x = (x.max()+x.min()) * 0.5
mid_y = (y.max()+y.min()) * 0.5
mid_z = (z.max()+z.min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)
ax.view_init(elev=20, azim=30)

plt.tight_layout()
plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
print('Saved', output_path)
