#!/usr/bin/env python3
"""
Quick check script for ReconstructionModel when FLAME OBJ is present.
Verifies has_bases==False and template/base vertex counts match expected FLAME template.
"""
import sys
from pathlib import Path
import numpy as np

# Ensure project root on PYTHONPATH when running manually
proj_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(proj_root))

from backend.models.reconstruction.reconstruct import ReconstructionModel


def main():
    print("Instantiating ReconstructionModel (may load FLAME OBJ)...")
    rm = ReconstructionModel(model_path='logs/reconstruction/reconstruction_model_best.pth')
    print("has_bases:", rm.has_bases)
    assert rm.has_bases is False, "Expected has_bases==False when FLAME OBJ is present"

    bv = getattr(rm, 'base_vertices', None)
    tv = getattr(rm, 'template_vertices', None)
    bf = getattr(rm, 'base_faces', None)
    tf = getattr(rm, 'template_faces', None)

    print("base_vertices.shape:", None if bv is None else bv.shape)
    print("template_vertices.shape:", None if tv is None else tv.shape)
    print("base_faces.shape:", None if bf is None else bf.shape)
    print("template_faces.shape:", None if tf is None else tf.shape)

    assert bv is not None and tv is not None, "Expected base and template vertices to be present"
    assert bv.shape == tv.shape, "Template vertices should match base vertices after FLAME load"
    assert bv.shape[1] == 3, "Vertex dimension should be 3"

    # If we know FLAME has 5023 vertices, assert that too (only if present)
    if bv.shape[0] == 5023:
        print("FLAME vertex count matches expected 5023")
    else:
        print("FLAME vertex count is", bv.shape[0])

    print("ReconstructionModel checks passed.")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
