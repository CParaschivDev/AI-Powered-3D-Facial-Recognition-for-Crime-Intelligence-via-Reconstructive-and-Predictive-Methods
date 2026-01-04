import { Ray, Matrix4, Mesh, Vector3, Sphere, REVISION } from 'three';
import { convertRaycastIntersect } from 'three-mesh-bvh/src/utils/GeometryRayIntersectUtilities.js';
import { MeshBVH } from 'three-mesh-bvh/src/core/MeshBVH.js';

// Instead of importing BatchedMesh, we'll create a mock
const IS_REVISION_166 = parseInt(REVISION) >= 166;
const ray = /* @__PURE__ */ new Ray();
const direction = /* @__PURE__ */ new Vector3();
const tmpInverseMatrix = /* @__PURE__ */ new Matrix4();
const origMeshRaycastFunc = Mesh.prototype.raycast;
const _worldScale = /* @__PURE__ */ new Vector3();
const _mesh = /* @__PURE__ */ new Mesh();
const _batchIntersects = [];

export function acceleratedRaycast(raycaster, intersects) {
  acceleratedMeshRaycast.call(this, raycaster, intersects);
}

function acceleratedMeshRaycast(raycaster, intersects) {
  if (!this.geometry.boundsTree) {
    origMeshRaycastFunc.call(this, raycaster, intersects);
    return;
  }

  if (this.material === undefined) return;

  tmpInverseMatrix.copy(this.matrixWorld).invert();
  ray.copy(raycaster.ray).applyMatrix4(tmpInverseMatrix);

  if (raycaster.firstHitOnly === true) {
    const res = this.geometry.boundsTree.raycastFirst(ray, raycaster.raycastParams);
    if (res) {
      const intersection = convertRaycastIntersect(res, this, raycaster);
      if (intersection) {
        intersects.push(intersection);
      }
    }
  } else {
    const intersections = this.geometry.boundsTree.raycast(ray, raycaster.raycastParams);
    if (intersections && intersections.length > 0) {
      for (let i = 0, l = intersections.length; i < l; i++) {
        const intersection = convertRaycastIntersect(intersections[i], this, raycaster);
        if (intersection) {
          intersects.push(intersection);
        }
      }
    }
  }
}

export function computeBoundsTree(options) {
  this.boundsTree = new MeshBVH(this, options);
  return this.boundsTree;
}

export function disposeBoundsTree() {
  this.boundsTree = null;
}