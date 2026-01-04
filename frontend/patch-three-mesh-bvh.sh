#!/bin/bash

echo "Patching three-mesh-bvh library..."

FILE_PATH="node_modules/three-mesh-bvh/src/utils/ExtensionUtilities.js"
BACKUP_PATH="node_modules/three-mesh-bvh/src/utils/ExtensionUtilities.js.bak"

if [ ! -f "$FILE_PATH" ]; then
    echo "Error: Cannot find $FILE_PATH"
    exit 1
fi

if [ ! -f "$BACKUP_PATH" ]; then
    echo "Creating backup at $BACKUP_PATH..."
    cp "$FILE_PATH" "$BACKUP_PATH"
fi

echo "Patching $FILE_PATH..."

cat > "$FILE_PATH" << 'EOL'
import { Ray, Matrix4, Mesh, Vector3, Sphere, REVISION } from 'three';
import { convertRaycastIntersect } from '../utils/GeometryRayIntersectUtilities.js';
import { MeshBVH } from '../core/MeshBVH.js';

// Instead of importing BatchedMesh, we create a mock
const BatchedMesh = null; // Fix for three.js compatibility
const IS_REVISION_166 = parseInt( REVISION ) >= 166;
const ray = /* @__PURE__ */ new Ray();
const direction = /* @__PURE__ */ new Vector3();
const tmpInverseMatrix = /* @__PURE__ */ new Matrix4();
const origMeshRaycastFunc = Mesh.prototype.raycast;
const origBatchedRaycastFunc = BatchedMesh !== null ? BatchedMesh.prototype.raycast : null;
const _worldScale = /* @__PURE__ */ new Vector3();
const _mesh = /* @__PURE__ */ new Mesh();
const _batchIntersects = [];

export function acceleratedRaycast( raycaster, intersects ) {

  if ( BatchedMesh !== null && this instanceof BatchedMesh ) {

    acceleratedBatchedMeshRaycast.call( this, raycaster, intersects );

  } else {

    acceleratedMeshRaycast.call( this, raycaster, intersects );

  }

}

function acceleratedMeshRaycast( raycaster, intersects ) {

  if ( !this.geometry.boundsTree ) {

    origMeshRaycastFunc.call( this, raycaster, intersects );
    return;

  }

  if ( this.material === undefined ) return;

  tmpInverseMatrix.copy( this.matrixWorld ).invert();
  ray.copy( raycaster.ray ).applyMatrix4( tmpInverseMatrix );

  if ( raycaster.firstHitOnly === true ) {

    const res = this.geometry.boundsTree.raycastFirst( ray, raycaster.raycastParams );
    if ( res ) {

      const intersection = convertRaycastIntersect( res, this, raycaster );
      if ( intersection ) {

        intersects.push( intersection );

      }

    }

  } else {

    const intersections = this.geometry.boundsTree.raycast( ray, raycaster.raycastParams );
    if ( intersections && intersections.length > 0 ) {

      for ( let i = 0, l = intersections.length; i < l; i++ ) {

        const intersection = convertRaycastIntersect( intersections[i], this, raycaster );
        if ( intersection ) {

          intersects.push( intersection );

        }

      }

    }

  }

}

// Define a stub for this function to prevent errors
function acceleratedBatchedMeshRaycast(raycaster, intersects) {
  // BatchedMesh is not available, fallback to regular raycast
  acceleratedMeshRaycast.call(this, raycaster, intersects);
}

export function computeBoundsTree( options ) {

  this.boundsTree = new MeshBVH( this, options );
  return this.boundsTree;

}

export function disposeBoundsTree() {

  this.boundsTree = null;

}
EOL

echo "Patch applied successfully!"
echo ""
echo "Now you can run: npm start"

exit 0