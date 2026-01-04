const fs = require('fs');
const path = require('path');

console.log('Patching libraries for compatibility...');

// Function to patch the three-mesh-bvh library
function patchThreeMeshBVH() {
  const filePath = path.join(
    __dirname, 
    '../node_modules/three-mesh-bvh/src/utils/ExtensionUtilities.js'
  );
  
  const backupPath = `${filePath}.bak`;
  
  if (!fs.existsSync(filePath)) {
    console.error(`Error: File not found at ${filePath}`);
    return false;
  }
  
  // Create backup if it doesn't exist
  if (!fs.existsSync(backupPath)) {
    console.log(`Creating backup at ${backupPath}`);
    fs.copyFileSync(filePath, backupPath);
  }
  
  console.log(`Patching file: ${filePath}`);
  
  const patchedContent = `import { Ray, Matrix4, Mesh, Vector3, Sphere, REVISION } from 'three';
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

  // Skip BatchedMesh check since we don't have it
  acceleratedMeshRaycast.call( this, raycaster, intersects );

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

export function computeBoundsTree( options ) {

  this.boundsTree = new MeshBVH( this, options );
  return this.boundsTree;

}

export function disposeBoundsTree() {

  this.boundsTree = null;

}`;
  
  // Write the patched content
  fs.writeFileSync(filePath, patchedContent);
  console.log('Successfully patched three-mesh-bvh!');
  return true;
}

// Main execution
try {
  let success = patchThreeMeshBVH();
  
  if (success) {
    console.log('\nAll patches applied successfully!');
    console.log('You can now run the application with: npm start');
  } else {
    console.error('\nSome patches failed to apply.');
    console.error('Please check the errors above and try again.');
    process.exit(1);
  }
} catch (error) {
  console.error('Error while applying patches:', error);
  process.exit(1);
}