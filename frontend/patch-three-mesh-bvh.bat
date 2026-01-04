@echo off
echo Patching three-mesh-bvh library...

set FILE_PATH=node_modules\three-mesh-bvh\src\utils\ExtensionUtilities.js
set BACKUP_PATH=node_modules\three-mesh-bvh\src\utils\ExtensionUtilities.js.bak

if not exist %FILE_PATH% (
    echo Error: Cannot find %FILE_PATH%
    exit /b 1
)

if not exist %BACKUP_PATH% (
    echo Creating backup at %BACKUP_PATH%...
    copy %FILE_PATH% %BACKUP_PATH%
)

echo Patching %FILE_PATH%...

(
echo import { Ray, Matrix4, Mesh, Vector3, Sphere, REVISION } from 'three';
echo import { convertRaycastIntersect } from '../utils/GeometryRayIntersectUtilities.js';
echo import { MeshBVH } from '../core/MeshBVH.js';
echo.
echo // Instead of importing BatchedMesh, we create a mock
echo const BatchedMesh = null; // Fix for three.js compatibility
echo const IS_REVISION_166 = parseInt^( REVISION ^) ^>= 166;
echo const ray = /* @__PURE__ */ new Ray^(^);
echo const direction = /* @__PURE__ */ new Vector3^(^);
echo const tmpInverseMatrix = /* @__PURE__ */ new Matrix4^(^);
echo const origMeshRaycastFunc = Mesh.prototype.raycast;
echo const origBatchedRaycastFunc = BatchedMesh !== null ? BatchedMesh.prototype.raycast : null;
echo const _worldScale = /* @__PURE__ */ new Vector3^(^);
echo const _mesh = /* @__PURE__ */ new Mesh^(^);
echo const _batchIntersects = [];
echo.
echo export function acceleratedRaycast^( raycaster, intersects ^) {
echo.
echo   if ^( BatchedMesh !== null ^&^& this instanceof BatchedMesh ^) {
echo.
echo     acceleratedBatchedMeshRaycast.call^( this, raycaster, intersects ^);
echo.
echo   } else {
echo.
echo     acceleratedMeshRaycast.call^( this, raycaster, intersects ^);
echo.
echo   }
echo.
echo }
echo.
echo function acceleratedMeshRaycast^( raycaster, intersects ^) {
echo.
echo   if ^( !this.geometry.boundsTree ^) {
echo.
echo     origMeshRaycastFunc.call^( this, raycaster, intersects ^);
echo     return;
echo.
echo   }
echo.
echo   if ^( this.material === undefined ^) return;
echo.
echo   tmpInverseMatrix.copy^( this.matrixWorld ^).invert^(^);
echo   ray.copy^( raycaster.ray ^).applyMatrix4^( tmpInverseMatrix ^);
echo.
echo   if ^( raycaster.firstHitOnly === true ^) {
echo.
echo     const res = this.geometry.boundsTree.raycastFirst^( ray, raycaster.raycastParams ^);
echo     if ^( res ^) {
echo.
echo       const intersection = convertRaycastIntersect^( res, this, raycaster ^);
echo       if ^( intersection ^) {
echo.
echo         intersects.push^( intersection ^);
echo.
echo       }
echo.
echo     }
echo.
echo   } else {
echo.
echo     const intersections = this.geometry.boundsTree.raycast^( ray, raycaster.raycastParams ^);
echo     if ^( intersections ^&^& intersections.length ^> 0 ^) {
echo.
echo       for ^( let i = 0, l = intersections.length; i ^< l; i++ ^) {
echo.
echo         const intersection = convertRaycastIntersect^( intersections[i], this, raycaster ^);
echo         if ^( intersection ^) {
echo.
echo           intersects.push^( intersection ^);
echo.
echo         }
echo.
echo       }
echo.
echo     }
echo.
echo   }
echo.
echo }
echo.
echo // Define a stub for this function to prevent errors
echo function acceleratedBatchedMeshRaycast^(raycaster, intersects^) {
echo   // BatchedMesh is not available, fallback to regular raycast
echo   acceleratedMeshRaycast.call^(this, raycaster, intersects^);
echo }
echo.
echo export function computeBoundsTree^( options ^) {
echo.
echo   this.boundsTree = new MeshBVH^( this, options ^);
echo   return this.boundsTree;
echo.
echo }
echo.
echo export function disposeBoundsTree^(^) {
echo.
echo   this.boundsTree = null;
echo.
echo }
) > %FILE_PATH%

echo Patch applied successfully!
echo.
echo Now you can run: npm start

exit /b 0