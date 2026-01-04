/**
 * This file loads all patches to fix third-party library compatibility issues
 */

// Import our patches - this will load the patches into the bundle
import './patches/three-mesh-bvh-patch';

console.log('Patches for third-party libraries have been applied.');

/**
 * This approach uses a webpack setup that will override the problematic module
 * with our patched version. Make sure your webpack.config.js includes the following:
 * 
 * resolve: {
 *   alias: {
 *     'three-mesh-bvh/src/utils/ExtensionUtilities.js': 
 *       path.resolve(__dirname, 'src/patches/three-mesh-bvh-patch.js'),
 *   }
 * }
 */