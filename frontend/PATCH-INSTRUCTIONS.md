# Fix for Three-Mesh-BVH Compatibility Issue

This document provides a solution for the BatchedMesh error you're encountering when starting the frontend application.

## The Problem

The error occurs because the three-mesh-bvh library is trying to use BatchedMesh from Three.js, but your version of Three.js (0.157.0) doesn't include this feature, which was added in a later version.

## Solution

We've created several ways to patch this issue. Choose the method that works best for you:

### Option 1: JavaScript Patcher (Recommended)

1. Run the JavaScript-based patcher:
   ```bash
   node scripts/patch-libraries.js
   ```

2. Start the application:
   ```bash
   npm start
   ```

### Option 2: Batch Script (Windows)

1. Run the Windows batch script:
   ```cmd
   .\patch-three-mesh-bvh.bat
   ```

2. Start the application:
   ```cmd
   npm start
   ```

### Option 3: Shell Script (Linux/macOS)

1. Run the shell script:
   ```bash
   chmod +x ./patch-three-mesh-bvh.sh
   ./patch-three-mesh-bvh.sh
   ```

2. Start the application:
   ```bash
   npm start
   ```

## What the Patch Does

The patch modifies the `three-mesh-bvh/src/utils/ExtensionUtilities.js` file to:

1. Remove the direct import of BatchedMesh from Three.js
2. Create a mock/null value for BatchedMesh
3. Modify the acceleratedRaycast function to skip BatchedMesh checks

This allows the library to work with your current version of Three.js without requiring an upgrade.

## Note

If you update or reinstall node_modules, you'll need to apply the patch again.