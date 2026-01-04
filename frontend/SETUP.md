# Frontend Setup and Troubleshooting

This guide will help you set up and run the frontend application for the Police 3D Face App.

## Prerequisites

- Node.js 18+ installed
- npm 9+ installed
- Backend API running on http://127.0.0.1:8000

## Installation

1. Install the dependencies:
   ```bash
   npm install --legacy-peer-deps
   ```

2. Apply the patch for three-mesh-bvh library:
   ```bash
   # Windows
   .\patch-three-mesh-bvh.bat
   
   # Linux/macOS
   chmod +x patch-three-mesh-bvh.sh
   ./patch-three-mesh-bvh.sh
   ```

3. Start the development server:
   ```bash
   npm start
   ```

## Troubleshooting

### BatchedMesh Error

If you see an error about `BatchedMesh` not being exported from 'three', run the patch script above which fixes this compatibility issue.

### ES Lint Warnings

To suppress ESLint warnings, you can add the following to your `.env` file:
```
ESLINT_NO_DEV_ERRORS=true
DISABLE_ESLINT_PLUGIN=true
```

### Authentication

The application is pre-configured with the following login credentials:
- Regular User:
  - Username: `officer`
  - Password: `password`

- Admin User:
  - Username: `admin`
  - Password: `admin_password`

## Features

- 3D Face Reconstruction visualization
- Face Recognition capabilities
- Evidence Management
- Reporting Dashboard
- Authentication & Access Control

## Architecture

The frontend is built with:
- React for UI components
- Three.js for 3D visualizations
- React Router for navigation
- Axios for API communication
- Socket.IO for real-time updates