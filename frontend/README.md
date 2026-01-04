# Police 3D Face App - Frontend

This is the frontend for the Police 3D Face Reconstruction application. It provides a user interface for interacting with the backend API and visualizing 3D face models.

## Getting Started

### Prerequisites

- Node.js 18+ installed
- npm 9+ installed
- Backend API running on http://127.0.0.1:8000

### Installation

1. Install the dependencies:
   ```bash
   npm install
   ```

2. Install additional required packages:
   ```bash
   # Windows
   .\install-packages.bat
   
   # Linux/macOS
   chmod +x install-packages.sh
   ./install-packages.sh
   ```

### Running the Application

Start the development server:
```bash
npm start
```

The application will be available at http://localhost:3000

### Login Credentials

The application is pre-configured with the following login credentials:

- Regular User:
  - Username: `officer`
  - Password: `password`

- Admin User:
  - Username: `admin`
  - Password: `admin_password`

## Features

- 3D Face Reconstruction
- Face Recognition
- Evidence Management
- Report Generation
- Predictive Analytics
- Case Management
- Authentication & Authorization
- Audit Trail

## Troubleshooting

If you encounter any issues:

1. Make sure the backend API is running
2. Check the console for error messages
3. Ensure you have all dependencies installed
4. Verify that the .env file has the correct API URL