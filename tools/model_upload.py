#!/usr/bin/env python
"""
Model Registry Upload Tool

This script uploads a model file to the model registry and registers it in the database.
It calculates the SHA-256 hash of the file for integrity verification.

Usage:
    python model_upload.py --name recognition --version 1 --path /path/to/model.pth [--activate]

Parameters:
    --name: Model name (e.g., 'recognition', 'reconstruction')
    --version: Model version number
    --path: Path to the model file or directory
    --activate: If set, activates the model after registration
"""

import argparse
import hashlib
import os
import sys
import logging

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.database.dependencies import SessionLocal, get_db
from backend.database.models import ModelVersion
from sqlalchemy import update

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_sha256(file_path: str) -> str:
    """Calculates the SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Justification: The loop is bounded by the file size and chunk size (4096 bytes).
        while True:
            chunk = f.read(4096)
            if not chunk:
                break
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

def register_model(name: str, version: int, path: str):
    """Registers a new model version in the database."""
    if not os.path.exists(path):
        print(f"Error: Path does not exist: {path}")
        return

    print(f"Calculating SHA256 for {path}...")
    if os.path.isfile(path):
        sha256 = calculate_sha256(path)
    elif os.path.isdir(path):
        # For directories, we can't easily hash. Let's just use a placeholder.
        # A better approach would be to hash a manifest file or tarball.
        # For this simple implementation, a placeholder is fine.
        print("Warning: Path is a directory. Using placeholder SHA256. For production, use a tarball.")
        sha256 = "directory_placeholder_" + hashlib.sha256(path.encode()).hexdigest()
    else:
        print(f"Error: Path is not a file or directory: {path}")
        return
    
    print(f"SHA256: {sha256}")

    db = SessionLocal()
    try:
        # Check if this version already exists
        existing = db.query(ModelVersion).filter(
            ModelVersion.name == name,
            ModelVersion.version == version
        ).first()
        if existing:
            print(f"Error: Model '{name}' version {version} already exists.")
            return

        new_model = ModelVersion(
            name=name,
            version=version,
            path=path,
            sha256=sha256,
            active=False
        )
        db.add(new_model)
        db.commit()
        print(f"Successfully registered model '{name}' version {version} from path '{path}'.")
        print("Run `POST /api/v1/models/activate` to activate this version.")
    finally:
        db.close()

def main():
    parser = argparse.ArgumentParser(description="Register a new model version in the model registry.")
    parser.add_argument("--name", required=True, help="The name of the model (e.g., 'recognition', 'reconstruction').")
    parser.add_argument("--version", required=True, type=int, help="The integer version number for this model.")
    parser.add_argument("--path", required=True, help="The path to the model file or directory.")
    args = parser.parse_args()

    register_model(args.name, args.version, args.path)

if __name__ == "__main__":
    main()