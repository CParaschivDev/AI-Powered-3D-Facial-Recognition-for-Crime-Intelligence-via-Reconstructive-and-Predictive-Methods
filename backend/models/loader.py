import hashlib
import json
import os
from sqlalchemy import select
from backend.database.dependencies import get_db
from backend.database.models import ModelVersion

_active_cache = {}

def active_model_path(name: str, fallback_path: str = None) -> str:
    """
    Get the path to the active model of the given type.
    Caches results to avoid repeated database queries.
    
    Args:
        name: The model name (e.g., 'recognition', 'reconstruction')
        fallback_path: Optional fallback path if no active model is found
        
    Returns:
        Path to the active model
        
    Raises:
        RuntimeError: If no active model is found and no fallback is provided
    """
    if name in _active_cache:
        return _active_cache[name]
    
    # get_db returns a generator, get a session from it
    db = next(get_db())
    try:
        row = db.execute(
            select(ModelVersion).where(
                ModelVersion.name == name,
                ModelVersion.active == True
            )
        ).scalar_one_or_none()
        
        if row:
            _active_cache[name] = row.path
            return row.path
        
        if fallback_path:
            return fallback_path
            
        raise RuntimeError(f"No active model found for {name}")
    finally:
        db.close()

def calculate_sha256(file_path: str) -> str:
    """
    Calculate the SHA-256 hash of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        The SHA-256 hash as a hexadecimal string
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Justification: The loop is bounded by the file size and chunk size (4096 bytes).
        while True:
            chunk = f.read(4096)
            if not chunk:
                break
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

def register_model(name: str, version: int, path: str) -> bool:
    """
    Register a model in the model registry.
    
    Args:
        name: The model name (e.g., 'recognition', 'reconstruction')
        version: The model version
        path: Path to the model file or directory
        
    Returns:
        True if the model was registered successfully
    """
    if not os.path.exists(path):
        raise ValueError(f"Model path does not exist: {path}")
    
    # Calculate SHA-256 hash of the model file
    # If path is a directory, just hash the path string for now
    if os.path.isdir(path):
        # In a real system, you might want to hash all files in the directory
        # or use a manifest file
        sha256 = hashlib.sha256(path.encode()).hexdigest()
    else:
        sha256 = calculate_sha256(path)
    
    # get_db returns a generator, get a session from it
    db = next(get_db())
    try:
        # Check if the model already exists
        existing_model = db.execute(
            select(ModelVersion).where(
                ModelVersion.name == name,
                ModelVersion.version == version
            )
        ).scalar_one_or_none()
        
        if existing_model:
            # Update the existing model
            existing_model.path = path
            existing_model.sha256 = sha256
        else:
            # Create a new model
            new_model = ModelVersion(
                name=name,
                version=version,
                path=path,
                sha256=sha256
            )
            db.add(new_model)
            
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()