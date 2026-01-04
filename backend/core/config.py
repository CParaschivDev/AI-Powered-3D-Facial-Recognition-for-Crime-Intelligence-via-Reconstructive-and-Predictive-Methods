from functools import lru_cache
import os
import json
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from typing import Optional, List, Any
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    # API settings
    PROJECT_NAME: str = "An AI-Powered 3D Facial Recognition Framework for Crime Intelligence Using Reconstructive and Predictive Techniques"
    API_V1_STR: str = "/api/v1"

    # Database settings
    DATABASE_URL: str

    # Security settings
    SECRET_KEY: str
    ENCRYPTION_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Dummy user for demo auth
    DUMMY_USER_USERNAME: str = "officer"
    DUMMY_USER_PASSWORD: str = "password"
    DUMMY_ADMIN_USERNAME: str = "admin"
    DUMMY_ADMIN_PASSWORD: str = "admin_password"

    # Orchestration settings
    OLLAMA_BASE_URL: str = "http://ollama:11434"

    # Federated Learning Settings
    FEDERATED_AGGREGATOR_URL: str = "http://backend:8000/federated/aggregate" # Internal service URL

    # Homomorphic Encryption Settings
    HOMOMORPHIC_CONTEXT_PATH: str = "/app/secrets/he_context.bin"

    # Email Notification Settings
    SMTP_HOST: Optional[str] = None
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    SMTP_SENDER_NAME: str = "Police Alert System"
    INVESTIGATOR_EMAILS: str = "investigators@example.com"

    # Gen5: Knowledge Graph Settings
    NEO4J_URI: Optional[str] = "bolt://neo4j:7687"
    NEO4J_USER: Optional[str] = "neo4j"
    NEO4J_PASSWORD: Optional[str] = "password"

    # Biometric Encryption Settings
    BIOMETRIC_MASTER_KEY: Optional[str] = None  # Master key for envelope encryption

    # Data locations (outside repo)
    DATA_ROOT: str = ""  # e.g., C:/Users/Paras/Desktop/Police App/police-3d-face-app/Data
    WATCHLIST_DIRS: List[str] = ["actor_faces", "actress_faces"]
    MINE_DIR: str = "mine"
    AFLW2K3D_DIR: str = "AFLW2000"
    UK_POLICE_DIR: str = "UK DATA CRIME 2022 - 2025"
    FACES_EMBEDDINGS_DB_PATH: str = "backend/database/faces_embeddings.db"
    
    @field_validator("WATCHLIST_DIRS", mode="before")
    @classmethod
    def parse_watchlist_dirs(cls, v: Any) -> List[str]:
        """Parse WATCHLIST_DIRS from a JSON string if provided as a string."""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                # Handle malformed JSON by returning a list with the string itself
                # This is a fallback for values like "actor_faces,actress_faces" (without brackets)
                return [dir.strip() for dir in v.strip("[]").split(",")]
        return v
 
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()
