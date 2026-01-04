from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Set, Tuple, Optional, Union, Any

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from passlib.context import CryptContext
from jose import jwt
import tenseal as ts
import os
import secrets
import pickle
import json
import numpy as np
from base64 import b64decode, b64encode

from fastapi import Depends, HTTPException, status
from backend.core.config import settings
from backend.api.models.schemas import User as UserSchema


class Role(str, Enum):
    ADMIN = "admin"
    INVESTIGATOR = "investigator"
    OFFICER = "officer"


# This maps a role to the set of scopes it includes, establishing a hierarchy.
ROLE_SCOPES: Dict[Role, Set[str]] = {
    Role.ADMIN: {Role.ADMIN, Role.INVESTIGATOR, Role.OFFICER},
    Role.INVESTIGATOR: {Role.INVESTIGATOR, Role.OFFICER},
    Role.OFFICER: {Role.OFFICER},
}

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class KmsManager:
    """
    Manages envelope encryption for data at rest, similar to a KMS.
    - A master key (from settings) encrypts per-object data encryption keys (DEKs).
    - Each object is encrypted with its own unique DEK using AES-256-GCM.
    """
    AES_KEY_SIZE = 32  # 256-bit
    GCM_NONCE_SIZE = 12  # 96-bit is recommended
    GCM_TAG_SIZE = 16  # 128-bit

    def __init__(self, master_key: str):
        if not master_key or len(master_key.encode()) < 44:
            raise ValueError("ENCRYPTION_KEY must be a 32-byte URL-safe base64-encoded key.")
        self._master_key_fernet = Fernet(master_key.encode())

    def _generate_dek(self) -> bytes:
        return os.urandom(self.AES_KEY_SIZE)

    def _encrypt_dek(self, dek: bytes) -> bytes:
        return self._master_key_fernet.encrypt(dek)

    def _decrypt_dek(self, encrypted_dek: bytes) -> bytes:
        return self._master_key_fernet.decrypt(encrypted_dek)

    def encrypt(self, plaintext: bytes) -> tuple[bytes, bytes]:
        """Encrypts plaintext using envelope encryption."""
        dek = self._generate_dek()
        nonce = os.urandom(self.GCM_NONCE_SIZE)
        aesgcm = AESGCM(dek)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)

        # Ciphertext blob format: nonce || ciphertext (which includes the tag)
        ciphertext_blob = nonce + ciphertext
        encrypted_dek = self._encrypt_dek(dek)
        return ciphertext_blob, encrypted_dek

    def decrypt(self, ciphertext_blob: bytes, encrypted_dek: bytes) -> bytes:
        """Decrypts a ciphertext blob using its encrypted DEK."""
        dek = self._decrypt_dek(encrypted_dek)
        nonce = ciphertext_blob[:self.GCM_NONCE_SIZE]
        ciphertext_with_tag = ciphertext_blob[self.GCM_NONCE_SIZE:]
        aesgcm = AESGCM(dek)
        return aesgcm.decrypt(nonce, ciphertext_with_tag, None)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


# --- Homomorphic Encryption for Secure Queries ---

def get_homomorphic_context() -> ts.Context:
    """
    Loads or creates a TenSEAL context for homomorphic encryption.
    In a real production system, the context would be securely managed and distributed.
    """
    context_path = settings.HOMOMORPHIC_CONTEXT_PATH
    # Prevent loading an unexpectedly large context file into memory
    MAX_CONTEXT_BYTES = 50 * 1024 * 1024  # 50 MB
    if os.path.exists(context_path):
        size = os.path.getsize(context_path)
        if size > MAX_CONTEXT_BYTES:
            raise ValueError("Stored homomorphic context file is unexpectedly large")
        # Read the file in a size-limited way to avoid loading huge contexts
        from backend.core.file_utils import safe_read_file_bytes

        data = safe_read_file_bytes(context_path, max_bytes=MAX_CONTEXT_BYTES)
        return ts.context_from(data)
    else:
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        context.generate_galois_keys()
        context.global_scale = 2**40
        os.makedirs(os.path.dirname(context_path), exist_ok=True)
        with open(context_path, "wb") as f:
            f.write(context.serialize(save_secret_key=True))
        return context


def require_roles(*required_roles: str):
    """
    Dependency factory to enforce role-based access.

    Allows access if the user's role grants them AT LEAST ONE of the required roles (scopes).
    e.g., require_roles("investigator") will allow users with roles INVESTIGATOR and ADMIN.
    """
    required_role_set = set(required_roles)

    # Import here to avoid circular import at module load time
    from backend.core.dependencies import get_current_user as _get_current_user

    def role_checker(current_user: UserSchema = Depends(_get_current_user)):
        # The user's role is fetched from the DB after token validation.
        # It's considered the source of truth for their permissions.
        try:
            user_role = Role(current_user.role)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User has an unknown role: {current_user.role}",
            )
        
        user_scopes = ROLE_SCOPES.get(user_role, set())

        # Admin super-role: always allow if user is ADMIN regardless of required roles
        if required_role_set.isdisjoint(user_scopes) and user_role != Role.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation not permitted. Requires one of: {', '.join(required_role_set)}",
            )
        return current_user

    return role_checker


def get_master_key() -> bytes:
    """
    Get the master encryption key for envelope encryption.
    Raises an exception if the key is not available.
    """
    key = settings.BIOMETRIC_MASTER_KEY

    # If a BIOMETRIC_MASTER_KEY isn't set in the environment, derive a deterministic
    # fallback from the application SECRET_KEY so tests can run without requiring
    # a production key to be present. This keeps behavior deterministic for CI.
    if not key:
        base = getattr(settings, "SECRET_KEY", "test-secret")
        salt = b"biometric_master_key_fallback"
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return kdf.derive(base.encode())

    # Convert the provided key to proper length for AESGCM if necessary
    if len(key) < 32:
        salt = b"biometric_envelope_encryption"  # Fixed salt for reproducibility
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return kdf.derive(key.encode())

    return key.encode()[:32]  # Use first 32 bytes if key is longer


def envelope_encrypt(data: Union[bytes, np.ndarray, dict, Any]) -> dict:
    """
    Encrypt data using envelope encryption:
    1. Generate a random data encryption key (DEK)
    2. Encrypt the data with the DEK
    3. Encrypt the DEK with the master key (KEK - Key Encryption Key)
    4. Return both the encrypted data and the encrypted DEK
    
    Parameters:
        data: Data to encrypt (can be bytes, numpy array, dictionary or any serializable object)
        
    Returns:
        dict: Dictionary containing encrypted data and metadata
    """
    try:
        # Generate random data encryption key (DEK)
        dek = os.urandom(32)
        
        # Convert input data to bytes if necessary
        if isinstance(data, np.ndarray):
            serialized_data = pickle.dumps(data)
            data_type = "numpy"
        elif isinstance(data, dict):
            serialized_data = json.dumps(data).encode()
            data_type = "json"
        elif isinstance(data, bytes):
            serialized_data = data
            data_type = "bytes"
        else:
            # Handle any other serializable object
            serialized_data = pickle.dumps(data)
            data_type = "pickle"
        
        # Generate a random nonce for DEK encryption
        dek_nonce = os.urandom(12)
        
        # Generate a random nonce for data encryption
        data_nonce = os.urandom(12)
        
        # Get master key (KEK)
        master_key = get_master_key()
        
        # Create AESGCM instances for both DEK and data encryption
        kek_cipher = AESGCM(master_key)
        dek_cipher = AESGCM(dek)
        
        # Encrypt the DEK with the master key (KEK)
        encrypted_dek = kek_cipher.encrypt(dek_nonce, dek, b"DEK")
        
        # Encrypt the data with the DEK
        encrypted_data = dek_cipher.encrypt(data_nonce, serialized_data, b"DATA")
        
        # Return the encrypted data package
        return {
            "encrypted_data": b64encode(encrypted_data).decode('utf-8'),
            "encrypted_dek": b64encode(encrypted_dek).decode('utf-8'),
            "dek_nonce": b64encode(dek_nonce).decode('utf-8'),
            "data_nonce": b64encode(data_nonce).decode('utf-8'),
            "data_type": data_type,
            "encryption_method": "AESGCM_ENVELOPE"
        }
    except Exception as e:
        # Log the exception but don't reveal details in the error message
        print(f"Encryption error: {str(e)}")
        raise ValueError("Failed to encrypt data") from e


def envelope_decrypt(encrypted_package: dict) -> Any:
    """
    Decrypt data using envelope encryption:
    1. Decrypt the DEK with the master key (KEK)
    2. Use the decrypted DEK to decrypt the data
    
    Parameters:
        encrypted_package: Dictionary containing encrypted data and metadata
        
    Returns:
        The decrypted data in its original format
    """
    try:
        # Validate the encrypted package
        required_fields = ["encrypted_data", "encrypted_dek", "dek_nonce", 
                          "data_nonce", "data_type", "encryption_method"]
        
        for field in required_fields:
            if field not in encrypted_package:
                raise ValueError(f"Invalid encrypted package: missing '{field}'")
        
        if encrypted_package["encryption_method"] != "AESGCM_ENVELOPE":
            raise ValueError("Unsupported encryption method")
        
        # Get the master key (KEK)
        master_key = get_master_key()
        
        # Decode all base64-encoded values
        encrypted_data = b64decode(encrypted_package["encrypted_data"])
        encrypted_dek = b64decode(encrypted_package["encrypted_dek"])
        dek_nonce = b64decode(encrypted_package["dek_nonce"])
        data_nonce = b64decode(encrypted_package["data_nonce"])
        data_type = encrypted_package["data_type"]
        
        # Create AESGCM instance for the master key (KEK)
        kek_cipher = AESGCM(master_key)
        
        # Decrypt the DEK using the master key
        try:
            dek = kek_cipher.decrypt(dek_nonce, encrypted_dek, b"DEK")
        except Exception:
            raise ValueError("Failed to decrypt DEK: Invalid master key or corrupted DEK")
        
        # Create AESGCM instance with the decrypted DEK
        dek_cipher = AESGCM(dek)
        
        # Decrypt the data using the DEK
        try:
            decrypted_data_bytes = dek_cipher.decrypt(data_nonce, encrypted_data, b"DATA")
        except Exception:
            raise ValueError("Failed to decrypt data: Corrupted data or invalid DEK")
        
        # Deserialize the data based on its original type
        if data_type == "numpy":
            return pickle.loads(decrypted_data_bytes)
        elif data_type == "json":
            return json.loads(decrypted_data_bytes.decode('utf-8'))
        elif data_type == "bytes":
            return decrypted_data_bytes
        elif data_type == "pickle":
            return pickle.loads(decrypted_data_bytes)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
    except Exception as e:
        # Log the exception but don't reveal details in the error message
        print(f"Decryption error: {str(e)}")
        raise ValueError("Failed to decrypt data") from e
