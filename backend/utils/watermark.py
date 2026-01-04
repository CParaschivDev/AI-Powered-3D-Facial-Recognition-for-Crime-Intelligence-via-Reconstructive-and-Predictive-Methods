import hashlib
import struct
import numpy as np
from typing import Tuple

WATERMARK_BITS = 256  # Using SHA-256, so 256 bits

def _float_to_bits(f: np.float32) -> int:
    """Converts a float32 to its 32-bit integer representation."""
    return struct.unpack('<I', struct.pack('<f', f))[0]

def _bits_to_float(b: int) -> np.float32:
    """Converts a 32-bit integer representation back to a float32."""
    return struct.unpack('<f', struct.pack('<I', b))[0]

def _generate_watermark_bits(case_id: str, file_hash: str) -> str:
    """Generates a binary string representation of the watermark hash."""
    h = hashlib.sha256(f"{case_id}:{file_hash}".encode()).digest()
    return ''.join(format(byte, '08b') for byte in h)

def embed_watermark(vertices: np.ndarray, case_id: str, file_hash: str) -> np.ndarray:
    """
    Embeds a watermark into the least significant bits of vertex coordinates.

    Args:
        vertices: A NumPy array of shape (N, 3) for vertex positions, dtype=float32.
        case_id: The case ID.
        file_hash: The hash of the original evidence file.

    Returns:
        A new NumPy array with the watermark embedded.
    """
    watermark_bits = _generate_watermark_bits(case_id, file_hash)
    if vertices.size < len(watermark_bits):
        raise ValueError(f"Not enough vertex coordinates ({vertices.size}) to embed a {len(watermark_bits)}-bit watermark.")

    watermarked_vertices = vertices.flatten().astype(np.float32)
    
    for i, bit in enumerate(watermark_bits):
        coord = watermarked_vertices[i]
        coord_bits = _float_to_bits(coord)
        
        if bit == '1':
            coord_bits |= 1  # Set LSB to 1
        else:
            coord_bits &= ~1 # Set LSB to 0
            
        watermarked_vertices[i] = _bits_to_float(coord_bits)
        
    return watermarked_vertices.reshape(vertices.shape)

def verify_watermark(vertices: np.ndarray, case_id: str, file_hash: str) -> Tuple[bool, str]:
    """
    Verifies a watermark in the vertex coordinates.

    Args:
        vertices: A NumPy array of shape (N, 3) for vertex positions, dtype=float32.
        case_id: The case ID to check against.
        file_hash: The original file hash to check against.

    Returns:
        A tuple (is_valid, message).
    """
    expected_watermark_bits = _generate_watermark_bits(case_id, file_hash)
    
    if vertices.size < len(expected_watermark_bits):
        return False, f"Mesh is too small ({vertices.size} coordinates) to contain a {len(expected_watermark_bits)}-bit watermark."

    flat_vertices = vertices.flatten().astype(np.float32)
    extracted_bits = []
    
    for i in range(len(expected_watermark_bits)):
        coord_bits = _float_to_bits(flat_vertices[i])
        extracted_bits.append('1' if (coord_bits & 1) else '0')
        
    extracted_watermark = "".join(extracted_bits)
    
    if extracted_watermark == expected_watermark_bits:
        return True, "Watermark is valid and matches the provided case and file hash."
    else:
        return False, "Watermark is invalid or does not match."
