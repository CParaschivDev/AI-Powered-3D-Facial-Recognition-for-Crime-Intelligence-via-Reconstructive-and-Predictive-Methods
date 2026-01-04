import pytest
import numpy as np
import uuid
import struct

from backend.utils.watermark import embed_watermark, verify_watermark

@pytest.fixture
def sample_data():
    """Provides sample data for testing."""
    vertices = np.random.rand(500, 3).astype(np.float32) * 100
    case_id = str(uuid.uuid4())
    file_hash = "a" * 64
    return vertices, case_id, file_hash

def test_embed_and_verify_success(sample_data):
    """Tests that a watermark can be embedded and then successfully verified."""
    vertices, case_id, file_hash = sample_data
    
    watermarked_vertices = embed_watermark(vertices, case_id, file_hash)
    
    # The vertices should have changed slightly
    assert not np.array_equal(vertices, watermarked_vertices)
    
    is_valid, message = verify_watermark(watermarked_vertices, case_id, file_hash)
    
    assert is_valid is True
    assert "is valid" in message

def test_verify_fails_on_unwatermarked_data(sample_data):
    """Tests that verification fails on original, non-watermarked data."""
    vertices, case_id, file_hash = sample_data
    
    is_valid, message = verify_watermark(vertices, case_id, file_hash)
    
    assert is_valid is False
    assert "does not match" in message

def test_verify_fails_with_wrong_case_id(sample_data):
    """Tests that verification fails if the wrong case_id is provided."""
    vertices, case_id, file_hash = sample_data
    wrong_case_id = str(uuid.uuid4())
    
    watermarked_vertices = embed_watermark(vertices, case_id, file_hash)
    
    is_valid, message = verify_watermark(watermarked_vertices, wrong_case_id, file_hash)
    
    assert is_valid is False

def test_verify_fails_with_wrong_file_hash(sample_data):
    """Tests that verification fails if the wrong file_hash is provided."""
    vertices, case_id, file_hash = sample_data
    wrong_file_hash = "b" * 64
    
    watermarked_vertices = embed_watermark(vertices, case_id, file_hash)
    
    is_valid, message = verify_watermark(watermarked_vertices, case_id, wrong_file_hash)
    
    assert is_valid is False

def test_tampered_data_fails_verification(sample_data):
    """Tests that slightly modifying the watermarked data causes verification to fail."""
    vertices, case_id, file_hash = sample_data
    
    watermarked_vertices = embed_watermark(vertices, case_id, file_hash)
    
    # Tamper with one of the coordinates that holds a watermark bit by flipping its LSB
    coord_bits = struct.unpack('<I', struct.pack('<f', watermarked_vertices[0, 0]))[0]
    tampered_coord_bits = coord_bits ^ 1 # Flip the LSB
    watermarked_vertices[0, 0] = struct.unpack('<f', struct.pack('<I', tampered_coord_bits))[0]
    
    is_valid, message = verify_watermark(watermarked_vertices, case_id, file_hash)
    
    assert is_valid is False

def test_mesh_too_small_for_embedding():
    """Tests that an error is raised if the mesh is too small to hold the watermark."""
    # 256 bits watermark needs 256 floats. 256/3 = 85.3 vertices. So 86 vertices needed.
    # Let's use 85 vertices (255 floats).
    small_vertices = np.random.rand(85, 3).astype(np.float32)
    case_id = "test-case"
    file_hash = "test-hash"
    
    with pytest.raises(ValueError, match="Not enough vertex coordinates"):
        embed_watermark(small_vertices, case_id, file_hash)

def test_mesh_too_small_for_verification():
    """Tests that verification fails gracefully if the mesh is too small."""
    small_vertices = np.random.rand(85, 3).astype(np.float32)
    case_id = "test-case"
    file_hash = "test-hash"
    
    is_valid, message = verify_watermark(small_vertices, case_id, file_hash)
    assert is_valid is False
    assert "Mesh is too small" in message
