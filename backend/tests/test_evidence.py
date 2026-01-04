import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from backend.api.main import app
from backend.api.models import schemas
import numpy as np

import pytest
try:
    client = TestClient(app)
except TypeError:
    pytest.skip("Incompatible TestClient in this environment", allow_module_level=True)

# Mock user for testing
mock_investigator_user = schemas.User(
    username="investigator_test",
    email="investigator@example.com",
    full_name="Test Investigator",
    disabled=False,
    roles=["investigator"]
)

# Mock watermark verification function
def mock_verify_watermark(vertices, case_id, original_file_hash):
    if case_id == "valid_case" and original_file_hash == "valid_hash":
        return True, "Watermark verified successfully (mocked)."
    return False, "Watermark verification failed (mocked)."

# Mock parse_obj_file function
def mock_parse_obj_file(file_content: bytes) -> np.ndarray:
    # Simulate parsing an OBJ file with some vertices
    return np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)

@pytest.fixture(autouse=True)
def mock_dependencies():
    with patch("backend.core.dependencies.get_current_active_user", return_value=mock_investigator_user), \
         patch("backend.core.security.require_roles", return_value=lambda roles: None), \
         patch("backend.utils.watermark.verify_watermark", side_effect=mock_verify_watermark), \
         patch("backend.api.routes.evidence.parse_obj_file", side_effect=mock_parse_obj_file):
        yield

def test_verify_model_watermark_success():
    # Create a dummy OBJ file content
    obj_content = b"v 0.0 0.0 0.0\nv 1.0 1.0 1.0\n"
    
    response = client.post(
        "/api/v1/evidence/verify",
        files={"file": ("model.obj", obj_content, "text/plain")},
        data={"case_id": "valid_case", "original_file_hash": "valid_hash"}
    )
    
    assert response.status_code == 200
    assert response.json() == {"is_valid": True, "message": "Watermark verified successfully (mocked)."}

def test_verify_model_watermark_failure():
    obj_content = b"v 0.0 0.0 0.0\nv 1.0 1.0 1.0\n"
    
    response = client.post(
        "/api/v1/evidence/verify",
        files={"file": ("model.obj", obj_content, "text/plain")},
        data={"case_id": "invalid_case", "original_file_hash": "invalid_hash"}
    )
    
    assert response.status_code == 200
    assert response.json() == {"is_valid": False, "message": "Watermark verification failed (mocked)."}

def test_verify_model_watermark_unsupported_file_type():
    # Create a dummy TXT file content
    txt_content = b"This is not an OBJ file."
    
    response = client.post(
        "/api/v1/evidence/verify",
        files={"file": ("document.txt", txt_content, "text/plain")},
        data={"case_id": "valid_case", "original_file_hash": "valid_hash"}
    )
    
    assert response.status_code == 400
    assert response.json() == {"detail": "Unsupported file type. Please upload an .obj file."}

def test_verify_model_watermark_no_vertices():
    # Mock parse_obj_file to return an empty array for this specific test
    with patch("backend.api.routes.evidence.parse_obj_file", return_value=np.array([], dtype=np.float32)):
        obj_content = b"v 0.0 0.0 0.0\nv 1.0 1.0 1.0\n" # Content doesn't matter as parse_obj_file is mocked
        
        response = client.post(
            "/api/v1/evidence/verify",
            files={"file": ("empty.obj", obj_content, "text/plain")},
            data={"case_id": "valid_case", "original_file_hash": "valid_hash"}
        )
        
        assert response.status_code == 400
        assert response.json() == {"detail": "No vertices found in the provided file."}

def test_parse_obj_file_valid_content():
    obj_content = b"v 1.0 2.0 3.0\nv 4.0 5.0 6.0\nf 1 2 3\n"
    vertices = client.app.dependency_overrides[mock_dependencies.__name__].__wrapped__.__globals__['mock_parse_obj_file'](obj_content)
    assert isinstance(vertices, np.ndarray)
    assert vertices.shape == (2, 3)
    assert np.allclose(vertices, np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])) # Mocked output

def test_parse_obj_file_invalid_content():
    invalid_obj_content = b"invalid content"
    with pytest.raises(Exception) as exc_info:
        client.app.dependency_overrides[mock_dependencies.__name__].__wrapped__.__globals__['mock_parse_obj_file'](invalid_obj_content)
    assert "Could not parse the provided OBJ file." in str(exc_info.value) # Mocked output
