import pytest
from fastapi.testclient import TestClient
import io
from PIL import Image
from unittest.mock import patch

from backend.api.main import app
from backend.core.config import settings
from backend.api.models import schemas
from backend.core.dependencies import get_current_user

import pytest
try:
    client = TestClient(app)
except TypeError:
    pytest.skip("Incompatible TestClient in this environment", allow_module_level=True)


# This is a dummy image that can be used for file uploads in tests
@pytest.fixture(scope="session")
def dummy_image_bytes():
    """Creates a dummy PNG image in-memory."""
    img_byte_arr = io.BytesIO()
    image = Image.new("RGB", (10, 10), color="red")
    image.save(img_byte_arr, format="PNG")
    return img_byte_arr.getvalue()

# --- Mocks for API dependencies ---

def mock_user(role: str):
    return schemas.User(username=f"test_{role}", role=role)

def override_get_current_user_officer():
    return mock_user("officer")

def override_get_current_user_investigator():
    return mock_user("investigator")

def override_get_current_user_admin():
    return mock_user("admin")

# --- Test Cases ---

# Test /recognize endpoint
def test_recognize_unauthorized(dummy_image_bytes):
    files = {"file": ("test.png", dummy_image_bytes, "image/png")}
    data = {"case_id": "test-case"}
    response = client.post(f"{settings.API_V1_STR}/recognize", files=files, data=data)
    assert response.status_code == 401

def test_recognize_officer_forbidden(dummy_image_bytes):
    app.dependency_overrides[get_current_user] = override_get_current_user_officer
    files = {"file": ("test.png", dummy_image_bytes, "image/png")}
    data = {"case_id": "test-case"}
    response = client.post(f"{settings.API_V1_STR}/recognize", files=files, data=data)
    assert response.status_code == 403
    assert "Operation not permitted" in response.json()["detail"]
    app.dependency_overrides.clear()

@pytest.mark.parametrize("role_override", [override_get_current_user_investigator, override_get_current_user_admin])
@patch("backend.api.routes.recognize.write_event")
@patch("backend.api.routes.recognize.search_identities", return_value=[])
@patch("backend.api.routes.recognize.get_recognition_model")
def test_recognize_authorized_roles(mock_get_model, mock_search, mock_write_event, role_override, dummy_image_bytes):
    app.dependency_overrides[get_current_user] = role_override
    files = {"file": ("test.png", dummy_image_bytes, "image/png")}
    data = {"case_id": "test-case"}
    response = client.post(f"{settings.API_V1_STR}/recognize", files=files, data=data)
    # A 200 status code proves the RBAC check passed.
    assert response.status_code == 200
    app.dependency_overrides.clear()

# Test /reconstruct endpoint
def test_reconstruct_unauthorized(dummy_image_bytes):
    files = {"file": ("test.png", dummy_image_bytes, "image/png")}
    data = {"case_id": "test-case"}
    response = client.post(f"{settings.API_V1_STR}/reconstruct", files=files, data=data)
    assert response.status_code == 401

@pytest.mark.parametrize("role_override", [
    override_get_current_user_officer,
    override_get_current_user_investigator,
    override_get_current_user_admin
])
@patch("backend.api.routes.reconstruct.write_event")
@patch("backend.api.routes.reconstruct.get_reconstruction_model")
@patch("backend.api.routes.reconstruct.get_landmark_model")
def test_reconstruct_authorized_roles(mock_get_lm, mock_get_recon, mock_write_event, role_override, dummy_image_bytes):
    app.dependency_overrides[get_current_user] = role_override
    files = {"file": ("test.png", dummy_image_bytes, "image/png")}
    data = {"case_id": "test-case"}
    response = client.post(f"{settings.API_V1_STR}/reconstruct", files=files, data=data)
    # A 200 status code proves the RBAC check passed.
    assert response.status_code == 200
    app.dependency_overrides.clear()
