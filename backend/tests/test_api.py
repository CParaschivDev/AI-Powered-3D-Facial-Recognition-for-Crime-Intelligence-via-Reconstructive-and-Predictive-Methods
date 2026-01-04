import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import os
import numpy as np

from backend.api.main import app
import pytest
try:
    client = TestClient(app)
except TypeError:
    pytest.skip("Incompatible TestClient in this environment", allow_module_level=True)

@pytest.fixture
def auth_token():
    """Fixture to get a valid auth token."""
    # This uses the dummy user credentials from config
    response = client.post("/api/v1/auth/token", data={"username": "officer", "password": "password"})
    assert response.status_code == 200
    return response.json()["access_token"]


def test_reconstruct_endpoint(auth_token):
    """
    Test the /reconstruct endpoint with a mock image and mocked models.
    """
    headers = {"Authorization": f"Bearer {auth_token}"}
    # A dummy 1x1 pixel black image
    dummy_image_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc`\x00\x00\x00\x02\x00\x01\xe2!\xbc\x33\x00\x00\x00\x00IEND\xaeB`\x82'

    with patch('backend.api.routes.reconstruct.write_event'), \
         patch('backend.api.routes.reconstruct.get_landmark_model') as mock_get_lm, \
         patch('backend.api.routes.reconstruct.get_reconstruction_model') as mock_get_recon, \
         patch('backend.api.routes.reconstruct.get_db'):
        
        # Mock the return values of the model predictions
        mock_get_lm.return_value.predict.return_value = np.random.rand(10, 2)
        mock_get_recon.return_value.reconstruct.return_value = (np.random.rand(5, 3), np.random.randint(0, 4, (3, 3)))

        response = client.post(
            "/api/v1/reconstruct",
            data={"case_id": "test-case"},
            files={"file": ("test.png", dummy_image_bytes, "image/png")},
            headers=headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "vertices" in data
        assert "faces" in data
        assert len(data["vertices"]) == 5

def test_recognize_endpoint(auth_token, tmp_path):
    """
    Test the /recognize endpoint, including saliency map generation.
    """
    headers = {"Authorization": f"Bearer {auth_token}"}
    dummy_image_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc`\x00\x00\x00\x02\x00\x01\xe2!\xbc\x33\x00\x00\x00\x00IEND\xaeB`\x82'
    case_id = "test-case-saliency"

    # Use a temporary directory for evidence storage
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()

    with patch('backend.api.routes.recognize.write_event'), \
         patch('backend.api.routes.recognize.get_db'), \
         patch('backend.api.routes.recognize.os.path.join', side_effect=lambda *args: str(tmp_path.joinpath(*args))), \
         patch('backend.api.routes.recognize.get_recognition_model') as mock_get_recog, \
         patch('backend.api.routes.recognize.search_identities') as mock_search:
        
        mock_get_recog.return_value.extract_fused_embedding.return_value = np.random.rand(512)
        mock_get_recog.return_value.generate_saliency_map.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_search.return_value = [{"identity_id": "person_001", "confidence": 0.95}]

        response = client.post(
            "/api/v1/recognize",
            files={"file": ("test.png", dummy_image_bytes, "image/png")},
            data={"case_id": case_id},
            headers=headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "matches" in data
        assert len(data["matches"]) == 1
        assert data["matches"][0]["identity_id"] == "person_001"

        # Test saliency generation
        assert "saliency_url" in data["matches"][0]
        saliency_url = data["matches"][0]["saliency_url"]
        assert saliency_url.startswith(f"/evidence/{case_id}/")
        assert saliency_url.endswith("_saliency.png")

        # Verify the file was created in the temp directory
        saliency_filename = os.path.basename(saliency_url)
        expected_file_path = tmp_path / "evidence" / case_id / saliency_filename
        assert expected_file_path.exists()
