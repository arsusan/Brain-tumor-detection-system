import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
from backend.main import app

client = TestClient(app)

@pytest.fixture
def mock_upload():
    # Mocks Cloudinary uploader
    with patch("cloudinary.uploader.upload") as mocked:
        mocked.return_value = {"secure_url": "https://test-cloud.com/image.png"}
        yield mocked

@pytest.fixture
def mock_ai_model():
    # Mocks the model.predict behavior
    with patch("backend.main.model") as mocked:
        # Simulate a 'glioma' prediction (Index 0)
        mocked.predict.return_value = np.array([[0.9, 0.05, 0.02, 0.03]])
        yield mocked

def test_predict_endpoint_success(mock_upload, mock_ai_model):
    # Prepare a dummy image for upload
    file_content = b"fake-image-binary-content"
    
    response = client.post(
        "/predict",
        data={"user_name": "TestUser"},
        files={"file": ("test.jpg", file_content, "image/jpeg")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["user_name"] == "TestUser"
    assert "prediction" in data
    assert "heatmap_url" in data
    assert data["heatmap_url"].startswith("https://")

def test_history_endpoint(mock_ai_model):
    response = client.get("/history")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_delete_scan_invalid_id():
    # Testing 404 behavior
    response = client.delete("/history/999999")
    assert response.status_code == 404