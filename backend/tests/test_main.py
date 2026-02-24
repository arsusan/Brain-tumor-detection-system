import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import numpy as np
from backend.main import app
from backend.database import User, ScanResult

client = TestClient(app)

@pytest.fixture
def mock_upload():
    with patch("cloudinary.uploader.upload") as mocked:
        mocked.return_value = {"secure_url": "https://test-cloud.com/image.png"}
        yield mocked

@pytest.fixture
def mock_ai_model():
    with patch("backend.main.model") as mocked:
        # Probabilities for: [glioma, meningioma, no_tumor, pituitary]
        mocked.predict.return_value = np.array([[0.9, 0.05, 0.02, 0.03]])
        yield mocked

def test_predict_endpoint_success(mock_upload, mock_ai_model, db_session):
    # 1. Create User
    user_name = "TestUser"
    test_user = User(name=user_name)
    db_session.add(test_user)
    db_session.commit()

    # 2. Post Request
    file_content = b"fake-image-binary-content"
    response = client.post(
        "/predict",
        data={"user_name": user_name},
        files={"file": ("test.jpg", file_content, "image/jpeg")}
    )
    
    # Debugging print if it fails
    if response.status_code != 200:
        print(f"Response Error: {response.json()}")

    assert response.status_code == 200
    data = response.json()
    
    # Check keys based on your actual API response structure
    assert "prediction" in data
    assert data["prediction"] == "glioma"

def test_history_endpoint_with_data(db_session):
    # Seed data
    user = User(name="HistoryUser")
    db_session.add(user)
    db_session.commit()
    
    scan = ScanResult(
        user_id=user.id, 
        prediction="meningioma", 
        confidence=0.85,
        filename="test.jpg"
    )
    db_session.add(scan)
    db_session.commit()

    response = client.get("/history")
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0
    # Match the last added scan
    assert data[0]["prediction"] == "meningioma"

def test_delete_scan_invalid_id():
    # Should work now that 'scans' table exists
    response = client.delete("/history/999999")
    assert response.status_code == 404