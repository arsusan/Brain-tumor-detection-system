import os
os.environ["TESTING"] = "True"
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
import tensorflow as tf
from backend.main import app
from backend.database import User, ScanResult

client = TestClient(app)

# --- FIXTURES ---

@pytest.fixture(scope="session")
def dummy_model():
    """
    Creates a tiny, lightweight model for CI testing.
    This prevents the runner from loading the heavy 100MB+ real model.
    Adjust input_shape to match your real model (e.g., 150, 150, 3).
    """
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    return model

@pytest.fixture
def mock_upload():
    """Mocks Cloudinary image uploading."""
    with patch("cloudinary.uploader.upload") as mocked:
        mocked.return_value = {
            "secure_url": "https://test-cloud.com/image.png",
            "public_id": "test_id"
        }
        yield mocked

@pytest.fixture
def mock_ai_model(dummy_model):
    """
    Swaps the real model in main.py with our tiny dummy model.
    Also mocks the 'predict' return value for consistency.
    """
    with patch("backend.main.model") as mocked:
        # Mocking the predict call to return specific probabilities:
        # [glioma, meningioma, no_tumor, pituitary]
        mocked.predict.return_value = np.array([[0.9, 0.05, 0.02, 0.03]])
        # Ensure it has the dummy model structure if needed
        mocked.input_shape = dummy_model.input_shape
        yield mocked

# --- TEST CASES ---

def test_predict_endpoint_success(mock_upload, mock_ai_model, db_session):
    """Tests the full prediction flow from upload to DB saving."""
    # 1. Create User in the isolated test DB
    user_name = "TestUser"
    test_user = User(name=user_name)
    db_session.add(test_user)
    db_session.commit()

    # 2. Simulate image upload
    file_content = b"fake-image-binary-content"
    response = client.post(
        "/predict",
        data={"user_name": user_name},
        files={"file": ("test.jpg", file_content, "image/jpeg")}
    )
    
    # Check for success
    if response.status_code != 200:
        print(f"Response Error: {response.json()}")

    assert response.status_code == 200
    data = response.json()
    
    # Assertions based on your API response keys
    assert "prediction" in data
    assert data["prediction"] == "glioma"  # Matches our 0.9 mock probability
    assert "heatmap_url" in data
    assert data["user_name"] == user_name

def test_history_endpoint_with_data(db_session):
    """Tests that the history route returns records from the DB."""
    # 1. Seed the test database
    user = User(name="HistoryUser")
    db_session.add(user)
    db_session.commit()
    
    scan = ScanResult(
        user_id=user.id, 
        prediction="meningioma", 
        confidence=0.85,
        filename="test.jpg",
        heatmap_url="https://test.com/heat.png"
    )
    db_session.add(scan)
    db_session.commit()

    # 2. Fetch history
    response = client.get("/history")
    assert response.status_code == 200
    data = response.json()
    
    assert isinstance(data, list)
    assert len(data) > 0
    # Ensure the most recent scan is returned
    assert data[0]["prediction"] == "meningioma"

def test_delete_scan_invalid_id():
    """Tests 404 behavior for deleting non-existent records."""
    response = client.delete("/history/999999")
    assert response.status_code == 404