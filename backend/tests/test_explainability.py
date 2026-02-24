import os
os.environ["TESTING"] = "True"
import numpy as np
import tensorflow as tf
import pytest
from backend.explainability import generate_gradcam, superimpose_heatmap

@pytest.fixture
def mock_model():
    # Create a tiny functional model for testing
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Conv2D(4, (3, 3), name="conv2d_final")(inputs)
    outputs = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(4, activation="softmax")(outputs)
    return tf.keras.Model(inputs, outputs)

def test_generate_gradcam_logic(mock_model):
    # Create a dummy image tensor
    img_tensor = tf.random.uniform((1, 224, 224, 3))
    
    heatmap = generate_gradcam(
        img_tensor, 
        mock_model, 
        last_conv_layer_name="conv2d_final"
    )
    
    assert heatmap is not None
    assert heatmap.shape == (222, 222) # Conv output size for 224 with 3x3 kernel
    assert np.max(heatmap) <= 1.0
    assert np.min(heatmap) >= 0.0

def test_superimpose_heatmap_output(tmp_path):
    # Create a dummy image file
    img_path = str(tmp_path / "test_mri.png")
    import cv2
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.imwrite(img_path, dummy_img)
    
    dummy_heatmap = np.random.rand(224, 224).astype(np.float32)
    save_path = str(tmp_path / "result.png")
    
    result = superimpose_heatmap(img_path, dummy_heatmap, save_path)
    
    assert result == save_path
    assert os.path.exists(save_path)