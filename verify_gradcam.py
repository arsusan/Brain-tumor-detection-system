import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from backend.explainability import generate_gradcam, superimpose_heatmap

# 1. Load the model and a test image
model = tf.keras.models.load_model(r"models/final_model_cnn_20260218_011908.keras")
img_path = r"data/brain-tumor-mri-dataset/Testing/pituitary/Te-pi_0010.jpg" # Change to an existing path

# 2. Preprocess
img = cv2.imread(img_path)
img_resized = cv2.resize(img, (128, 128))
img_tensor = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)

# 3. Generate
heatmap = generate_gradcam(img_tensor, model, last_conv_layer_name="conv2d_6")

# 4. Save and Show
if heatmap is not None:
    output_path = "test_heatmap_output.png"
    superimpose_heatmap(img_path, heatmap, output_path)
    
    # Display the result
    result = cv2.imread(output_path)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title("Grad-CAM Test Result")
    plt.show()
    print(f"✅ Success! Check {output_path}")
else:
    print("❌ Failed to generate heatmap.")