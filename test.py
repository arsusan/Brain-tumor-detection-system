import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

def test_heatmap(img_path, model_path, last_layer_name):
    model = tf.keras.models.load_model(model_path)
    
    # Preprocess image
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (224, 224)) # Adjust size to match your model
    img_array = np.expand_dims(img_resized, axis=0) / 255.0

    # Gradient Model
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, np.argmax(preds[0])]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    heatmap = last_conv_layer_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # CHECK INTENSITY
    max_val = np.max(heatmap)
    print(f"Heatmap Peak Intensity: {max_val}")
    
    if max_val <= 0:
        print("ERROR: Heatmap intensity is zero or negative. Math failed.")
    else:
        heatmap = np.maximum(heatmap, 0) / max_val
        plt.matshow(heatmap)
        plt.show()
        print("Heatmap generated and displayed successfully!")

# Usage (Update these paths)
test_heatmap('research\data\brain-tumor-mri-dataset\testing\glioma\Te-gl_0011.jpg', 'models/final_model_cnn_20260217_173102.keras', 'conv2d_6')