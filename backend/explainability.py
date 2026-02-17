import numpy as np
import tensorflow as tf
import cv2
from tensorflow import keras

def generate_gradcam(img_array, model, last_conv_layer_name="conv2d_6"):
    # 1. Find the target layer
    try:
        target_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        # If it's nested inside 'BrainTumorCNN'
        inner_model = model.get_layer("BrainTumorCNN")
        target_layer = inner_model.get_layer(last_conv_layer_name)
        model = inner_model # Shift focus to the inner model

    # 2. Reconstruct the model using the Functional API 
    # This specifically avoids the 'never been called' AttributeError
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.layers[-1].output]
    )

    # 3. Record gradients
    with tf.GradientTape() as tape:
        img_tensor = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(img_tensor)
        # Assuming binary classification (index 0)
        loss = predictions[:, 0]

    # 4. Extract and process gradients
    grads = tape.gradient(loss, conv_outputs)
    
    if grads is None:
        return np.zeros((128, 128))

    # Average the gradients across the width and height
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map by 'how important it is'
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def superimpose_heatmap(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    if img is None: return ""
    img = cv2.resize(img, (128, 128))
    heatmap = cv2.resize(heatmap, (128, 128))
    
    heatmap_color = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)
    
    # Merge heatmap with original image
    superimposed = cv2.addWeighted(heatmap_color, alpha, img, 1 - alpha, 0)
    output_path = "static/results/latest_gradcam.png"
    cv2.imwrite(output_path, superimposed)
    return output_path