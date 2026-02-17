import numpy as np
import tensorflow as tf
import cv2
from tensorflow import keras

def generate_gradcam(img_tensor, model, last_conv_layer_name="conv2d_6"):
    # Since your model is "flat", we build the grad_model directly from the main model
    try:
        grad_model = keras.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
    except Exception as e:
        print(f"Error creating Grad-CAM model: {e}")
        return np.zeros((128, 128))

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        # For multi-class: Find the index of the highest prediction
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        return np.zeros((128, 128))

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def superimpose_heatmap(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    if img is None: return ""
    img = cv2.resize(img, (128, 128))
    
    heatmap_color = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)
    
    superimposed = cv2.addWeighted(heatmap_color, alpha, img, 1 - alpha, 0)
    output_path = "static/results/latest_gradcam.png"
    cv2.imwrite(output_path, superimposed)
    return output_path