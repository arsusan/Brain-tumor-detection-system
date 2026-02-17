import numpy as np
import tensorflow as tf
import cv2
import os

def generate_gradcam(img_tensor, model, last_conv_layer_name="conv2d_6"):
    """
    Generates Grad-CAM heatmap for your retrained Functional 4-class model.
    """
    try:
        # 1. Create a specific model that maps the input image to 
        # BOTH the last conv layer and the final predictions.
        # This is only possible because you used the Functional API in model.py.
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )

        # 2. Record gradients of the predicted class with respect to the conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_tensor)
            
            # Find the index of the highest probability class (0-3)
            top_class_index = tf.argmax(preds[0])
            top_class_channel = preds[:, top_class_index]

        # 3. Calculate gradients
        grads = tape.gradient(top_class_channel, last_conv_layer_output)

        # 4. Global Average Pooling of gradients
        # pooled_grads is a vector where each entry is the mean intensity of the gradient for a specific feature map
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # 5. Multiply the pooled gradients by the output of the last conv layer
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # 6. Apply ReLU and Normalize to [0, 1]
        # We only care about pixels that have a POSITIVE impact on the prediction
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        
        return heatmap.numpy()

    except Exception as e:
        print(f"❌ Grad-CAM failed: {e}")
        return None

def superimpose_heatmap(img_path, heatmap, save_path, alpha=0.5):
    """
    Overlays the heatmap on the original MRI image.
    """
    img = cv2.imread(img_path)
    if img is None or heatmap is None:
        return None
        
    # Resize heatmap to match original MRI dimensions
    heatmap_rescaled = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(heatmap_rescaled, (img.shape[1], img.shape[0]))
    
    # Use COLORMAP_JET (Blue-to-Red) for medical visualization
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    
    # Overlay with alpha transparency
    superimposed = cv2.addWeighted(heatmap_color, alpha, img, 1 - alpha, 0)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, superimposed)
    return save_path