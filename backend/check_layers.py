import tensorflow as tf

# Load your specific model file
MODEL_PATH = "models/final_model_cnn_20260217_173102.keras"  # Change this to your actual model filename

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("\n--- MODEL SUMMARY ---")
    model.summary()
    
    print("\n--- CANDIDATE LAYERS FOR HEATMAP ---")
    print("Look for the LAST 'Conv2D' layer in the list below:")
    
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            print(f"Layer Name: {layer.name} | Output Shape: {layer.output_shape}")

except Exception as e:
    print(f"Error: {e}")