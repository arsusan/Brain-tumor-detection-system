from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
import uvicorn
import shutil
import os
import numpy as np
import tensorflow as tf
from PIL import Image

from research.src.model import BrainTumorModel
from research.src.config import Config
from research.src.preprocessing import ImagePreprocessor
from .explainability import generate_gradcam, superimpose_heatmap

app = FastAPI()

# Ensure directories exist
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/results", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load configuration and model
cfg = Config()
preprocessor = ImagePreprocessor(cfg)
model_builder = BrainTumorModel(cfg)
MODEL_PATH = "models/final_model_cnn_20260207_003723.keras"
model = model_builder.load_model(MODEL_PATH)

# --- THE CRITICAL FIX: Initialization ---
print("🚀 Initializing Model Tracing...")
dummy_input = tf.zeros((1, 128, 128, 3))
# Warmup the main wrapper
_ = model(dummy_input, training=False)

# Warmup the internal Sequential model specifically
try:
    inner = model.get_layer("BrainTumorCNN")
    _ = inner(dummy_input, training=False)
    print("✅ BrainTumorCNN tracing complete.")
except Exception:
    print("ℹ️ Model is flat; no internal wrapper to trace.")

print("✅ Backend Server is Ready.")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 1. Save and Preprocess
        file_path = f"static/uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Use your research preprocessor
        img_array = preprocessor.preprocess_single(file_path, augment=False)
        # Convert to Tensor and add batch dimension
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        img_tensor = tf.expand_dims(img_tensor, axis=0)

        # 2. Get Prediction
        # Pass the tensor directly to avoid the 'not called' error
        preds = model(img_tensor, training=False)
        prediction_val = float(preds[0][0])
        
        label = "Tumor Detected" if prediction_val > 0.5 else "No Tumor"
        confidence = prediction_val if prediction_val > 0.5 else 1 - prediction_val

        # 3. Generate Heatmap
        print(f"🔍 Analyzing image: {file.filename}")
        # Pass the SAME img_tensor we used for prediction
        heatmap = generate_gradcam(img_tensor, model, last_conv_layer_name="conv2d_6")
        output_image_path = superimpose_heatmap(file_path, heatmap)

        return {
            "prediction": label,
            "confidence": f"{confidence*100:.2f}%",
            "heatmap_url": f"/{output_image_path}"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)