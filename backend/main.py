from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
import uvicorn
import shutil
import os
import numpy as np
import tensorflow as tf
from PIL import Image

# Internal Project Imports
from research.src.model import BrainTumorModel
from research.src.config import Config
from research.src.preprocessing import ImagePreprocessor
from .explainability import generate_gradcam, superimpose_heatmap
from .database import init_db, get_db, ScanResult 

app = FastAPI()

# --- 1. SETUP DIRECTORIES ---
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/results", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- 2. LOAD MODEL & DB ---
cfg = Config()
preprocessor = ImagePreprocessor(cfg)
model_builder = BrainTumorModel(cfg)
MODEL_PATH = "models/final_model_cnn_20260207_003723.keras"
model = model_builder.load_model(MODEL_PATH)

# Initialize PostgreSQL Tables
init_db() 

# --- 3. MODEL WARMUP (CRITICAL FOR GRAD-CAM) ---
print("🚀 Initializing Model Tracing...")
dummy_input = tf.zeros((1, 128, 128, 3))
_ = model(dummy_input, training=False)

try:
    inner = model.get_layer("BrainTumorCNN")
    _ = inner(dummy_input, training=False)
    print("✅ BrainTumorCNN tracing complete.")
except Exception:
    print("ℹ️ Model is flat; no internal wrapper to trace.")

print("✅ Backend Server is Ready.")

# --- 4. ROUTES ---

@app.post("/predict")
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        # 1. Save and Preprocess
        file_path = f"static/uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        img_array = preprocessor.preprocess_single(file_path, augment=False)
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        img_tensor = tf.expand_dims(img_tensor, axis=0)

        # 2. Get Prediction
        preds = model(img_tensor, training=False)
        prediction_val = float(preds[0][0])
        
        label = "Tumor Detected" if prediction_val > 0.5 else "No Tumor"
        confidence = prediction_val if prediction_val > 0.5 else 1 - prediction_val

        # 3. Generate Heatmap
        print(f"🔍 Analyzing image: {file.filename}")
        heatmap = generate_gradcam(img_tensor, model, last_conv_layer_name="conv2d_6")
        output_image_path = superimpose_heatmap(file_path, heatmap)

        # --- 4. DATABASE INTEGRATION (SAVE TO POSTGRES) ---
        new_record = ScanResult(
            filename=file.filename,
            prediction=label,
            confidence=float(confidence),
            heatmap_url=output_image_path
        )
        db.add(new_record)      # Stage the record
        db.commit()             # Push to PostgreSQL
        db.refresh(new_record)  # Get the record back with its unique ID

        return {
            "id": new_record.id,
            "prediction": label,
            "confidence": f"{confidence*100:.2f}%",
            "heatmap_url": f"http://127.0.0.1:8000/{output_image_path}",
            "created_at": new_record.created_at
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.get("/history")
async def get_history(db: Session = Depends(get_db)):
    """Fetch all past scan results for the Frontend history page."""
    return db.query(ScanResult).order_by(ScanResult.created_at.desc()).all()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)