from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import uvicorn
import shutil
import os
import numpy as np
import tensorflow as tf
import uuid

# Internal Project Imports
from research.src.model import BrainTumorModel
from research.src.config import Config
from research.src.preprocessing import ImagePreprocessor
from .explainability import generate_gradcam, superimpose_heatmap
from .database import init_db, get_db, ScanResult 

app = FastAPI()
CLASS_LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Enable CORS for Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup directories
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/results", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 1. Initialize Configuration and Model
cfg = Config()
preprocessor = ImagePreprocessor(cfg)
model_builder = BrainTumorModel(cfg)

# --- Pointing to your NEW model ---
MODEL_PATH = r"models/final_model_cnn_20260218_011908.keras"
model = model_builder.load_model(MODEL_PATH)

# Since it's a Functional Model, the graph is ready instantly!
print(f"🚀 Model loaded successfully from {MODEL_PATH}")
try:
    # Target the layer name from your model.py
    _ = model.get_layer("conv2d_6")
    print("✅ Grad-CAM Target Layer 'conv2d_6' detected.")
except Exception as e:
    print(f"⚠️ Warning: Could not find 'conv2d_6'. Check model.summary(). Error: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        # 2. File Handling
        scan_id = str(uuid.uuid4())
        file_ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{scan_id}{file_ext}"
        file_path = f"static/uploads/{unique_filename}"
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 3. Preprocessing & Prediction
        img_array = preprocessor.preprocess_single(file_path, augment=False)
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        if len(img_tensor.shape) == 3:
            img_tensor = tf.expand_dims(img_tensor, axis=0)
        
        preds = model.predict(img_tensor, verbose=0)[0] 
        pred_idx = np.argmax(preds)
        label = CLASS_LABELS[pred_idx]
        confidence = float(preds[pred_idx])

        # 4. HEATMAP LOGIC
        heatmap_filename = f"heatmap_{scan_id}.png"
        # We store the local path for DB, but return a URL to frontend
        heatmap_path = f"static/results/{heatmap_filename}"
        
        try:
            # Generate heatmap using the optimized Functional API logic
            heatmap = generate_gradcam(img_tensor, model, last_conv_layer_name="conv2d_6") 
            
            if heatmap is not None and np.max(heatmap) > 0:
                superimpose_heatmap(file_path, heatmap, save_path=heatmap_path)
                print(f"✅ Heatmap saved to {heatmap_path}")
            else:
                print("⚠️ Heatmap generated was empty. Using original image as fallback.")
                shutil.copy(file_path, heatmap_path) 
        except Exception as grad_err:
            print(f"❌ Grad-CAM Error: {grad_err}")
            shutil.copy(file_path, heatmap_path)

        # 5. DATABASE INTEGRATION
        new_record = ScanResult(
            filename=unique_filename,
            prediction=label,
            confidence=confidence,
            prob_glioma=float(preds[0]),
            prob_meningioma=float(preds[1]),
            prob_notumor=float(preds[2]),
            prob_pituitary=float(preds[3]),
            heatmap_url=heatmap_path 
        )
        db.add(new_record)
        db.commit()
        db.refresh(new_record)

        # 6. RESPONSE
        return {
            "id": new_record.id,
            "prediction": label,
            "confidence": f"{confidence*100:.2f}%",
            "probabilities": {
                "glioma": f"{float(preds[0])*100:.1f}%",
                "meningioma": f"{float(preds[1])*100:.1f}%",
                "notumor": f"{float(preds[2])*100:.1f}%",
                "pituitary": f"{float(preds[3])*100:.1f}%"
            },
            # Return absolute URL so the frontend can display it immediately
            "heatmap_url": f"http://127.0.0.1:8000/{heatmap_path}",
            "created_at": new_record.created_at
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# History and Delete endpoints remain the same...

@app.get("/history")
async def get_history(db: Session = Depends(get_db)):
    return db.query(ScanResult).order_by(ScanResult.created_at.desc()).all()

@app.delete("/history/{scan_id}")
async def delete_scan(scan_id: int, db: Session = Depends(get_db)):
    record = db.query(ScanResult).filter(ScanResult.id == scan_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    
    # Physical file cleanup
    for path in [f"static/uploads/{record.filename}", record.heatmap_url]:
        if path and os.path.exists(path):
            os.remove(path)
            
    db.delete(record)
    db.commit()
    return {"message": "Successfully deleted record"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)