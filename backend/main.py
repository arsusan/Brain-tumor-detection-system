from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import uvicorn
import shutil
import os
import numpy as np
import tensorflow as tf

# Internal Project Imports
from research.src.model import BrainTumorModel
from research.src.config import Config
from research.src.preprocessing import ImagePreprocessor
from .explainability import generate_gradcam, superimpose_heatmap
from .database import init_db, get_db, ScanResult 


app = FastAPI()

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

# Load configuration and Categorization Model
cfg = Config()
preprocessor = ImagePreprocessor(cfg)
model_builder = BrainTumorModel(cfg)
# Update this filename to your 4-class model
MODEL_PATH = "models/final_model_cnn_20260217_173102.keras" 
model = model_builder.load_model(MODEL_PATH)

CLASS_LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']

init_db() 

# --- 3. MODEL WARMUP ---
print("🚀 Initializing Multiclass Model Tracing...")
dummy_input = tf.zeros((1, 128, 128, 3))
_ = model(dummy_input, training=False) # This is enough for a flat model
print("✅ Backend Categorization Server is Ready.")

@app.post("/predict")
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        file_path = f"static/uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        img_array = preprocessor.preprocess_single(file_path, augment=False)
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        img_tensor = tf.expand_dims(img_tensor, axis=0)

        # Get 4-class prediction
        preds = model(img_tensor, training=False).numpy()[0]
        pred_idx = np.argmax(preds)
        label = CLASS_LABELS[pred_idx]
        confidence = float(preds[pred_idx])

        # Generate Heatmap
        print(f"🔍 Analyzing MRI: {file.filename} -> {label}")
        heatmap = generate_gradcam(img_tensor, model, last_conv_layer_name="conv2d_6")
        output_image_path = superimpose_heatmap(file_path, heatmap)

        # Database Integration
        new_record = ScanResult(
            filename=file.filename,
            prediction=label,
            confidence=confidence,
            prob_glioma=float(preds[0]),
            prob_meningioma=float(preds[1]),
            prob_notumor=float(preds[2]),
            prob_pituitary=float(preds[3]),
            heatmap_url=output_image_path
        )
        db.add(new_record)
        db.commit()
        db.refresh(new_record)

        return {
            "id": new_record.id,
            "prediction": label,
            "confidence": f"{confidence*100:.2f}%",
            "probabilities": {CLASS_LABELS[i]: f"{preds[i]*100:.2f}%" for i in range(4)},
            "heatmap_url": f"http://127.0.0.1:8000/{output_image_path}",
            "created_at": new_record.created_at
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.get("/history")
async def get_history(db: Session = Depends(get_db)):
    return db.query(ScanResult).order_by(ScanResult.created_at.desc()).all()

# --- MOVE THIS SECTION ABOVE THE MAIN BLOCK ---
@app.delete("/history/{scan_id}")
async def delete_scan(scan_id: int, db: Session = Depends(get_db)):
    record = db.query(ScanResult).filter(ScanResult.id == scan_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    
    db.delete(record)
    db.commit()
    return {"message": "Successfully deleted record"}

# main 
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)