from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import uvicorn
import shutil
import os
import numpy as np
import tensorflow as tf
import keras
import uuid
import cloudinary
import cloudinary.uploader
import h5py
import json
from dotenv import load_dotenv

# Internal Project Imports
from research.src.model import BrainTumorModel
from research.src.config import Config
from research.src.preprocessing import ImagePreprocessor
from .explainability import generate_gradcam, superimpose_heatmap
from .database import init_db, get_db, ScanResult, User

# 1. INITIALIZATION & CONFIGURATION
load_dotenv()

# Cloudinary Setup
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

app = FastAPI()
init_db()

CLASS_LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']

# 2. CORS CONFIGURATION
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    """Health check endpoint for Hugging Face and deployment monitoring"""
    return {
        "status": "success", 
        "message": "NeuroScan AI Backend is Running",
        "mode": "Testing" if os.getenv("TESTING") == "True" else "Production"
    }

# Create a temporary directory for local processing before uploading to cloud
TEMP_DIR = "static/temp"
os.makedirs(TEMP_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 3. AI MODEL LOADING & PREPROCESSOR SETUP
# --- THE SURGICAL PATCH START ---
@keras.utils.register_keras_serializable()
class PatchedDense(keras.layers.Dense):
    """Intercepts and removes the quantization_config argument during loading"""
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

def safe_load_model(path):
    """Loads the Keras model while bypassing version-mismatch errors"""
    try:
        # We inject the PatchedDense into the loading process
        return keras.models.load_model(
            path, 
            custom_objects={"Dense": PatchedDense},
            compile=False
        )
    except Exception as e:
        print(f"❌ Critical Load Error: {e}")
        raise e
# --- THE SURGICAL PATCH END ---

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), "models", "final_model_cnn_20260218_011908.keras")

model = None
preprocessor = None

if os.getenv("TESTING") == "True":
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(128, 128, 3)),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    preprocessor = None 
    print("⚠️ running in TESTING mode: Dummy model initialized.")

elif os.path.exists(MODEL_PATH):
    try:
        cfg = Config()
        preprocessor = ImagePreprocessor(cfg)
        print(f"🔄 Attempting to load model with patch: {MODEL_PATH}")
        model = safe_load_model(MODEL_PATH) 
        print(f"🚀 Model loaded successfully from {MODEL_PATH}")
    except Exception as final_err:
        print(f"❌ CRITICAL ERROR during model loading: {final_err}")
else:
    print(f"❌ ERROR: Model file not found at {MODEL_PATH}")


# 4. PREDICTION ENDPOINT
@app.post("/predict")
async def predict(
    user_name: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    temp_file_path = ""
    local_heatmap_path = ""
    try:
        # A. USER HANDLING
        user = db.query(User).filter(User.name == user_name).first()
        if not user:
            user = User(name=user_name)
            db.add(user)
            db.commit()
            db.refresh(user)

        # B. TEMPORARY FILE HANDLING
        scan_id = str(uuid.uuid4())
        file_ext = os.path.splitext(file.filename)[1]
        temp_file_path = os.path.join(TEMP_DIR, f"{scan_id}{file_ext}")
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # C. UPLOAD ORIGINAL TO CLOUDINARY
        upload_result = cloudinary.uploader.upload(temp_file_path, folder="neuroscan/uploads")
        original_url = upload_result['secure_url']

        # D. PREPROCESSING & PREDICTION
        if os.getenv("TESTING") == "True":
            img_tensor = tf.random.uniform((1, 128, 128, 3))
            preds = np.array([0.9, 0.05, 0.02, 0.03], dtype=np.float32)
        else:
            img_array = preprocessor.preprocess_single(temp_file_path, augment=False)
            img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
            if len(img_tensor.shape) == 3:
                img_tensor = tf.expand_dims(img_tensor, axis=0)
            
            preds = model.predict(img_tensor, verbose=0)[0]

        pred_idx = np.argmax(preds)
        label = CLASS_LABELS[pred_idx]
        confidence = float(preds[pred_idx])

        # E. HEATMAP LOGIC
        final_heatmap_url = original_url 
        
        if label != 'notumor':
            local_heatmap_path = os.path.join(TEMP_DIR, f"heatmap_{scan_id}.png")
            try:
                # Based on your logs, "conv2d_final" is the correct target layer
                heatmap = generate_gradcam(img_tensor, model, last_conv_layer_name="conv2d_final")
                if heatmap is not None and np.max(heatmap) > 0:
                    superimpose_heatmap(temp_file_path, heatmap, save_path=local_heatmap_path)
                    hm_upload = cloudinary.uploader.upload(local_heatmap_path, folder="neuroscan/results")
                    final_heatmap_url = hm_upload['secure_url']
            except Exception as grad_err:
                print(f"❌ Grad-CAM Error: {grad_err}")

        # F. DATABASE INTEGRATION
        new_record = ScanResult(
            user_id=user.id,
            filename=file.filename,
            prediction=label,
            confidence=confidence,
            prob_glioma=float(preds[0]),
            prob_meningioma=float(preds[1]),
            prob_notumor=float(preds[2]),
            prob_pituitary=float(preds[3]),
            heatmap_url=final_heatmap_url,
            original_image_url=original_url
        )
        db.add(new_record)
        db.commit()
        db.refresh(new_record)

        # G. CLEANUP
        if os.path.exists(temp_file_path): os.remove(temp_file_path)
        if local_heatmap_path and os.path.exists(local_heatmap_path): os.remove(local_heatmap_path)

        return {
            "id": new_record.id,
            "user_name": user.name,
            "prediction": label,
            "confidence": f"{confidence*100:.2f}%",
            "probabilities": {name: f"{prob*100:.1f}%" for name, prob in zip(CLASS_LABELS, preds)},
            "heatmap_url": final_heatmap_url,
            "created_at": new_record.created_at
        }
    except Exception as e:
        if temp_file_path and os.path.exists(temp_file_path): os.remove(temp_file_path)
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# 5. HISTORY ENDPOINT
@app.get("/history")
async def get_history(db: Session = Depends(get_db)):
    results = db.query(ScanResult, User.name).\
        join(User, ScanResult.user_id == User.id).\
        order_by(ScanResult.created_at.desc()).all()
    
    return [{
        "id": scan.id,
        "user_name": name,
        "prediction": scan.prediction,
        "confidence": f"{scan.confidence*100:.2f}%",
        "created_at": scan.created_at,
        "heatmap_url": scan.heatmap_url
    } for scan, name in results]

# 6. DELETE ENDPOINT
@app.delete("/history/{scan_id}")
async def delete_scan(scan_id: int, db: Session = Depends(get_db)):
    record = db.query(ScanResult).filter(ScanResult.id == scan_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    db.delete(record)
    db.commit()
    return {"message": "Successfully deleted record"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)