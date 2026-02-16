#!/usr/bin/env python3
"""
Brain Tumor Detection - Basic Streamlit UI
Final Year Project (Research Only)
"""

import streamlit as st
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image
import cv2
import tensorflow as tf


# Page config

st.set_page_config(
    page_title="Brain Tumor Detection",
    layout="centered"
)

st.title("🧠 Brain Tumor Detection")
st.caption("AI-based MRI analysis (Research Use Only)")

st.warning(
    "⚠️ This AI analysis is intended for research and decision-support purposes only. "
    "Final diagnosis must be made by a qualified radiologist."
)

st.divider()


# Predictor

class Predictor:
    def __init__(self):
        self.model = None
        self.image_size = (160, 160)
        self.model_path = None

    def load_latest_model(self):
        models_dir = Path("models")
        if not models_dir.exists():
            return False

        models = list(models_dir.glob("*.keras")) + list(models_dir.glob("*.h5"))
        if not models:
            return False

        models.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        self.model_path = models[0].name
        self.model = tf.keras.models.load_model(models[0])
        return True

    def preprocess(self, img):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        img = cv2.resize(img, self.image_size)
        img = img.astype(np.float32) / 255.0
        return img

    def predict(self, img_array, threshold=0.5):
        img = self.preprocess(img_array)
        score = self.model.predict(
            np.expand_dims(img, axis=0),
            verbose=0
        )[0][0]

        if score >= threshold:
            diagnosis = "TUMOR DETECTED"
            confidence = float(score)
        else:
            diagnosis = "NO TUMOR DETECTED"
            confidence = 1.0 - float(score)

        return {
            "diagnosis": diagnosis,
            "confidence": confidence,
            "raw_score": float(score),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }


# Session state

if "predictor" not in st.session_state:
    st.session_state.predictor = Predictor()

if "history" not in st.session_state:
    st.session_state.history = []


# Load model

if not st.session_state.predictor.model:
    with st.spinner("Loading latest trained model..."):
        loaded = st.session_state.predictor.load_latest_model()
        if not loaded:
            st.error("No trained model found in `models/` directory.")
            st.code("python train.py")
            st.stop()

st.success(f"Model loaded: {st.session_state.predictor.model_path}")

st.divider()


# Upload & predict

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

threshold = st.slider(
    "Decision Threshold",
    0.0, 1.0, 0.5, 0.05
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption=uploaded_file.name, width=300)

    if st.button("Run Analysis"):
        with st.spinner("Running AI inference..."):
            img_array = np.array(image)
            result = st.session_state.predictor.predict(
                img_array, threshold
            )

            st.session_state.history.append({
                "file": uploaded_file.name,
                "result": result
            })

        st.subheader("Result")

        if "TUMOR" in result["diagnosis"]:
            st.error(result["diagnosis"])
        else:
            st.success(result["diagnosis"])

        st.write(f"**Confidence:** {result['confidence']:.2%}")
        st.write(f"**Raw Score:** {result['raw_score']:.4f}")
        st.write(f"**Time:** {result['timestamp']}")

st.divider()


# Recent history

if st.session_state.history:
    st.subheader("Recent Analyses")

    for item in st.session_state.history[-5:][::-1]:
        r = item["result"]
        st.write(
            f"- `{item['file']}` → {r['diagnosis']} "
            f"({r['confidence']:.0%})"
        )

st.divider()
st.caption(
    "Final Year Project • BSc (Hons) BIT • Susan Aryal • 2026"
)
