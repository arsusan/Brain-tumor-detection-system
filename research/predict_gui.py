#!/usr/bin/env python3
"""
Brain Tumor Categorization - GUI Predictor (4-Class Version)
Updated for Glioma, Meningioma, Pituitary, and No Tumor classification.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.preprocessing import ImagePreprocessor

# Define the classes based on your training configuration
# IMPORTANT: These must match the order of your training generator
CLASS_LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Color mapping for the UI
CLASS_COLORS = {
    'glioma': '#ff6b6b',      # Light Red
    'meningioma': '#fcc419',  # Orange/Yellow
    'notumor': '#51cf66',     # Green
    'pituitary': '#cc5de8'    # Purple
}

try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                QHBoxLayout, QLabel, QPushButton, QTextEdit, 
                                QListWidget, QListWidgetItem, QFileDialog, 
                                QMessageBox, QProgressBar, QGroupBox, QSplitter,
                                QFrame, QScrollArea, QGridLayout, QDialog)
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QPixmap, QFont, QColor
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False
    print("❌ PyQt5 not found. Please install it: pip install PyQt5")
    sys.exit(1)

class PredictionWorker(QThread):
    """Worker thread for multi-class prediction"""
    prediction_complete = pyqtSignal(dict)
    progress_update = pyqtSignal(int, int)
    image_processed = pyqtSignal(str, str, float)
    
    def __init__(self, predictor, image_paths):
        super().__init__()
        self.predictor = predictor
        self.image_paths = image_paths
        
    def run(self):
        results = []
        total = len(self.image_paths)
        
        for i, image_path in enumerate(self.image_paths):
            try:
                result = self.predictor.predict_single(image_path)
                results.append(result)
                self.image_processed.emit(
                    image_path,
                    result['predicted_class'],
                    result['confidence']
                )
            except Exception as e:
                results.append({
                    'image_path': image_path,
                    'error': str(e),
                    'predicted_class': 'Error',
                    'confidence': 0.0
                })
            self.progress_update.emit(i + 1, total)
        
        self.prediction_complete.emit({'predictions': results})

class BrainTumorPredictorGUI:
    """Logic handler for the 4-class model"""
    def __init__(self, model_path=None):
        self.config = Config()
        self.model_path = model_path
        self.predictor_model = None
        self.preprocessor = ImagePreprocessor(self.config)
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path):
        try:
            import tensorflow as tf
            print(f"📂 Loading Multi-Class Model: {model_path}")
            self.predictor_model = tf.keras.models.load_model(model_path)
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict_single(self, image_path):
        """Modified for 4-Class Softmax Output"""
        if self.predictor_model is None:
            raise ValueError("Model not loaded.")
        
        # Preprocess using the shared preprocessor
        image = self.preprocessor.preprocess_single(image_path, augment=False)
        
        # Predict: Expected output shape is (1, 4)
        predictions = self.predictor_model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
        
        # Get highest probability index
        predicted_idx = np.argmax(predictions)
        predicted_class = CLASS_LABELS[predicted_idx]
        confidence = float(predictions[predicted_idx])
        
        return {
            'image_path': image_path,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_scores': {CLASS_LABELS[i]: float(predictions[i]) for i in range(len(CLASS_LABELS))},
            'timestamp': datetime.now().isoformat()
        }

class PyQt5GUI(QMainWindow):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor
        self.image_files = []
        self.predictions = []
        self.init_ui()
        self.apply_dark_theme()
        self.setWindowTitle("🧠 Brain Tumor AI - 4 Class Analyzer")
        self.resize(1100, 800)

    def apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #121212; color: white; }
            QGroupBox { border: 1px solid #333; margin-top: 15px; font-weight: bold; padding: 10px; }
            QPushButton { background-color: #007acc; padding: 8px; border-radius: 4px; font-weight: bold; }
            QPushButton#success { background-color: #2b8a3e; }
            QPushButton#danger { background-color: #c92a2a; }
            QListWidget, QTextEdit { background-color: #1e1e1e; border: 1px solid #333; }
            QProgressBar { border: 1px solid #333; text-align: center; }
            QProgressBar::chunk { background-color: #007acc; }
        """)

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Header
        title = QLabel("BRAIN TUMOR CATEGORIZATION SYSTEM")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Main Splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left Panel (Controls)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        group_io = QGroupBox("Input / Output")
        io_layout = QVBoxLayout()
        self.load_btn = QPushButton("📂 Load MRI Scans")
        self.load_btn.clicked.connect(self.load_images)
        self.predict_btn = QPushButton("🔍 Start Categorization")
        self.predict_btn.setObjectName("success")
        self.predict_btn.setEnabled(False)
        self.predict_btn.clicked.connect(self.start_prediction)
        self.clear_btn = QPushButton("🗑️ Clear")
        self.clear_btn.setObjectName("danger")
        self.clear_btn.clicked.connect(self.clear_all)
        
        io_layout.addWidget(self.load_btn)
        io_layout.addWidget(self.predict_btn)
        io_layout.addWidget(self.clear_btn)
        group_io.setLayout(io_layout)
        
        self.image_list = QListWidget()
        left_layout.addWidget(group_io)
        left_layout.addWidget(self.image_list)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        # Right Panel (Results)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.results_scroll = QScrollArea()
        self.results_widget = QWidget()
        self.results_grid = QGridLayout(self.results_widget)
        self.results_scroll.setWidget(self.results_widget)
        self.results_scroll.setWidgetResizable(True)
        
        self.summary_box = QTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setMaximumHeight(150)
        
        right_layout.addWidget(QLabel("Analysis Results:"))
        right_layout.addWidget(self.results_scroll)
        right_layout.addWidget(self.summary_box)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        layout.addWidget(splitter)

    def load_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Open MRI", "", "Images (*.jpg *.png *.jpeg)")
        if files:
            self.image_files.extend(files)
            for f in files:
                self.image_list.addItem(os.path.basename(f))
            self.predict_btn.setEnabled(True)

    def clear_all(self):
        self.image_files = []
        self.image_list.clear()
        self.summary_box.clear()
        self.predict_btn.setEnabled(False)
        for i in reversed(range(self.results_grid.count())): 
            self.results_grid.itemAt(i).widget().setParent(None)

    def start_prediction(self):
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(self.image_files))
        self.worker = PredictionWorker(self.predictor, self.image_files)
        self.worker.image_processed.connect(self.add_result_card)
        self.worker.progress_update.connect(lambda cur, tot: self.progress_bar.setValue(cur))
        self.worker.prediction_complete.connect(self.finalize_summary)
        self.worker.start()

    def add_result_card(self, path, label, conf):
        card = QFrame()
        card.setFrameShape(QFrame.StyledPanel)
        card.setStyleSheet(f"border: 2px solid {CLASS_COLORS.get(label, '#333')}; border-radius: 5px; padding: 5px;")
        l = QVBoxLayout(card)
        
        pix = QPixmap(path).scaled(120, 120, Qt.KeepAspectRatio)
        img_label = QLabel()
        img_label.setPixmap(pix)
        img_label.setAlignment(Qt.AlignCenter)
        
        res_label = QLabel(f"{label.upper()}\n{conf:.1%}")
        res_label.setAlignment(Qt.AlignCenter)
        res_label.setStyleSheet(f"color: {CLASS_COLORS.get(label, 'white')}; font-weight: bold;")
        
        l.addWidget(img_label)
        l.addWidget(res_label)
        
        idx = self.results_grid.count()
        self.results_grid.addWidget(card, idx // 3, idx % 3)

    def finalize_summary(self, data):
        self.progress_bar.setVisible(False)
        preds = data['predictions']
        counts = {c: sum(1 for p in preds if p['predicted_class'] == c) for c in CLASS_LABELS}
        summary = "📊 FINAL REPORT\n" + "="*20 + "\n"
        for c, count in counts.items():
            summary += f"{c.capitalize()}: {count}\n"
        self.summary_box.setPlainText(summary)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # REPLACE WITH YOUR ACTUAL MODEL PATH
    MODEL_PATH = r"D:\Brain-tumor-detection-system\research\models\final_model_cnn_20260217_173102.keras"
    predictor = BrainTumorPredictorGUI(MODEL_PATH)
    gui = PyQt5GUI(predictor)
    gui.show()
    sys.exit(app.exec_())