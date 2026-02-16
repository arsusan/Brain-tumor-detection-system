#!/usr/bin/env python3
"""
Brain Tumor Detection - GUI Predictor
Drag & drop MRI images to see tumor predictions
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

try:
    # Try PyQt5 first (more professional)
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                QHBoxLayout, QLabel, QPushButton, QTextEdit, 
                                QListWidget, QListWidgetItem, QFileDialog, 
                                QMessageBox, QProgressBar, QGroupBox, QSplitter,
                                QFrame, QScrollArea, QGridLayout, QSizePolicy)
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
    from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QPalette, QIcon
    QT_AVAILABLE = True
except ImportError:
    # Fallback to Tkinter
    QT_AVAILABLE = False
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from PIL import Image, ImageTk
    import threading

class PredictionWorker(QThread):
    """Worker thread for prediction to keep GUI responsive"""
    prediction_complete = pyqtSignal(dict)
    progress_update = pyqtSignal(int, int)  # current, total
    image_processed = pyqtSignal(str, str, float)  # path, class, confidence
    
    def __init__(self, predictor, image_paths, threshold=0.5):
        super().__init__()
        self.predictor = predictor
        self.image_paths = image_paths
        self.threshold = threshold
        
    def run(self):
        """Run prediction in background thread"""
        results = []
        total = len(self.image_paths)
        
        for i, image_path in enumerate(self.image_paths):
            try:
                result = self.predictor.predict_single(image_path, self.threshold)
                results.append(result)
                
                # Emit signals for GUI updates
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
            
            # Update progress
            self.progress_update.emit(i + 1, total)
        
        self.prediction_complete.emit({'predictions': results})

class BrainTumorPredictorGUI:
    """Brain tumor predictor with GUI"""
    
    def __init__(self, model_path=None):
        self.config = Config()
        self.model_path = model_path
        self.predictor = None
        self.current_predictions = []
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load the trained model"""
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            print(f"📂 Loading model from: {model_path}")
            self.predictor_model = keras.models.load_model(model_path)
            
            # Initialize preprocessor
            self.preprocessor = ImagePreprocessor(self.config)
            
            print("✅ Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict_single(self, image_path, threshold=0.5):
        """Make prediction on a single image"""
        if not hasattr(self, 'predictor_model'):
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Preprocess image
        image = self.preprocessor.preprocess_single(image_path, augment=False)
        
        # Make prediction
        prediction = self.predictor_model.predict(
            np.expand_dims(image, axis=0), verbose=0
        )[0][0]
        
        # Determine class and confidence
        if prediction > threshold:
            predicted_class = "Tumor"
            confidence = float(prediction)
        else:
            predicted_class = "No Tumor"
            confidence = 1.0 - float(prediction)
        
        return {
            'image_path': image_path,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'raw_score': float(prediction),
            'threshold': threshold,
            'timestamp': datetime.now().isoformat()
        }

class PyQt5GUI(QMainWindow):
    """PyQt5 based GUI application"""
    
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor
        self.image_files = []
        self.predictions = []
        self.worker_thread = None
        
        self.init_ui()
        self.setWindowTitle("🧠 Brain Tumor Detection - MRI Analyzer")
        self.setGeometry(100, 100, 1200, 700)
        
        # Apply dark theme
        self.apply_dark_theme()
    
    def apply_dark_theme(self):
        """Apply a modern dark theme"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QLabel {
                color: #ffffff;
                font-size: 12px;
            }
            QLabel#title {
                color: #ffffff;
                font-size: 20px;
                font-weight: bold;
            }
            QLabel#subtitle {
                color: #cccccc;
                font-size: 14px;
            }
            QPushButton {
                background-color: #007acc;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #505050;
                color: #808080;
            }
            QPushButton#danger {
                background-color: #d13438;
            }
            QPushButton#danger:hover {
                background-color: #a4262c;
            }
            QPushButton#success {
                background-color: #107c10;
            }
            QPushButton#success:hover {
                background-color: #0c5c0c;
            }
            QListWidget {
                background-color: #252526;
                color: #cccccc;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                font-size: 11px;
            }
            QListWidget::item {
                padding: 4px;
            }
            QListWidget::item:selected {
                background-color: #094771;
            }
            QTextEdit {
                background-color: #252526;
                color: #cccccc;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                font-family: Consolas, monospace;
                font-size: 11px;
            }
            QGroupBox {
                color: #cccccc;
                border: 2px solid #3e3e42;
                border-radius: 6px;
                margin-top: 10px;
                font-weight: bold;
                font-size: 13px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QProgressBar {
                border: 1px solid #3e3e42;
                border-radius: 4px;
                text-align: center;
                color: white;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #007acc;
                border-radius: 4px;
            }
            QFrame#separator {
                background-color: #3e3e42;
                max-height: 1px;
                min-height: 1px;
            }
        """)
    
    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Title
        title_label = QLabel("🧠 BRAIN TUMOR DETECTION - MRI ANALYZER")
        title_label.setObjectName("title")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        subtitle_label = QLabel("Drag & drop MRI images or use buttons below")
        subtitle_label.setObjectName("subtitle")
        subtitle_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(subtitle_label)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setObjectName("separator")
        main_layout.addWidget(separator)
        
        # Create splitter for left/right panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Image selection
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Image selection group
        selection_group = QGroupBox("📁 Image Selection")
        selection_layout = QVBoxLayout()
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("📂 Load Images")
        self.load_btn.clicked.connect(self.load_images)
        button_layout.addWidget(self.load_btn)
        
        self.load_folder_btn = QPushButton("📁 Load Folder")
        self.load_folder_btn.clicked.connect(self.load_folder)
        button_layout.addWidget(self.load_folder_btn)
        
        self.clear_btn = QPushButton("🗑️ Clear All")
        self.clear_btn.setObjectName("danger")
        self.clear_btn.clicked.connect(self.clear_images)
        button_layout.addWidget(self.clear_btn)
        
        selection_layout.addLayout(button_layout)
        
        # Drop area
        self.drop_label = QLabel("📤 Drag & drop MRI images here\n(.jpg, .png, .jpeg)")
        self.drop_label.setAlignment(Qt.AlignCenter)
        self.drop_label.setStyleSheet("""
            QLabel {
                background-color: #252526;
                border: 2px dashed #3e3e42;
                border-radius: 8px;
                padding: 40px;
                color: #808080;
                font-size: 14px;
            }
        """)
        self.drop_label.setAcceptDrops(True)
        self.drop_label.dragEnterEvent = self.drag_enter_event
        self.drop_label.dropEvent = self.drop_event
        selection_layout.addWidget(self.drop_label)
        
        # Image list
        self.image_list = QListWidget()
        self.image_list.itemDoubleClicked.connect(self.show_image_details)
        selection_layout.addWidget(self.image_list)
        
        selection_group.setLayout(selection_layout)
        left_layout.addWidget(selection_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)
        
        # Add left panel to splitter
        splitter.addWidget(left_panel)
        
        # Right panel - Results
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Results group
        results_group = QGroupBox("📊 Prediction Results")
        results_layout = QVBoxLayout()
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.predict_btn = QPushButton("🔍 Analyze Images")
        self.predict_btn.setObjectName("success")
        self.predict_btn.clicked.connect(self.start_prediction)
        self.predict_btn.setEnabled(False)
        control_layout.addWidget(self.predict_btn)
        
        self.export_btn = QPushButton("💾 Export Results")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        control_layout.addWidget(self.export_btn)
        
        results_layout.addLayout(control_layout)
        
        # Results display - Use scroll area for multiple images
        self.results_scroll = QScrollArea()
        self.results_widget = QWidget()
        self.results_grid = QGridLayout(self.results_widget)
        self.results_scroll.setWidget(self.results_widget)
        self.results_scroll.setWidgetResizable(True)
        results_layout.addWidget(self.results_scroll)
        
        # Summary text
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(150)
        results_layout.addWidget(self.summary_text)
        
        results_group.setLayout(results_layout)
        right_layout.addWidget(results_group)
        
        # Add right panel to splitter
        splitter.addWidget(right_panel)
        
        # Set initial sizes
        splitter.setSizes([400, 600])
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def drag_enter_event(self, event):
        """Handle drag enter event"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def drop_event(self, event):
        """Handle drop event"""
        urls = event.mimeData().urls()
        image_paths = []
        
        for url in urls:
            file_path = url.toLocalFile()
            if self.is_image_file(file_path):
                image_paths.append(file_path)
        
        if image_paths:
            self.add_image_paths(image_paths)
    
    def is_image_file(self, file_path):
        """Check if file is an image"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
        return os.path.splitext(file_path)[1].lower() in image_extensions
    
    def load_images(self):
        """Load images using file dialog"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select MRI Images", "",
            "Image Files (*.jpg *.jpeg *.png *.bmp *.tiff);;All Files (*.*)"
        )
        
        if file_paths:
            self.add_image_paths(file_paths)
    
    def load_folder(self):
        """Load all images from a folder"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Folder with MRI Images"
        )
        
        if folder_path:
            image_paths = []
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if self.is_image_file(file):
                        image_paths.append(os.path.join(root, file))
            
            if image_paths:
                self.add_image_paths(image_paths)
            else:
                QMessageBox.warning(self, "No Images", "No image files found in the selected folder.")
    
    def add_image_paths(self, image_paths):
        """Add image paths to the list"""
        for path in image_paths:
            if path not in self.image_files:
                self.image_files.append(path)
                item = QListWidgetItem(os.path.basename(path))
                item.setData(Qt.UserRole, path)
                self.image_list.addItem(item)
        
        self.predict_btn.setEnabled(len(self.image_files) > 0)
        self.statusBar().showMessage(f"Loaded {len(image_paths)} images. Total: {len(self.image_files)}")
    
    def clear_images(self):
        """Clear all images and results"""
        self.image_files.clear()
        self.image_list.clear()
        self.clear_results()
        self.predict_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.statusBar().showMessage("Cleared all images")
    
    def clear_results(self):
        """Clear prediction results"""
        # Clear the results grid
        for i in reversed(range(self.results_grid.count())): 
            widget = self.results_grid.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        self.predictions.clear()
        self.summary_text.clear()
    
    def show_image_details(self, item):
        """Show details of a selected image"""
        image_path = item.data(Qt.UserRole)
        
        try:
            # Load and display image
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                # Scale if too large
                if pixmap.width() > 800 or pixmap.height() > 600:
                    pixmap = pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                
                # Show in dialog
                dialog = QDialog(self)
                dialog.setWindowTitle(f"Image: {os.path.basename(image_path)}")
                layout = QVBoxLayout(dialog)
                
                image_label = QLabel()
                image_label.setPixmap(pixmap)
                image_label.setAlignment(Qt.AlignCenter)
                layout.addWidget(image_label)
                
                info_label = QLabel(f"Path: {image_path}\nSize: {pixmap.width()}x{pixmap.height()}")
                layout.addWidget(info_label)
                
                dialog.setLayout(layout)
                dialog.resize(pixmap.width() + 50, pixmap.height() + 100)
                dialog.exec_()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Cannot display image: {str(e)}")
    
    def start_prediction(self):
        """Start prediction on all loaded images"""
        if not self.image_files:
            QMessageBox.warning(self, "No Images", "Please load images first.")
            return
        
        if not self.predictor.predictor_model:
            QMessageBox.warning(self, "Model Error", "Predictor model not loaded.")
            return
        
        # Clear previous results
        self.clear_results()
        
        # Disable buttons during prediction
        self.predict_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.load_folder_btn.setEnabled(False)
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(self.image_files))
        self.progress_bar.setValue(0)
        
        # Create worker thread
        self.worker_thread = PredictionWorker(
            self.predictor, 
            self.image_files,
            threshold=0.5
        )
        
        # Connect signals
        self.worker_thread.image_processed.connect(self.on_image_processed)
        self.worker_thread.progress_update.connect(self.on_progress_update)
        self.worker_thread.prediction_complete.connect(self.on_prediction_complete)
        self.worker_thread.finished.connect(self.on_prediction_finished)
        
        # Start thread
        self.worker_thread.start()
        self.statusBar().showMessage("Analyzing images...")
    
    def on_image_processed(self, image_path, predicted_class, confidence):
        """Handle single image processed signal"""
        # Store prediction
        prediction = {
            'image_path': image_path,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
        self.predictions.append(prediction)
        
        # Display result in grid
        self.display_prediction_result(prediction)
    
    def display_prediction_result(self, prediction):
        """Display a single prediction result in the grid"""
        row = len(self.predictions) - 1
        col = 0
        
        # Create container widget
        container = QFrame()
        container.setFrameStyle(QFrame.Box)
        container.setStyleSheet("""
            QFrame {
                background-color: #252526;
                border: 1px solid #3e3e42;
                border-radius: 6px;
                padding: 10px;
                margin: 5px;
            }
        """)
        
        container_layout = QVBoxLayout(container)
        
        # Load and display thumbnail
        try:
            pixmap = QPixmap(prediction['image_path'])
            if not pixmap.isNull():
                # Create thumbnail
                thumbnail = pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                
                image_label = QLabel()
                image_label.setPixmap(thumbnail)
                image_label.setAlignment(Qt.AlignCenter)
                image_label.setToolTip(prediction['image_path'])
                container_layout.addWidget(image_label)
        except:
            pass
        
        # File name
        file_label = QLabel(os.path.basename(prediction['image_path']))
        file_label.setStyleSheet("font-weight: bold; color: #cccccc;")
        file_label.setWordWrap(True)
        container_layout.addWidget(file_label)
        
        # Prediction result with color coding
        result_text = f"Prediction: {prediction['predicted_class']}"
        result_label = QLabel(result_text)
        
        if prediction['predicted_class'] == "Tumor":
            color = "#ff6b6b"  # Red for tumor
            result_label.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 14px;")
        else:
            color = "#51cf66"  # Green for no tumor
            result_label.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 14px;")
        
        container_layout.addWidget(result_label)
        
        # Confidence
        confidence_text = f"Confidence: {prediction['confidence']:.2%}"
        confidence_label = QLabel(confidence_text)
        confidence_label.setStyleSheet("color: #adb5bd;")
        container_layout.addWidget(confidence_label)
        
        # Add to grid (3 columns)
        col = (row % 3)
        row_pos = row // 3
        self.results_grid.addWidget(container, row_pos, col)
    
    def on_progress_update(self, current, total):
        """Handle progress update signal"""
        self.progress_bar.setValue(current)
        self.statusBar().showMessage(f"Processing: {current}/{total} images...")
    
    def on_prediction_complete(self, results):
        """Handle prediction complete signal"""
        self.predictions = results['predictions']
        
        # Update summary
        self.update_summary()
        
        # Enable export button
        self.export_btn.setEnabled(True)
    
    def on_prediction_finished(self):
        """Handle prediction finished signal"""
        # Re-enable buttons
        self.predict_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.load_folder_btn.setEnabled(True)
        
        # Hide progress bar
        self.progress_bar.setVisible(False)
        
        self.statusBar().showMessage(f"Analysis complete: {len(self.predictions)} images processed")
    
    def update_summary(self):
        """Update the summary text box"""
        if not self.predictions:
            self.summary_text.clear()
            return
        
        tumor_count = sum(1 for p in self.predictions if p['predicted_class'] == 'Tumor')
        no_tumor_count = sum(1 for p in self.predictions if p['predicted_class'] == 'No Tumor')
        error_count = sum(1 for p in self.predictions if p.get('error'))
        
        valid_predictions = [p for p in self.predictions if not p.get('error')]
        if valid_predictions:
            avg_confidence = np.mean([p['confidence'] for p in valid_predictions])
        else:
            avg_confidence = 0
        
        summary = f"""📊 ANALYSIS SUMMARY
══════════════════════════════════════

📁 Images Analyzed: {len(self.predictions)}
✅ Successful: {len(valid_predictions)}
❌ Errors: {error_count}

🧠 DETECTION RESULTS:
├── Tumor Detected: {tumor_count}
├── No Tumor: {no_tumor_count}
└── Average Confidence: {avg_confidence:.2%}

📈 STATISTICS:
├── Tumor Rate: {(tumor_count/len(valid_predictions)*100):.1f}%
├── Healthy Rate: {(no_tumor_count/len(valid_predictions)*100):.1f}%
└── Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

💡 Click on any image in the list to view details.
"""
        
        self.summary_text.setPlainText(summary)
    
    def export_results(self):
        """Export results to JSON file"""
        if not self.predictions:
            QMessageBox.warning(self, "No Results", "No prediction results to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Results As", "brain_tumor_predictions.json",
            "JSON Files (*.json);;All Files (*.*)"
        )
        
        if file_path:
            try:
                # Prepare data for export
                export_data = {
                    'model': self.predictor.model_path,
                    'timestamp': datetime.now().isoformat(),
                    'total_images': len(self.predictions),
                    'predictions': self.predictions,
                    'summary': {
                        'tumor_count': sum(1 for p in self.predictions if p['predicted_class'] == 'Tumor'),
                        'no_tumor_count': sum(1 for p in self.predictions if p['predicted_class'] == 'No Tumor'),
                        'error_count': sum(1 for p in self.predictions if p.get('error'))
                    }
                }
                
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                QMessageBox.information(self, "Export Successful", 
                    f"Results exported to:\n{file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", 
                    f"Error exporting results: {str(e)}")

class TkinterGUI:
    """Tkinter based GUI (fallback if PyQt5 not available)"""
    
    def __init__(self, predictor):
        self.predictor = predictor
        self.root = tk.Tk()
        self.root.title("🧠 Brain Tumor Detection - MRI Analyzer")
        self.root.geometry("1000x700")
        
        self.image_files = []
        self.predictions = []
        
        self.init_ui()
        
        # Make window resizable
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
    
    def init_ui(self):
        """Initialize Tkinter UI"""
        # Title
        title_label = tk.Label(
            self.root,
            text="🧠 BRAIN TUMOR DETECTION - MRI ANALYZER",
            font=("Arial", 16, "bold"),
            bg="#1e1e1e",
            fg="white"
        )
        title_label.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(10, 5))
        
        subtitle_label = tk.Label(
            self.root,
            text="Load MRI images to detect brain tumors",
            font=("Arial", 11),
            bg="#1e1e1e",
            fg="#cccccc"
        )
        subtitle_label.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        
        # Configure root background
        self.root.configure(bg="#1e1e1e")
        
        # Create paned window for split view
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=10, pady=(0, 10))
        
        # Left frame - Image selection
        left_frame = ttk.LabelFrame(paned, text="📁 Image Selection", padding=10)
        
        # Buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(
            button_frame,
            text="📂 Load Images",
            command=self.load_images
        ).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(
            button_frame,
            text="📁 Load Folder",
            command=self.load_folder
        ).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(
            button_frame,
            text="🗑️ Clear All",
            command=self.clear_images,
            style="Danger.TButton"
        ).pack(side=tk.LEFT, padx=2)
        
        # Image list with scrollbar
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.image_list = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            bg="#252526",
            fg="#cccccc",
            font=("Consolas", 10),
            selectbackground="#094771"
        )
        self.image_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.image_list.bind('<Double-Button-1>', self.show_image_details)
        
        scrollbar.config(command=self.image_list.yview)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            left_frame,
            variable=self.progress_var,
            maximum=100,
            mode='determinate'
        )
        self.progress_bar.pack(fill=tk.X, pady=(10, 0))
        
        paned.add(left_frame, weight=1)
        
        # Right frame - Results
        right_frame = ttk.LabelFrame(paned, text="📊 Prediction Results", padding=10)
        
        # Control buttons
        control_frame = ttk.Frame(right_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.predict_btn = ttk.Button(
            control_frame,
            text="🔍 Analyze Images",
            command=self.start_prediction,
            style="Success.TButton"
        )
        self.predict_btn.pack(side=tk.LEFT, padx=2)
        self.predict_btn.state(['disabled'])
        
        self.export_btn = ttk.Button(
            control_frame,
            text="💾 Export Results",
            command=self.export_results
        )
        self.export_btn.pack(side=tk.LEFT, padx=2)
        self.export_btn.state(['disabled'])
        
        # Results canvas with scrollbar
        results_canvas_frame = ttk.Frame(right_frame)
        results_canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        results_canvas = tk.Canvas(
            results_canvas_frame,
            bg="#1e1e1e",
            highlightthickness=0
        )
        
        scrollbar = ttk.Scrollbar(
            results_canvas_frame,
            orient=tk.VERTICAL,
            command=results_canvas.yview
        )
        
        self.results_frame = ttk.Frame(results_canvas, style="Results.TFrame")
        
        results_canvas.create_window((0, 0), window=self.results_frame, anchor=tk.NW)
        results_canvas.configure(yscrollcommand=scrollbar.set)
        
        results_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configure scroll region
        def configure_scroll_region(event):
            results_canvas.configure(scrollregion=results_canvas.bbox("all"))
        
        self.results_frame.bind("<Configure>", configure_scroll_region)
        
        # Summary text
        summary_frame = ttk.LabelFrame(right_frame, text="📈 Summary", padding=10)
        summary_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.summary_text = tk.Text(
            summary_frame,
            height=8,
            bg="#252526",
            fg="#cccccc",
            font=("Consolas", 9),
            wrap=tk.WORD,
            relief=tk.FLAT
        )
        self.summary_text.pack(fill=tk.BOTH, expand=True)
        self.summary_text.config(state=tk.DISABLED)
        
        paned.add(right_frame, weight=2)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        status_bar.grid(row=3, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
        
        # Configure styles
        self.configure_styles()
    
    def configure_styles(self):
        """Configure ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure("TLabelFrame", background="#1e1e1e", foreground="#cccccc")
        style.configure("TLabelFrame.Label", background="#1e1e1e", foreground="#cccccc")
        style.configure("TLabelframe", background="#1e1e1e", foreground="#cccccc")
        
        style.configure(
            "TButton",
            background="#007acc",
            foreground="white",
            borderwidth=0,
            focusthickness=0,
            focuscolor="none"
        )
        style.map(
            "TButton",
            background=[("active", "#005a9e"), ("disabled", "#505050")],
            foreground=[("disabled", "#808080")]
        )
        
        style.configure(
            "Danger.TButton",
            background="#d13438"
        )
        style.map(
            "Danger.TButton",
            background=[("active", "#a4262c")]
        )
        
        style.configure(
            "Success.TButton",
            background="#107c10"
        )
        style.map(
            "Success.TButton",
            background=[("active", "#0c5c0c")]
        )
        
        style.configure(
            "Results.TFrame",
            background="#1e1e1e"
        )
    
    def load_images(self):
        """Load images using file dialog"""
        file_paths = filedialog.askopenfilenames(
            title="Select MRI Images",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_paths:
            self.add_image_paths(file_paths)
    
    def load_folder(self):
        """Load all images from a folder"""
        folder_path = filedialog.askdirectory(title="Select Folder with MRI Images")
        
        if folder_path:
            image_paths = []
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                        image_paths.append(os.path.join(root, file))
            
            if image_paths:
                self.add_image_paths(image_paths)
            else:
                messagebox.showwarning("No Images", "No image files found in the selected folder.")
    
    def add_image_paths(self, image_paths):
        """Add image paths to the list"""
        for path in image_paths:
            if path not in self.image_files:
                self.image_files.append(path)
                self.image_list.insert(tk.END, os.path.basename(path))
        
        self.predict_btn.state(['!disabled'])
        self.status_var.set(f"Loaded {len(image_paths)} images. Total: {len(self.image_files)}")
    
    def clear_images(self):
        """Clear all images and results"""
        self.image_files.clear()
        self.image_list.delete(0, tk.END)
        self.clear_results()
        self.predict_btn.state(['disabled'])
        self.export_btn.state(['disabled'])
        self.status_var.set("Cleared all images")
    
    def clear_results(self):
        """Clear prediction results"""
        # Clear results frame
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        self.predictions.clear()
        
        # Clear summary
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.config(state=tk.DISABLED)
    
    def show_image_details(self, event):
        """Show details of selected image"""
        selection = self.image_list.curselection()
        if selection:
            index = selection[0]
            image_path = self.image_files[index]
            
            try:
                # Open image in new window
                image_window = tk.Toplevel(self.root)
                image_window.title(f"Image: {os.path.basename(image_path)}")
                
                img = Image.open(image_path)
                photo = ImageTk.PhotoImage(img)
                
                label = tk.Label(image_window, image=photo)
                label.image = photo  # Keep reference
                label.pack()
                
                info_label = tk.Label(
                    image_window,
                    text=f"Path: {image_path}\nSize: {img.width}x{img.height}",
                    font=("Arial", 10)
                )
                info_label.pack()
                
            except Exception as e:
                messagebox.showerror("Error", f"Cannot display image: {str(e)}")
    
    def start_prediction(self):
        """Start prediction in a separate thread"""
        if not self.image_files:
            messagebox.showwarning("No Images", "Please load images first.")
            return
        
        if not self.predictor.predictor_model:
            messagebox.showwarning("Model Error", "Predictor model not loaded.")
            return
        
        # Clear previous results
        self.clear_results()
        
        # Disable buttons
        self.predict_btn.state(['disabled'])
        self.export_btn.state(['disabled'])
        
        # Show progress
        self.progress_bar['value'] = 0
        
        # Start prediction in thread
        thread = threading.Thread(target=self.run_prediction, daemon=True)
        thread.start()
    
    def run_prediction(self):
        """Run prediction in thread"""
        try:
            total = len(self.image_files)
            
            for i, image_path in enumerate(self.image_files):
                try:
                    # Make prediction
                    prediction = self.predictor.predict_single(image_path)
                    self.predictions.append(prediction)
                    
                    # Update UI in main thread
                    self.root.after(0, self.display_prediction_result, prediction)
                    
                except Exception as e:
                    self.predictions.append({
                        'image_path': image_path,
                        'error': str(e),
                        'predicted_class': 'Error',
                        'confidence': 0.0
                    })
                
                # Update progress
                progress = (i + 1) / total * 100
                self.root.after(0, lambda p=progress: self.progress_bar.config(value=p))
                self.root.after(0, lambda s=f"Processing: {i+1}/{total}": self.status_var.set(s))
            
            # Final update
            self.root.after(0, self.on_prediction_complete)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Prediction failed: {str(e)}"))
            self.root.after(0, self.on_prediction_finished)
    
    def display_prediction_result(self, prediction):
        """Display a single prediction result"""
        # Create frame for this result
        frame = ttk.Frame(self.results_frame, style="Results.TFrame")
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Configure frame
        frame.configure(relief=tk.RAISED, borderwidth=1)
        
        # Try to load thumbnail
        try:
            img = Image.open(prediction['image_path'])
            img.thumbnail((100, 100))
            photo = ImageTk.PhotoImage(img)
            
            img_label = tk.Label(frame, image=photo, bg="#252526")
            img_label.image = photo  # Keep reference
            img_label.pack(side=tk.LEFT, padx=5, pady=5)
        except:
            pass
        
        # Text info
        text_frame = ttk.Frame(frame, style="Results.TFrame")
        text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # File name
        file_label = tk.Label(
            text_frame,
            text=os.path.basename(prediction['image_path']),
            font=("Arial", 10, "bold"),
            bg="#252526",
            fg="#cccccc",
            anchor=tk.W
        )
        file_label.pack(fill=tk.X)
        
        # Prediction with color
        prediction_text = f"Prediction: {prediction['predicted_class']}"
        prediction_label = tk.Label(
            text_frame,
            text=prediction_text,
            font=("Arial", 11, "bold"),
            anchor=tk.W
        )
        
        if prediction['predicted_class'] == "Tumor":
            prediction_label.config(fg="#ff6b6b")  # Red
        else:
            prediction_label.config(fg="#51cf66")  # Green
        
        prediction_label.pack(fill=tk.X)
        
        # Confidence
        confidence_label = tk.Label(
            text_frame,
            text=f"Confidence: {prediction['confidence']:.2%}",
            font=("Arial", 9),
            bg="#252526",
            fg="#adb5bd",
            anchor=tk.W
        )
        confidence_label.pack(fill=tk.X)
    
    def on_prediction_complete(self):
        """Handle prediction completion"""
        self.update_summary()
        self.export_btn.state(['!disabled'])
        self.on_prediction_finished()
    
    def on_prediction_finished(self):
        """Clean up after prediction"""
        self.predict_btn.state(['!disabled'])
        self.progress_bar['value'] = 100
        self.status_var.set(f"Analysis complete: {len(self.predictions)} images processed")
    
    def update_summary(self):
        """Update summary text"""
        if not self.predictions:
            return
        
        tumor_count = sum(1 for p in self.predictions if p['predicted_class'] == 'Tumor')
        no_tumor_count = sum(1 for p in self.predictions if p['predicted_class'] == 'No Tumor')
        error_count = sum(1 for p in self.predictions if p.get('error'))
        
        valid_predictions = [p for p in self.predictions if not p.get('error')]
        if valid_predictions:
            avg_confidence = np.mean([p['confidence'] for p in valid_predictions])
        else:
            avg_confidence = 0
        
        summary = f"""📊 ANALYSIS SUMMARY
══════════════════════════════════════

📁 Images Analyzed: {len(self.predictions)}
✅ Successful: {len(valid_predictions)}
❌ Errors: {error_count}

🧠 DETECTION RESULTS:
├── Tumor Detected: {tumor_count}
├── No Tumor: {no_tumor_count}
└── Average Confidence: {avg_confidence:.2%}

📈 STATISTICS:
├── Tumor Rate: {(tumor_count/len(valid_predictions)*100):.1f}%
├── Healthy Rate: {(no_tumor_count/len(valid_predictions)*100):.1f}%
└── Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(1.0, summary)
        self.summary_text.config(state=tk.DISABLED)
    
    def export_results(self):
        """Export results to JSON file"""
        if not self.predictions:
            messagebox.showwarning("No Results", "No prediction results to export.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Results As",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                export_data = {
                    'model': self.predictor.model_path,
                    'timestamp': datetime.now().isoformat(),
                    'total_images': len(self.predictions),
                    'predictions': self.predictions,
                    'summary': {
                        'tumor_count': sum(1 for p in self.predictions if p['predicted_class'] == 'Tumor'),
                        'no_tumor_count': sum(1 for p in self.predictions if p['predicted_class'] == 'No Tumor'),
                        'error_count': sum(1 for p in self.predictions if p.get('error'))
                    }
                }
                
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                messagebox.showinfo("Export Successful", f"Results exported to:\n{file_path}")
                
            except Exception as e:
                messagebox.showerror("Export Failed", f"Error exporting results: {str(e)}")
    
    def run(self):
        """Run the Tkinter application"""
        self.root.mainloop()

def find_latest_model():
    """Find the latest trained model"""
    models_dir = Path("models")
    if not models_dir.exists():
        return None
    
    # Look for .keras files
    model_files = list(models_dir.glob("*.keras"))
    
    # Also check checkpoints directory
    checkpoint_dir = models_dir / "checkpoints"
    if checkpoint_dir.exists():
        model_files.extend(checkpoint_dir.glob("*.keras"))
    
    if not model_files:
        return None
    
    # Sort by modification time (newest first)
    model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    return str(model_files[0])

def main():
    """Main function"""
    print("\n" + "="*70)
    print("🧠 BRAIN TUMOR DETECTION - GUI PREDICTOR")
    print("="*70)
    
    # Find the latest model
    model_path = find_latest_model()
    
    if not model_path:
        print("❌ No trained model found!")
        print("Please train a model first using: python train.py")
        
        # Ask user to select model
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()
        
        model_path = filedialog.askopenfilename(
            title="Select Trained Model File",
            filetypes=[("Keras models", "*.keras"), ("All files", "*.*")]
        )
        
        if not model_path:
            print("❌ No model selected. Exiting.")
            return
    
    print(f"📂 Using model: {model_path}")
    
    # Initialize predictor
    predictor = BrainTumorPredictorGUI(model_path)
    
    if not predictor.load_model(model_path):
        print("❌ Failed to load model. Exiting.")
        return
    
    # Choose GUI framework
    if QT_AVAILABLE:
        print("🚀 Starting PyQt5 GUI...")
        app = QApplication(sys.argv)
        gui = PyQt5GUI(predictor)
        gui.show()
        sys.exit(app.exec_())
    else:
        print("⚠️ PyQt5 not available, using Tkinter...")
        print("💡 Install PyQt5 for better UI: pip install pyqt5")
        gui = TkinterGUI(predictor)
        gui.run()

if __name__ == "__main__":
    main()