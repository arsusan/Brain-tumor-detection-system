# src/utils.py

import os
import sys
import json
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

from .config import Config


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✅ Random seeds set to: {seed}")


def setup_gpu_memory_growth():
    """Setup GPU memory growth to prevent OOM errors"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"⚠️ Error setting GPU memory growth: {e}")
    else:
        print("⚠️ No GPU available, using CPU")


def create_directory_structure():
    """Create project directory structure for 4-class classification"""
    config = Config()
    directories = [
        config.DATA_DIR,
        config.MODELS_DIR / "checkpoints",
        config.RESULTS_DIR / "plots",
        config.RESULTS_DIR / "metrics",
        config.RESULTS_DIR / "predictions",
        config.LOG_DIR
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    print("✅ Project directory structure created.")


def get_hardware_info():
    """Get hardware information for debugging and metadata"""
    info = {
        'tensorflow_version': tf.__version__,
        'python_version': sys.version.split()[0],
        'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0,
        'cpu_count': os.cpu_count()
    }
    
    print("\n🖥️ Hardware Information:")
    print(f"  TensorFlow Version: {info['tensorflow_version']}")
    print(f"  Python Version:     {info['python_version']}")
    print(f"  GPU Available:      {info['gpu_available']}")
    
    if info['gpu_available']:
        gpu_devices = tf.config.list_physical_devices('GPU')
        print(f"  GPU Devices:        {gpu_devices}")
    
    return info


def plot_multiclass_confusion_matrix(y_true: np.ndarray, 
                                    y_pred: np.ndarray, 
                                    classes: List[str],
                                    save_path: Optional[Path] = None):
    """
    Generates and plots a confusion matrix for the 4 tumor categories.
    """
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix: Brain Tumor Categorization')
    plt.ylabel('Actual Category')
    plt.xlabel('Predicted Category')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to: {save_path}")
    plt.show()


def plot_sample_predictions(images: np.ndarray, 
                           true_labels: np.ndarray, 
                           pred_probs: np.ndarray,
                           classes: List[str],
                           n_samples: int = 10,
                           save_path: Optional[Path] = None):
    """Plot sample MRI images with their Actual vs Predicted 4-class labels"""
    
    n_samples = min(n_samples, len(images))
    rows = (n_samples + 4) // 5
    fig, axes = plt.subplots(rows, 5, figsize=(18, 4 * rows))
    axes = axes.flatten()

    true_indices = np.argmax(true_labels, axis=1) if len(true_labels.shape) > 1 else true_labels
    pred_indices = np.argmax(pred_probs, axis=1)

    for i in range(n_samples):
        axes[i].imshow(images[i])
        
        actual_name = classes[true_indices[i]]
        pred_name = classes[pred_indices[i]]
        confidence = np.max(pred_probs[i])
        
        color = "green" if true_indices[i] == pred_indices[i] else "red"
        
        title = f"Actual: {actual_name}\nPred: {pred_name}\nConf: {confidence:.2f}"
        axes[i].set_title(title, fontsize=10, color=color)
        axes[i].axis('off')
    
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def save_detailed_report(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str], save_path: Path):
    """Save a text-based classification report for all 4 classes"""
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    report = classification_report(y_true, y_pred, target_names=classes)
    with open(save_path, 'w') as f:
        f.write(report)
    print(f"✅ Detailed classification report saved to: {save_path}")
    print("\n" + report)


def calculate_inference_time(model: tf.keras.Model, input_shape: Tuple):
    """Measure how fast the model processes a single MRI scan"""
    dummy_input = np.random.rand(1, *input_shape).astype(np.float32)
    
    for _ in range(10):
        _ = model.predict(dummy_input, verbose=0)
    
    start_time = time.time()
    iterations = 50
    for _ in range(iterations):
        _ = model.predict(dummy_input, verbose=0)
    
    avg_time = (time.time() - start_time) / iterations
    print(f"⏱️ Average Inference Time: {avg_time*1000:.2f} ms per image")
    return avg_time


# Export functions - MUST include get_hardware_info here
__all__ = [
    'set_random_seeds',
    'setup_gpu_memory_growth',
    'create_directory_structure',
    'get_hardware_info',
    'plot_multiclass_confusion_matrix',
    'plot_sample_predictions',
    'save_detailed_report',
    'calculate_inference_time'
]