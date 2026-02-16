# src/utils.py

import os
import sys
import json
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Optional
import tensorflow as tf

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
    """Create project directory structure"""
    config = Config()
    
    directories = [
        config.DATA_DIR,
        config.MODELS_DIR / "checkpoints",
        config.RESULTS_DIR / "plots",
        config.RESULTS_DIR / "metrics",
        config.RESULTS_DIR / "predictions",
        config.NOTEBOOKS_DIR,
        config.LOG_DIR
    ]
    
    created_dirs = []
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        created_dirs.append(str(directory))
    
    print("✅ Project directory structure created:")
    for dir_path in created_dirs:
        print(f"  - {dir_path}")
    
    return created_dirs


def get_hardware_info():
    """Get hardware information for debugging"""
    info = {
        'tensorflow_version': tf.__version__,
        'python_version': sys.version,
        'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0,
        'cpu_count': os.cpu_count()
    }
    
    # Try to get GPU info
    try:
        gpu_devices = tf.config.list_physical_devices('GPU')
        info['gpu_devices'] = [str(device) for device in gpu_devices]
    except:
        info['gpu_devices'] = []
    
    print("\n🖥️ Hardware Information:")
    print(f"  TensorFlow Version: {info['tensorflow_version']}")
    print(f"  Python Version: {info['python_version'].split()[0]}")
    print(f"  CPU Cores: {info['cpu_count']}")
    print(f"  GPU Available: {info['gpu_available']}")
    
    if info['gpu_available']:
        print(f"  GPU Devices: {info['gpu_devices']}")
    
    return info


def save_predictions(predictions: Dict, filename: str = "predictions.json"):
    """Save model predictions to file"""
    predictions_path = Config.RESULTS_DIR / "predictions" / filename
    
    with open(predictions_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"✅ Predictions saved to: {predictions_path}")
    return predictions_path


def load_predictions(filename: str = "predictions.json") -> Dict:
    """Load model predictions from file"""
    predictions_path = Config.RESULTS_DIR / "predictions" / filename
    
    if not predictions_path.exists():
        print(f"⚠️ Predictions file not found: {predictions_path}")
        return {}
    
    with open(predictions_path, 'r') as f:
        predictions = json.load(f)
    
    print(f"✅ Predictions loaded from: {predictions_path}")
    return predictions


def plot_sample_predictions(images: List[np.ndarray], 
                          true_labels: List[int], 
                          pred_labels: List[int], 
                          pred_probs: List[float],
                          save_path: Optional[Path] = None):
    """Plot sample predictions with true and predicted labels"""
    
    n_samples = min(10, len(images))
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(n_samples):
        axes[i].imshow(images[i])
        
        true_text = "Tumor" if true_labels[i] == 1 else "No Tumor"
        pred_text = "Tumor" if pred_labels[i] == 1 else "No Tumor"
        confidence = pred_probs[i] if pred_labels[i] == 1 else 1 - pred_probs[i]
        
        # Color code based on correctness
        color = "green" if true_labels[i] == pred_labels[i] else "red"
        
        title = f"True: {true_text}\nPred: {pred_text}\nConf: {confidence:.3f}"
        axes[i].set_title(title, fontsize=9, color=color)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Sample Predictions', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Sample predictions plot saved to: {save_path}")
    
    plt.show()


def calculate_inference_time(model: tf.keras.Model, 
                           sample_image: np.ndarray, 
                           n_iterations: int = 100) -> Dict:
    """Calculate model inference time"""
    
    # Warmup
    for _ in range(10):
        _ = model.predict(np.expand_dims(sample_image, axis=0), verbose=0)
    
    # Measure inference time
    start_time = time.time()
    for _ in range(n_iterations):
        _ = model.predict(np.expand_dims(sample_image, axis=0), verbose=0)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / n_iterations
    fps = 1 / avg_time
    
    results = {
        'total_time_seconds': total_time,
        'average_time_seconds': avg_time,
        'fps': fps,
        'iterations': n_iterations
    }
    
    print(f"\n⏱️ Inference Time Results ({n_iterations} iterations):")
    print(f"  Total time: {total_time:.4f} seconds")
    print(f"  Average time per image: {avg_time:.4f} seconds")
    print(f"  Frames per second: {fps:.2f}")
    
    return results


def check_dataset_integrity(data_dir: Path) -> Dict:
    """Check dataset integrity and report issues"""
    
    issues = []
    stats = {}
    
    if not data_dir.exists():
        issues.append(f"Dataset directory not found: {data_dir}")
        return {'issues': issues, 'stats': stats}
    
    for set_type in ['Training', 'Testing']:
        set_path = data_dir / set_type
        
        if not set_path.exists():
            issues.append(f"{set_type} directory not found: {set_path}")
            continue
        
        categories = ['glioma', 'meningioma', 'notumor', 'pituitary']
        set_stats = {}
        
        for category in categories:
            cat_path = set_path / category
            
            if not cat_path.exists():
                issues.append(f"Category {category} not found in {set_type}")
                set_stats[category] = 0
                continue
            
            # Count valid image files
            image_files = [f for f in os.listdir(cat_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if len(image_files) == 0:
                issues.append(f"No images found in {cat_path}")
            
            set_stats[category] = len(image_files)
        
        stats[set_type] = set_stats
    
    # Print summary
    print("\n🔍 Dataset Integrity Check:")
    if issues:
        print(f"  ⚠️ Found {len(issues)} issue(s):")
        for issue in issues[:5]:  # Show first 5 issues
            print(f"    - {issue}")
        if len(issues) > 5:
            print(f"    ... and {len(issues) - 5} more issues")
    else:
        print("  ✅ No issues found")
    
    return {'issues': issues, 'stats': stats}


# Export functions
__all__ = [
    'set_random_seeds',
    'setup_gpu_memory_growth',
    'create_directory_structure',
    'get_hardware_info',
    'save_predictions',
    'load_predictions',
    'plot_sample_predictions',
    'calculate_inference_time',
    'check_dataset_integrity'
]