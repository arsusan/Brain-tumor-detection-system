# src/preprocessing.py

import cv2
import numpy as np
from PIL import Image
import random
from typing import List, Tuple
from pathlib import Path
import tensorflow as tf
from tqdm import tqdm

from .config import Config


class ImagePreprocessor:
    """Image preprocessing and augmentation class"""
    
    def __init__(self, config: Config):
        self.config = config
        self.image_size = config.IMAGE_SIZE
        self.cache = {}  # For caching processed images
        
    def load_image(self, image_path: str) -> np.ndarray:
        """Load image from file path"""
        try:
            # Try OpenCV first (faster)
            img = cv2.imread(image_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img
            
            # Fallback to PIL
            img = Image.open(image_path).convert('RGB')
            return np.array(img)
            
        except Exception as e:
            print(f"Warning: Error loading {image_path}: {e}")
            return None
    
    def preprocess_single(self, image_path: str, augment: bool = False) -> np.ndarray:
        """Preprocess a single image"""
        
        # Check cache first
        if image_path in self.cache and not augment:
            return self.cache[image_path]
        
        img = self.load_image(image_path)
        if img is None:
            # Return blank image if loading fails
            return np.zeros((*self.image_size, 3), dtype=np.float32)
        
        # Resize
        img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)
        
        # Convert to float and normalize
        img = img.astype(np.float32) / 255.0
        
        # Apply augmentation if requested
        if augment:
            img = self._apply_augmentation(img)
        
        # Cache the result
        if not augment:
            self.cache[image_path] = img
        
        return img
    
    def _apply_augmentation(self, img: np.ndarray) -> np.ndarray:
        """Apply data augmentation to image"""
        
        # Random horizontal flip
        if random.random() > 0.5:
            img = np.fliplr(img)
        
        # Random vertical flip
        if random.random() > 0.5:
            img = np.flipud(img)
        
        # Random rotation
        if random.random() > 0.5:
            angle = random.uniform(-20, 20)
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Random brightness adjustment
        if random.random() > 0.5:
            brightness = random.uniform(0.8, 1.2)
            img = np.clip(img * brightness, 0, 1)
        
        # Random contrast adjustment
        if random.random() > 0.5:
            contrast = random.uniform(0.8, 1.2)
            mean = np.mean(img)
            img = np.clip((img - mean) * contrast + mean, 0, 1)
        
        return img
    
    def process_batch(self, image_paths: List[str], labels: List[int], 
                     augment: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Process a batch of images"""
        
        images = []
        valid_labels = []
        
        for img_path, label in zip(image_paths, labels):
            img = self.preprocess_single(img_path, augment=augment)
            images.append(img)
            valid_labels.append(label)
        
        return np.array(images), np.array(valid_labels)
    
    def prepare_tf_dataset(self, image_paths: List[str], labels: List[int], 
                          batch_size: int, augment: bool = False, 
                          shuffle: bool = True) -> tf.data.Dataset:
        """Create TensorFlow Dataset for efficient training"""
        
        # Create dataset from paths and labels
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        
        # Map preprocessing function
        def preprocess_map(img_path, label):
            # Convert TensorFlow tensor to string
            img_path_str = img_path.numpy().decode('utf-8')
            
            # Load and preprocess image
            img = self.preprocess_single(img_path_str, augment=augment)
            
            return tf.convert_to_tensor(img, dtype=tf.float32), tf.convert_to_tensor(label, dtype=tf.float32)
        
        # Use py_function to wrap the Python preprocessing
        dataset = dataset.map(
            lambda x, y: tf.py_function(
                func=preprocess_map,
                inp=[x, y],
                Tout=[tf.float32, tf.float32]
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Dataset optimizations
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def clear_cache(self):
        """Clear the image cache"""
        self.cache.clear()
        print("✅ Image cache cleared")


# Export class
__all__ = ['ImagePreprocessor']