# src/preprocessing.py

import cv2
import numpy as np
from PIL import Image
import random
from typing import List, Tuple, Union
import tensorflow as tf

from .config import Config


class ImagePreprocessor:
    """Image preprocessing and augmentation class (Multiclass Compatible)"""
    
    def __init__(self, config: Config):
        self.config = config
        self.image_size = config.IMAGE_SIZE
        self.cache = {}  # For caching processed images
        
    def load_image(self, image_path: str) -> np.ndarray:
        """Load image from file path using OpenCV with PIL fallback"""
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
        """Preprocess a single image: Resize, Normalize, and optionally Augment"""
        
        # Check cache first (only for non-augmented validation/test images)
        if image_path in self.cache and not augment:
            return self.cache[image_path]
        
        img = self.load_image(image_path)
        if img is None:
            # Return blank image if loading fails
            return np.zeros((*self.image_size, 3), dtype=np.float32)
        
        # 1. Resize
        img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)
        
        # 2. Normalize (Rescale pixel values to [0, 1])
        img = img.astype(np.float32) / 255.0
        
        # 3. Apply augmentation if requested (Training only)
        if augment:
            img = self._apply_augmentation(img)
        
        # Cache the result for validation/testing to speed up epochs
        if not augment:
            self.cache[image_path] = img
        
        return img
    
    def _apply_augmentation(self, img: np.ndarray) -> np.ndarray:
        """Apply custom data augmentation to increase model generalization"""
        
        # Random horizontal flip
        if random.random() > 0.5:
            img = np.fliplr(img)
        
        # Random vertical flip
        if random.random() > 0.5:
            img = np.flipud(img)
        
        # Random rotation (-20 to 20 degrees)
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
    
    def process_batch(self, image_paths: List[str], labels: Union[List, np.ndarray], 
                      augment: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Process a batch of images and their corresponding labels"""
        
        images = []
        for img_path in image_paths:
            img = self.preprocess_single(img_path, augment=augment)
            images.append(img)
        
        return np.array(images), np.array(labels)
    
    def prepare_tf_dataset(self, image_paths: List[str], labels: np.ndarray, 
                          batch_size: int, augment: bool = False, 
                          shuffle: bool = True) -> tf.data.Dataset:
        """Create a high-performance TensorFlow Dataset"""
        
        # Create dataset from paths and one-hot labels
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        
        def preprocess_map(img_path, label):
            # Convert TF tensor to string
            img_path_str = img_path.numpy().decode('utf-8')
            
            # Preprocess
            img = self.preprocess_single(img_path_str, augment=augment)
            
            # Return as tensors with explicit shapes
            return (tf.convert_to_tensor(img, dtype=tf.float32), 
                    tf.convert_to_tensor(label, dtype=tf.float32))
        
        # Wrap Python function for TF pipeline
        dataset = dataset.map(
            lambda x, y: tf.py_function(
                func=preprocess_map,
                inp=[x, y],
                Tout=[tf.float32, tf.float32]
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Set shapes (py_function loses shape information)
        dataset = dataset.map(lambda x, y: (
            tf.ensure_shape(x, (*self.image_size, 3)),
            tf.ensure_shape(y, (self.config.NUM_CLASSES,))
        ))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(len(image_paths), 1000))
        
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def clear_cache(self):
        """Clear the image cache to free up memory"""
        self.cache.clear()
        print("✅ Image cache cleared")


# Export class
__all__ = ['ImagePreprocessor']