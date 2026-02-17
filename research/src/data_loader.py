# src/data_loader.py

import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from .config import Config


class BrainTumorDataLoader:
    """Data loader for 4-class brain tumor MRI dataset (Glioma, Meningioma, Pituitary, No Tumor)"""
    
    def __init__(self, config: Config):
        self.config = config
        self.processor = None  # Will be set by trainer.py
        
    def load_dataset_stats(self) -> Dict[str, Dict]:
        """Load and display dataset statistics for all 4 categories"""
        
        stats = {}
        # Uses classes from config: ['glioma', 'meningioma', 'notumor', 'pituitary']
        categories = self.config.CLASSES
        
        print("\n📊 Dataset Statistics (4-Class):")
        print("=" * 60)
        
        for set_type in ['Training', 'Testing']:
            set_stats = {}
            set_path = self.config.DATA_DIR / set_type
            
            if not set_path.exists():
                print(f"⚠️ {set_type} directory not found at {set_path}")
                continue
            
            total_images = 0
            for category in categories:
                category_path = set_path / category
                if category_path.exists():
                    image_count = len([f for f in os.listdir(category_path) 
                                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    set_stats[category] = image_count
                    total_images += image_count
                else:
                    set_stats[category] = 0
            
            set_stats['total'] = total_images
            stats[set_type] = set_stats
            
            # Print statistics
            print(f"\n{set_type}:")
            for category in categories:
                count = set_stats[category]
                percentage = (count / total_images * 100) if total_images > 0 else 0
                print(f"  {category:12s}: {count:4d} images ({percentage:5.1f}%)")
            print(f"  {'Total':12s}: {total_images:4d} images")
        
        return stats
    
    def load_categorical_dataset(self, set_type: str = 'Training') -> Tuple[List[str], np.ndarray]:
        """
        Load dataset for 4-class categorical classification
        Returns: (image_paths, one_hot_labels)
        """
        
        set_path = self.config.DATA_DIR / set_type
        if not set_path.exists():
            raise FileNotFoundError(f"Dataset folder not found at {set_path}")
        
        image_paths = []
        labels = []
        
        # Mapping class names to integers based on Config order
        class_map = {label: i for i, label in enumerate(self.config.CLASSES)}
        
        print(f"\n📂 Loading {set_type} dataset (4 Classes)...")
        
        for category in self.config.CLASSES:
            cat_path = set_path / category
            if not cat_path.exists():
                print(f"  ⚠️ Missing Folder: {cat_path}")
                continue
            
            image_files = [f for f in os.listdir(cat_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_file in tqdm(image_files, desc=f"Loading {category}"):
                image_paths.append(str(cat_path / img_file))
                labels.append(class_map[category])
        
        # Convert to numpy arrays
        image_paths = np.array(image_paths)
        labels = np.array(labels)
        
        # Shuffle paths and labels together
        indices = np.arange(len(image_paths))
        np.random.shuffle(indices)
        image_paths = image_paths[indices]
        labels = labels[indices]
        
        # --- CRITICAL: Convert integer labels to One-Hot Encoding ---
        # Example: 2 (notumor) becomes [0, 0, 1, 0]
        one_hot_labels = to_categorical(labels, num_classes=self.config.NUM_CLASSES)
        
        print(f"✅ Loaded {len(image_paths)} images from {set_type}")
        return image_paths.tolist(), one_hot_labels
    
    def create_train_val_split(self, image_paths: List[str], labels: np.ndarray, 
                               val_split: float = 0.15) -> Tuple:
        """Split data into training and validation sets using stratification"""
        
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, 
            test_size=val_split, 
            stratify=labels, # Ensures 4-class balance in both sets
            random_state=42
        )
        
        print(f"\n📊 Multiclass Train/Validation Split:")
        print(f"  Training samples: {len(train_paths)}")
        print(f"  Validation samples: {len(val_paths)}")
        
        return train_paths, val_paths, train_labels, val_labels
    
    def create_data_generator(self, image_paths: List[str], labels: np.ndarray, 
                               batch_size: int, augment: bool = False, 
                               shuffle: bool = True):
        """Create a data generator for categorical training"""
        
        import tensorflow as tf
        
        # Ensure processor is attached
        if self.processor is None:
            from .preprocessing import ImagePreprocessor
            self.processor = ImagePreprocessor(self.config)
        
        def generator():
            """Generator function for streaming multiclass data"""
            num_samples = len(image_paths)
            indices = list(range(num_samples))
            
            while True:
                if shuffle:
                    random.shuffle(indices)
                
                for start_idx in range(0, num_samples, batch_size):
                    batch_indices = indices[start_idx:start_idx + batch_size]
                    
                    if len(batch_indices) == 0:
                        continue
                    
                    batch_paths = [image_paths[i] for i in batch_indices]
                    batch_labels = [labels[i] for i in batch_indices]
                    
                    # Process batch using the preprocessor
                    # Labels are already one-hot encoded from load_categorical_dataset
                    batch_images, batch_labels_arr = self.processor.process_batch(
                        batch_paths, batch_labels, augment=augment
                    )
                    
                    yield batch_images, batch_labels_arr
        
        # Create TF Dataset from generator
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(None, *self.config.INPUT_SHAPE), dtype=tf.float32),
                tf.TensorSpec(shape=(None, self.config.NUM_CLASSES), dtype=tf.float32)
            )
        )
        
        return dataset.prefetch(tf.data.AUTOTUNE)

    def get_class_distribution(self, labels: np.ndarray) -> Dict[str, int]:
        """Calculate distribution across the 4 classes"""
        # labels are one-hot, so we sum along axis 0
        counts = np.sum(labels, axis=0)
        dist = {self.config.CLASSES[i]: int(counts[i]) for i in range(len(counts))}
        return dist


# Export class
__all__ = ['BrainTumorDataLoader']