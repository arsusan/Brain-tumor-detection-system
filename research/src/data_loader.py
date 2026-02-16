# src/data_loader.py

import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
from tqdm import tqdm

from .config import Config


class BrainTumorDataLoader:
    """Data loader for brain tumor MRI dataset"""
    
    def __init__(self, config: Config):
        self.config = config
        self.processor = None  # Will be set by preprocessing module
        
    def load_dataset_stats(self) -> Dict[str, Dict]:
        """Load and display dataset statistics"""
        
        stats = {}
        categories = ['glioma', 'meningioma', 'notumor', 'pituitary']
        
        print("\n📊 Dataset Statistics:")
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
    
    def load_binary_dataset(self, set_type: str = 'Training') -> Tuple[List[str], List[int]]:
        """
        Load dataset for binary classification
        Returns: (image_paths, labels) where 1=Tumor, 0=No Tumor
        """
        
        set_path = self.config.DATA_DIR / set_type
        if not set_path.exists():
            raise FileNotFoundError(f"Dataset not found at {set_path}")
        
        image_paths = []
        labels = []
        
        # Tumor categories (label = 1)
        tumor_categories = ['glioma', 'meningioma', 'pituitary']
        
        print(f"\n📂 Loading {set_type} dataset...")
        
        # Load tumor images
        for tumor_cat in tqdm(tumor_categories, desc="Loading tumor images"):
            cat_path = set_path / tumor_cat
            if not cat_path.exists():
                print(f"  ⚠️ Missing: {cat_path}")
                continue
            
            image_files = [f for f in os.listdir(cat_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_file in image_files:
                image_paths.append(str(cat_path / img_file))
                labels.append(1)  # Tumor
        
        # Load no tumor images (label = 0)
        no_tumor_path = set_path / 'notumor'
        if no_tumor_path.exists():
            image_files = [f for f in os.listdir(no_tumor_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_file in tqdm(image_files, desc="Loading no tumor images"):
                image_paths.append(str(no_tumor_path / img_file))
                labels.append(0)  # No Tumor
        else:
            print(f"  ⚠️ Missing: {no_tumor_path}")
        
        # Shuffle the dataset
        combined = list(zip(image_paths, labels))
        random.shuffle(combined)
        image_paths, labels = zip(*combined)
        
        print(f"✅ Loaded {len(image_paths)} images from {set_type}")
        print(f"  Tumor: {sum(labels)}")
        print(f"  No Tumor: {len(labels) - sum(labels)}")
        
        return list(image_paths), list(labels)
    
    def create_train_val_split(self, image_paths: List[str], labels: List[int], 
                              val_split: float = 0.15) -> Tuple:
        """Split data into training and validation sets"""
        
        from sklearn.model_selection import train_test_split
        
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, 
            test_size=val_split, 
            stratify=labels,
            random_state=42
        )
        
        print(f"\n📊 Train/Validation Split:")
        print(f"  Training samples: {len(train_paths)}")
        print(f"  Validation samples: {len(val_paths)}")
        print(f"  Training Tumor/No Tumor: {sum(train_labels)}/{len(train_labels)-sum(train_labels)}")
        print(f"  Validation Tumor/No Tumor: {sum(val_labels)}/{len(val_labels)-sum(val_labels)}")
        
        return train_paths, val_paths, train_labels, val_labels
    
    def create_data_generator(self, image_paths: List[str], labels: List[int], 
                             batch_size: int, augment: bool = False, 
                             shuffle: bool = True):
        """Create a data generator for efficient memory usage"""
        
        import tensorflow as tf
        from .preprocessing import ImagePreprocessor
        
        # Initialize processor if not already done
        if self.processor is None:
            self.processor = ImagePreprocessor(self.config)
        
        def generator():
            """Generator function for streaming data"""
            num_samples = len(image_paths)
            indices = list(range(num_samples))
            
            while True:
                if shuffle:
                    random.shuffle(indices)
                
                for start_idx in range(0, num_samples, batch_size):
                    batch_indices = indices[start_idx:start_idx + batch_size]
                    
                    if len(batch_indices) == 0:
                        continue
                    
                    # Get batch paths and labels
                    batch_paths = [image_paths[i] for i in batch_indices]
                    batch_labels = [labels[i] for i in batch_indices]
                    
                    # Process batch
                    batch_images, batch_labels_arr = self.processor.process_batch(
                        batch_paths, batch_labels, augment=augment
                    )
                    
                    yield batch_images, batch_labels_arr
        
        return generator()
    
    def get_class_distribution(self, labels: List[int]) -> Dict[str, float]:
        """Calculate class distribution"""
        total = len(labels)
        tumor_count = sum(labels)
        no_tumor_count = total - tumor_count
        
        return {
            'tumor': tumor_count,
            'no_tumor': no_tumor_count,
            'tumor_percentage': tumor_count / total * 100,
            'no_tumor_percentage': no_tumor_count / total * 100
        }


# Export class
__all__ = ['BrainTumorDataLoader']