# src/config.py

import os
from pathlib import Path

class Config:
    """Configuration class for the project"""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data" / "brain-tumor-mri-dataset"
    SRC_DIR = PROJECT_ROOT / "src"
    MODELS_DIR = PROJECT_ROOT / "models"
    RESULTS_DIR = PROJECT_ROOT / "results"
    NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
    
    # Create directories
    for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, NOTEBOOKS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Dataset settings
    IMAGE_SIZE = (128, 128)  # Reduced for faster training
    BATCH_SIZE = 16
    VALIDATION_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    # Model settings
    INPUT_SHAPE = (*IMAGE_SIZE, 3)
    NUM_CLASSES = 2  # Binary: Tumor vs No Tumor
    DROPOUT_RATE = 0.4
    LEARNING_RATE = 0.0001
    
    # Training settings
    EPOCHS = 20
    PATIENCE = 6  # Early stopping patience
    MIN_DELTA = 0.001
    
    # Augmentation settings
    AUGMENTATION = {
        'rotation_range': 20,
        'width_shift_range': 0.1,
        'height_shift_range': 0.1,
        'shear_range': 0.1,
        'zoom_range': 0.1,
        'horizontal_flip': True,
        'vertical_flip': True,
        'fill_mode': 'nearest'
    }
    
    # Evaluation settings
    THRESHOLD = 0.5  # Binary classification threshold
    METRICS = ['accuracy', 'precision', 'recall', 'auc']
    
    # Logging
    LOG_DIR = PROJECT_ROOT / "logs"
    LOG_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def setup_paths(cls):
        """Ensure all required directories exist"""
        directories = [
            cls.MODELS_DIR / "checkpoints",
            cls.RESULTS_DIR / "plots",
            cls.RESULTS_DIR / "metrics",
            cls.RESULTS_DIR / "predictions"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print("✅ Project directories created successfully")

# Initialize paths
Config.setup_paths()

# Export configuration
__all__ = ['Config']