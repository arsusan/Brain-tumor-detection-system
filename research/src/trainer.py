# src/trainer.py

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import tensorflow as tf
from tensorflow import keras

from .config import Config
from .model import BrainTumorModel
from .data_loader import BrainTumorDataLoader
from .preprocessing import ImagePreprocessor


class ModelTrainer:
    """Model training and evaluation class (4-Class Version)"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model_builder = BrainTumorModel(config)
        self.data_loader = BrainTumorDataLoader(config)
        self.preprocessor = ImagePreprocessor(config)
        
        # Set data_loader's processor
        self.data_loader.processor = self.preprocessor
        
        # Training history
        self.history = None
        self.training_time = None
        
    def prepare_data(self):
        """Prepare train, validation, and test data for 4-class classification"""
        
        print("\n" + "="*60)
        print("📊 PREPARING 4-CLASS CATEGORICAL DATA")
        print("="*60)
        
        # Load dataset statistics
        self.data_loader.load_dataset_stats()
        
        # --- UPDATED: Load categorical (4-class) instead of binary ---
        train_paths, train_labels = self.data_loader.load_categorical_dataset('Training')
        
        # Load test data
        test_paths, test_labels = self.data_loader.load_categorical_dataset('Testing')
        
        # Split training into train and validation
        train_paths, val_paths, train_labels, val_labels = self.data_loader.create_train_val_split(
            train_paths, train_labels, val_split=self.config.VALIDATION_SPLIT
        )
        
        print(f"\n✅ Data preparation complete:")
        print(f"  Training: {len(train_paths)} samples")
        print(f"  Validation: {len(val_paths)} samples")
        print(f"  Testing: {len(test_paths)} samples")
        
        return (train_paths, train_labels, val_paths, val_labels, 
                test_paths, test_labels)
    
    def create_callbacks(self, model_name: str = "brain_tumor_cnn") -> List[keras.callbacks.Callback]:
        """Create training callbacks"""
        
        callbacks_dir = self.config.MODELS_DIR / "checkpoints"
        callbacks_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks = [
            # Model checkpoint - best validation loss
            keras.callbacks.ModelCheckpoint(
                filepath=str(callbacks_dir / f"{model_name}_best_loss_{timestamp}.keras"),
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            ),
            
            # Model checkpoint - best validation accuracy
            keras.callbacks.ModelCheckpoint(
                filepath=str(callbacks_dir / f"{model_name}_best_accuracy_{timestamp}.keras"),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.PATIENCE,
                restore_best_weights=True,
                verbose=1,
                min_delta=self.config.MIN_DELTA
            ),
            
            # Reduce learning rate on plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # CSV logger
            keras.callbacks.CSVLogger(
                filename=str(self.config.RESULTS_DIR / f"training_log_{timestamp}.csv"),
                separator=',',
                append=True
            )
        ]
        
        print("✅ Callbacks created.")
        return callbacks
    
    def train_model(self, model_type: str = 'cnn', 
                    epochs: Optional[int] = None) -> Dict:
        """Train the model"""
        
        if epochs is None:
            epochs = self.config.EPOCHS
        
        print("\n" + "="*60)
        print("🚀 STARTING MULTICLASS TRAINING")
        print("="*60)
        
        # Prepare data
        (train_paths, train_labels, val_paths, val_labels, 
         test_paths, test_labels) = self.prepare_data()
        
        # Build model
        if model_type.lower() == 'efficient':
            model = self.model_builder.build_efficient_cnn()
        else:
            model = self.model_builder.build_cnn()
        
        # Compile model
        model = self.model_builder.compile_model(model)
        
        # Display model summary
        self.model_builder.get_model_summary(model)
        
        # Create data generators
        train_gen = self.data_loader.create_data_generator(
            train_paths, train_labels,
            batch_size=self.config.BATCH_SIZE,
            augment=True,
            shuffle=True
        )
        
        val_gen = self.data_loader.create_data_generator(
            val_paths, val_labels,
            batch_size=self.config.BATCH_SIZE,
            augment=False,
            shuffle=False
        )
        
        # Calculate steps
        train_steps = len(train_paths) // self.config.BATCH_SIZE
        val_steps = len(val_paths) // self.config.BATCH_SIZE
        
        # Create callbacks
        callbacks = self.create_callbacks(model_type)
        
        # Start training
        print("\n⏳ Training started...")
        start_time = time.time()
        
        history = model.fit(
            train_gen,
            steps_per_epoch=train_steps,
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=val_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        # Save training history
        self.history = history.history
        self.training_time = training_time
        
        print(f"\n✅ Training completed in {training_time/60:.2f} minutes")
        
        # Save the final model
        final_model_path = self.config.MODELS_DIR / f"final_model_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
        self.model_builder.save_model(model, str(final_model_path))
        
        # Save training metadata
        self.save_training_metadata(model, training_time)
        
        return {
            'model': model,
            'history': history.history,
            'training_time': training_time,
            'model_path': final_model_path,
            'test_data': (test_paths, test_labels)
        }
    
    def save_training_metadata(self, model: keras.Model, training_time: float):
        """Save training metadata for the 4-class project"""
        
        metadata = {
            'training_date': datetime.now().isoformat(),
            'training_time_minutes': training_time / 60,
            'model_type': 'multiclass_classification',
            'num_classes': self.config.NUM_CLASSES,
            'classes': self.config.CLASSES,
            'input_shape': model.input_shape[1:],
            'parameters': {
                'total': int(model.count_params())
            },
            'config': {
                'batch_size': self.config.BATCH_SIZE,
                'learning_rate': self.config.LEARNING_RATE,
                'epochs': self.config.EPOCHS
            }
        }
        
        metadata_path = self.config.RESULTS_DIR / "training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Training metadata saved to: {metadata_path}")
    
    def plot_training_history(self, save_plots: bool = True):
        """Plot training history using original 2x3 grid layout"""
        
        if self.history is None:
            print("⚠️ No training history available.")
            return
        
        history = self.history
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot Logic
        metrics_to_plot = [
            ('loss', 'Loss', axes[0, 0]),
            ('accuracy', 'Accuracy', axes[0, 1]),
            ('precision', 'Precision', axes[0, 2]),
            ('recall', 'Recall', axes[1, 0]),
            ('auc', 'AUC', axes[1, 1])
        ]
        
        for key, name, ax in metrics_to_plot:
            if key in history:
                ax.plot(history[key], label=f'Train {name}', linewidth=2)
                ax.plot(history[f'val_{key}'], label=f'Val {name}', linewidth=2)
                ax.set_title(f'Training and Validation {name}')
                ax.set_xlabel('Epoch')
                ax.set_ylabel(name)
                ax.legend()
                ax.grid(True, alpha=0.3)

        # Plot learning rate
        if 'lr' in history:
            axes[1, 2].plot(history['lr'], label='Learning Rate', color='purple')
            axes[1, 2].set_title('Learning Rate Schedule')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].axis('off')
        
        plt.suptitle('4-Class Model Training History', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.config.RESULTS_DIR / "plots" / "training_history.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✅ Training history plot saved to: {plot_path}")
        
        plt.show()

# Export class
__all__ = ['ModelTrainer']