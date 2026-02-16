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
    """Model training and evaluation class"""
    
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
        """Prepare train, validation, and test data"""
        
        print("\n" + "="*60)
        print("📊 PREPARING DATA")
        print("="*60)
        
        # Load dataset statistics
        self.data_loader.load_dataset_stats()
        
        # Load training data
        train_paths, train_labels = self.data_loader.load_binary_dataset('Training')
        
        # Load test data
        test_paths, test_labels = self.data_loader.load_binary_dataset('Testing')
        
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
            ),
            
            # TensorBoard (optional)
            # keras.callbacks.TensorBoard(
            #     log_dir=str(self.config.LOG_DIR / f"{model_name}_{timestamp}"),
            #     histogram_freq=1,
            #     write_graph=True
            # )
        ]
        
        print("✅ Callbacks created:")
        print(f"  - Model checkpoint (loss & accuracy)")
        print(f"  - Early stopping (patience={self.config.PATIENCE})")
        print(f"  - ReduceLROnPlateau")
        print(f"  - CSV Logger")
        
        return callbacks
    
    def train_model(self, model_type: str = 'cnn', 
                   epochs: Optional[int] = None) -> Dict:
        """Train the model"""
        
        if epochs is None:
            epochs = self.config.EPOCHS
        
        print("\n" + "="*60)
        print("🚀 STARTING TRAINING")
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
        
        print(f"\n⚙️ Training Configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {self.config.BATCH_SIZE}")
        print(f"  Train steps per epoch: {train_steps}")
        print(f"  Validation steps: {val_steps}")
        
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
        
        print(f"\n✅ Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        
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
        """Save training metadata"""
        
        metadata = {
            'training_date': datetime.now().isoformat(),
            'training_time_seconds': training_time,
            'training_time_minutes': training_time / 60,
            'model_type': 'binary_classification',
            'input_shape': model.input_shape[1:],
            'parameters': {
                'trainable': int(sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])),
                'non_trainable': int(sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])),
                'total': int(model.count_params())
            },
            'config': {
                'image_size': self.config.IMAGE_SIZE,
                'batch_size': self.config.BATCH_SIZE,
                'learning_rate': self.config.LEARNING_RATE,
                'dropout_rate': self.config.DROPOUT_RATE,
                'epochs': self.config.EPOCHS
            }
        }
        
        metadata_path = self.config.RESULTS_DIR / "training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Training metadata saved to: {metadata_path}")
    
    def plot_training_history(self, save_plots: bool = True):
        """Plot training history"""
        
        if self.history is None:
            print("⚠️ No training history available. Train model first.")
            return
        
        history = self.history
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot loss
        axes[0, 0].plot(history['loss'], label='Training Loss', linewidth=2)
        axes[0, 0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=12)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot accuracy
        axes[0, 1].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0, 1].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_title('Training and Validation Accuracy', fontsize=12)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot precision
        axes[0, 2].plot(history['precision'], label='Training Precision', linewidth=2)
        axes[0, 2].plot(history['val_precision'], label='Validation Precision', linewidth=2)
        axes[0, 2].set_title('Training and Validation Precision', fontsize=12)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot recall
        axes[1, 0].plot(history['recall'], label='Training Recall', linewidth=2)
        axes[1, 0].plot(history['val_recall'], label='Validation Recall', linewidth=2)
        axes[1, 0].set_title('Training and Validation Recall', fontsize=12)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot AUC
        axes[1, 1].plot(history['auc'], label='Training AUC', linewidth=2)
        axes[1, 1].plot(history['val_auc'], label='Validation AUC', linewidth=2)
        axes[1, 1].set_title('Training and Validation AUC', fontsize=12)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot learning rate if available
        if 'lr' in history:
            axes[1, 2].plot(history['lr'], label='Learning Rate', linewidth=2, color='purple')
            axes[1, 2].set_title('Learning Rate Schedule', fontsize=12)
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Learning Rate')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].axis('off')
        
        plt.suptitle('Training History', fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Save plot
        if save_plots:
            plot_path = self.config.RESULTS_DIR / "plots" / "training_history.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✅ Training history plot saved to: {plot_path}")
        
        plt.show()
        
        # Print summary statistics
        print("\n📈 Training Summary:")
        print(f"  Final training loss: {history['loss'][-1]:.4f}")
        print(f"  Final validation loss: {history['val_loss'][-1]:.4f}")
        print(f"  Best validation loss: {min(history['val_loss']):.4f}")
        print(f"  Final training accuracy: {history['accuracy'][-1]:.4f}")
        print(f"  Final validation accuracy: {history['val_accuracy'][-1]:.4f}")
        print(f"  Best validation accuracy: {max(history['val_accuracy']):.4f}")
        print(f"  Training time: {self.training_time:.2f}s ({self.training_time/60:.2f} minutes)")


# Export class
__all__ = ['ModelTrainer']