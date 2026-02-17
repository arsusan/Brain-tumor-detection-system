# src/model.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Optional, Dict

from .config import Config


class BrainTumorModel:
    """Brain tumor classification model builder (4-Class Multiclass Version)"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
    
    def build_cnn(self, input_shape: tuple = None, dropout_rate: float = None) -> keras.Model:
        """Build a custom CNN model from scratch for 4-class classification"""
        
        if input_shape is None:
            input_shape = self.config.INPUT_SHAPE
        if dropout_rate is None:
            dropout_rate = self.config.DROPOUT_RATE
        
        print(f"🔧 Building 4-Class CNN with input shape: {input_shape}")
        
        model = models.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(32, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout_rate * 0.5),
            
            # Block 2
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout_rate),
            
            # Block 3
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout_rate),
            
            # Block 4
            layers.Conv2D(256, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(dropout_rate * 0.8),
            
            # Dense layers
            layers.Dense(128, kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(dropout_rate * 0.6),
            
            # Output layer: Changed to 4 units with Softmax for Multiclass
            layers.Dense(self.config.NUM_CLASSES, activation='softmax')
        ], name="BrainTumorCNN_4Class")
        
        self.model = model
        return model
    
    def build_efficient_cnn(self, input_shape: tuple = None) -> keras.Model:
        """Build a more efficient CNN using depthwise separable convolutions for 4-class classification"""
        
        if input_shape is None:
            input_shape = self.config.INPUT_SHAPE
        
        print(f"🔧 Building Efficient 4-Class CNN with input shape: {input_shape}")
        
        inputs = keras.Input(shape=input_shape)
        
        # Initial conv layer
        x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Depthwise separable convolutions
        x = layers.SeparableConv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.SeparableConv2D(128, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(128, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.SeparableConv2D(256, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        
        # Dense layers
        x = layers.Dense(128, kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output: Changed to 4 units with Softmax for Multiclass
        outputs = layers.Dense(self.config.NUM_CLASSES, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name="EfficientBrainTumorCNN_4Class")
        self.model = model
        return model
    
    def compile_model(self, model: Optional[keras.Model] = None, 
                       learning_rate: float = None) -> keras.Model:
        """Compile the model with Categorical Crossentropy for 4-class classification"""
        
        if model is None:
            model = self.model
        if learning_rate is None:
            learning_rate = self.config.LEARNING_RATE
        
        # Choose optimizer
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        # Compile model with categorical_crossentropy
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        print("✅ Model compiled successfully for 4-class classification")
        print(f"  Optimizer: Adam (lr={learning_rate})")
        print(f"  Loss: Categorical Crossentropy")
        
        return model
    
    def get_model_summary(self, model: Optional[keras.Model] = None):
        """Get detailed model summary"""
        if model is None:
            model = self.model
        
        model.summary()
        
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
        
        print(f"\n📊 Model Statistics:")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Non-trainable parameters: {non_trainable_params:,}")
        print(f"  Total parameters: {trainable_params + non_trainable_params:,}")
        print(f"  Model size: {(trainable_params + non_trainable_params) * 4 / 1024 / 1024:.2f} MB")
    
    def save_model(self, model: keras.Model, filepath: str):
        """Save model to file"""
        model.save(filepath)
        print(f"✅ Model saved to: {filepath}")
    
    def load_model(self, filepath: str) -> keras.Model:
        """Load model from file"""
        model = keras.models.load_model(filepath)
        self.model = model
        print(f"✅ Model loaded from: {filepath}")
        return model


# Export class
__all__ = ['BrainTumorModel']