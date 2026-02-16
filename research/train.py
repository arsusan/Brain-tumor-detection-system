#!/usr/bin/env python3
"""
Brain Tumor Detection - Training Script
Train a custom CNN for binary classification of brain tumors from MRI images
"""

import os
import sys
import argparse
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.utils import (
    set_random_seeds, 
    setup_gpu_memory_growth, 
    create_directory_structure,
    get_hardware_info
)
from src.trainer import ModelTrainer


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Brain Tumor Detection Model')
    
    parser.add_argument('--model', type=str, default='cnn',
                       choices=['cnn', 'efficient'],
                       help='Model architecture to use (default: cnn)')
    
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size for training')
    
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate for optimizer')
    
    parser.add_argument('--image-size', type=int, nargs=2, default=None,
                       help='Image size as two integers (height width)')
    
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Disable data augmentation')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    parser.add_argument('--verbose', type=int, default=1,
                       choices=[0, 1, 2],
                       help='Verbosity level (0=silent, 1=normal, 2=verbose)')
    
    return parser.parse_args()


def main():
    """Main training function"""
    
    print("\n" + "="*70)
    print("🧠 BRAIN TUMOR DETECTION - TRAINING")
    print("="*70)
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup
    set_random_seeds(args.seed)
    setup_gpu_memory_growth()
    create_directory_structure()
    hardware_info = get_hardware_info()
    
    # Create configuration
    config = Config()
    
    # Override config with command line arguments
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.learning_rate:
        config.LEARNING_RATE = args.learning_rate
    if args.image_size:
        config.IMAGE_SIZE = tuple(args.image_size)
        config.INPUT_SHAPE = (*config.IMAGE_SIZE, 3)
    if args.no_augmentation:
        config.AUGMENTATION = {}
    
    # Print configuration
    print("\n⚙️ Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Image size: {config.IMAGE_SIZE}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print(f"  Epochs: {config.EPOCHS}")
    print(f"  Augmentation: {'Enabled' if config.AUGMENTATION else 'Disabled'}")
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    try:
        # Train model
        results = trainer.train_model(
            model_type=args.model,
            epochs=config.EPOCHS
        )
        
        # Plot training history
        trainer.plot_training_history(save_plots=True)
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETED SUCCESSFULLY")
        print("="*70)
        
        # Print summary
        print(f"\n📋 Results Summary:")
        print(f"  Model saved to: {results['model_path']}")
        print(f"  Training time: {results['training_time']/60:.2f} minutes")
        print(f"  Final training accuracy: {results['history']['accuracy'][-1]:.4f}")
        print(f"  Final validation accuracy: {results['history']['val_accuracy'][-1]:.4f}")
        print(f"  Best validation accuracy: {max(results['history']['val_accuracy']):.4f}")
        
        # Save configuration
        config_path = config.RESULTS_DIR / "training_config.json"
        config_dict = {
            'model_type': args.model,
            'image_size': config.IMAGE_SIZE,
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE,
            'epochs': config.EPOCHS,
            'augmentation': bool(config.AUGMENTATION),
            'timestamp': datetime.now().isoformat(),
            'hardware_info': hardware_info
        }
        
        import json
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"\n✅ Configuration saved to: {config_path}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
    
    if results:
        print("\n🎉 Training completed successfully!")
    else:
        print("\n⚠️ Training failed. Please check the error messages above.")
        sys.exit(1)