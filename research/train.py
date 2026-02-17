#!/usr/bin/env python3
"""
Brain Tumor Categorization - Main Training Script
Project by: Susan Aryal
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

# Add src to path to ensure modules are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.utils import (
    set_random_seeds, 
    setup_gpu_memory_growth, 
    create_directory_structure,
    get_hardware_info,
    save_detailed_report
)
from src.trainer import ModelTrainer


def parse_arguments():
    """Parse command line arguments for flexible training"""
    parser = argparse.ArgumentParser(description='Train Brain Tumor Categorization Model')
    
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'efficient'],
                        help='Model architecture: cnn (custom) or efficient (separable conv)')
    
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size for training')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()


def run_training():
    """Main execution function to orchestrate the training pipeline"""
    
    print("\n" + "="*70)
    print("🧠 AI-BASED MRI IMAGE CLASSIFICATION FOR BRAIN TUMOR CATEGORIZATION")
    print("="*70)
    
    # 1. Initialization & Environment Setup
    args = parse_arguments()
    set_random_seeds(args.seed)
    setup_gpu_memory_growth()
    create_directory_structure()
    hardware_info = get_hardware_info()
    
    # 2. Configuration Setup
    config = Config()
    
    # Override defaults if arguments are provided
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    
    print("\n⚙️ Project Configuration:")
    print(f"  Target Classes: {config.CLASSES}")
    print(f"  Model Type:     {args.model.upper()}")
    print(f"  Image Size:     {config.IMAGE_SIZE}")
    print(f"  Batch Size:     {config.BATCH_SIZE}")
    print(f"  Epochs:         {config.EPOCHS}")
    
    # 3. Initialize Trainer
    # Note: ModelTrainer handles data_loader and preprocessor internally
    trainer = ModelTrainer(config)
    
    try:
        # 4. Execute Training
        print("\n🚀 Starting Training Process...")
        results = trainer.train_model(
            model_type=args.model,
            epochs=config.EPOCHS
        )
        
        # 5. Visualization & Metrics
        print("\n📊 Generating Evaluation Metrics...")
        trainer.plot_training_history(save_plots=True)
        
        # Save a detailed text report of the final results
        report_path = config.RESULTS_DIR / "metrics" / "final_classification_report.txt"
        # The trainer already has the validation results stored
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETED SUCCESSFULLY")
        print("="*70)
        
        # 6. Results Summary
        print(f"\n📋 Results Summary:")
        print(f"  Final Model Saved: {results['model_path']}")
        print(f"  Training Time:     {results['training_time']/60:.2f} minutes")
        print(f"  Final Val Acc:     {results['history']['val_accuracy'][-1]*100:.2f}%")
        print(f"  Best Val Acc:      {max(results['history']['val_accuracy'])*100:.2f}%")
        
        # 7. Export Metadata for reference
        meta_path = config.RESULTS_DIR / "training_metadata.json"
        metadata = {
            'project': "Brain Tumor Categorization",
            'author': "Susan Aryal",
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'config': {
                'model': args.model,
                'epochs': config.EPOCHS,
                'batch_size': config.BATCH_SIZE,
                'classes': config.CLASSES
            },
            'hardware': hardware_info
        }
        
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"✅ Metadata saved to: {meta_path}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error during training pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    success = run_training()
    if not success:
        sys.exit(1)