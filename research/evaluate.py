#!/usr/bin/env python3
"""
Brain Tumor Detection - Evaluation Script
Evaluate trained model on test set
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.utils import set_random_seeds
from src.trainer import ModelTrainer
from src.evaluator import ModelEvaluator
from src.data_loader import BrainTumorDataLoader
from src.preprocessing import ImagePreprocessor


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate Brain Tumor Detection Model')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file (.keras or .h5)')
    
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation')
    
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    return parser.parse_args()


def load_model_and_data(model_path: str, config: Config):
    """Load model and prepare data"""
    
    import tensorflow as tf
    from tensorflow import keras
    
    print(f"📂 Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    
    print("📊 Preparing test data...")
    data_loader = BrainTumorDataLoader(config)
    preprocessor = ImagePreprocessor(config)
    data_loader.processor = preprocessor
    
    # Load test data
    test_paths, test_labels = data_loader.load_binary_dataset('Testing')
    
    print(f"✅ Loaded {len(test_paths)} test samples")
    
    return model, test_paths, test_labels


def main():
    """Main evaluation function"""
    
    print("\n" + "="*70)
    print("🧠 BRAIN TUMOR DETECTION - EVALUATION")
    print("="*70)
    
    # Parse arguments
    args = parse_arguments()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        sys.exit(1)
    
    # Setup
    set_random_seeds(args.seed)
    
    # Create configuration
    config = Config()
    
    try:
        # Load model and data
        model, test_paths, test_labels = load_model_and_data(args.model, config)
        
        # Initialize evaluator
        evaluator = ModelEvaluator(config)
        
        # Evaluate model
        print("\n🔍 Evaluating model...")
        results = evaluator.evaluate_model(
            model=model,
            test_paths=test_paths,
            test_labels=test_labels,
            batch_size=args.batch_size,
            threshold=args.threshold
        )
        
        # Print detailed report
        evaluator.print_detailed_report(results['metrics'])
        
        # Save results
        evaluator.save_evaluation_results(results['metrics'])
        
        # Create visualizations
        plots_dir = config.RESULTS_DIR / "plots"
        
        # Confusion matrix
        cm = np.array(results['metrics']['confusion_matrix'])
        evaluator.plot_confusion_matrix(
            cm, 
            save_path=plots_dir / "confusion_matrix.png"
        )
        
        # ROC curve
        evaluator.plot_roc_curve(
            results['y_true'], 
            results['y_pred_probs'],
            save_path=plots_dir / "roc_curve.png"
        )
        
        # Precision-Recall curve
        evaluator.plot_precision_recall_curve(
            results['y_true'],
            results['y_pred_probs'],
            save_path=plots_dir / "precision_recall_curve.png"
        )
        
        # Metrics summary
        evaluator.plot_metrics_summary(
            results['metrics'],
            save_path=plots_dir / "metrics_summary.png"
        )
        
        print("\n" + "="*70)
        print("✅ EVALUATION COMPLETED SUCCESSFULLY")
        print("="*70)
        
        # Save complete results
        results_path = config.RESULTS_DIR / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'metrics': results['metrics'],
                'model_path': args.model,
                'config': {
                    'batch_size': args.batch_size,
                    'threshold': args.threshold
                }
            }, f, indent=2)
        
        print(f"✅ Complete results saved to: {results_path}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import numpy as np
    results = main()
    
    if results:
        print("\n🎉 Evaluation completed successfully!")
        print(f"\nModel performance: {results['metrics']['accuracy']:.2%} accuracy")
    else:
        print("\n⚠️ Evaluation failed.")
        sys.exit(1)