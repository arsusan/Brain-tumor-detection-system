#!/usr/bin/env python3
"""
Brain Tumor Categorization - Evaluation Script
Project by: Susan Aryal
"""

import os
import sys
import argparse
import json
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.utils import (
    set_random_seeds, 
    plot_multiclass_confusion_matrix, 
    save_detailed_report,
    plot_sample_predictions
)
from src.data_loader import BrainTumorDataLoader
from src.preprocessing import ImagePreprocessor

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate Brain Tumor 4-Class Model')
    
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model file (.keras)')
    
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

def load_testing_data(config: Config):
    """Prepare the testing dataset generator by first loading paths/labels"""
    print("📊 Preparing test data...")
    data_loader = BrainTumorDataLoader(config)
    preprocessor = ImagePreprocessor(config)
    data_loader.processor = preprocessor 
    
    # 1. Load the file paths and categorical labels from the Testing folder
    test_paths, test_labels = data_loader.load_categorical_dataset('Testing')
    
    print(f"✅ Found {len(test_paths)} images in the Testing set.")

    # 2. Create the generator passing the required 3 positional arguments
    # We set shuffle=False and augment=False for accurate evaluation
    test_gen = data_loader.create_data_generator(
        image_paths=test_paths,
        labels=test_labels,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        augment=False
    )
    
    return test_gen, test_labels

def main():
    print("\n" + "="*70)
    print("🧠 BRAIN TUMOR CATEGORIZATION - EVALUATION")
    print("="*70)
    
    args = parse_arguments()
    config = Config()
    set_random_seeds(args.seed)

    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        sys.exit(1)

    try:
        # 1. Load Model
        print(f"📂 Loading model from: {args.model}")
        model = tf.keras.models.load_model(args.model)
        
        # 2. Load Data
        test_gen, test_labels = load_testing_data(config)
        
        # 3. Run Predictions
        print("🔍 Predicting on test set...")
        
        y_true_list = []
        y_pred_probs_list = []
        sample_images = None
        sample_labels = None

        # Iterate through the dataset to collect predictions
        # Using a loop ensures we handle the tf.data.Dataset correctly
        for imgs, lbls in test_gen:
            # Check if we have processed all samples (tf.data generators can loop infinitely)
            if len(y_true_list) * args.batch_size >= len(test_labels):
                break
                
            preds = model.predict(imgs, verbose=0)
            y_pred_probs_list.append(preds)
            y_true_list.append(lbls.numpy())
            
            if sample_images is None:
                sample_images = imgs.numpy()
                sample_labels = lbls.numpy()

        # Concatenate all batches
        y_pred_probs = np.vstack(y_pred_probs_list)
        y_true = np.vstack(y_true_list)
        
        # Ensure we don't have extra samples if the last batch was smaller
        y_pred_probs = y_pred_probs[:len(test_labels)]
        y_true = y_true[:len(test_labels)]
        
        # Convert probabilities to class indices
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true_idx = np.argmax(y_true, axis=1)

        # 4. Generate Visualizations
        plots_dir = config.RESULTS_DIR / "plots"
        metrics_dir = config.RESULTS_DIR / "metrics"
        plots_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        # Confusion Matrix
        plot_multiclass_confusion_matrix(
            y_true_idx, 
            y_pred, 
            classes=config.CLASSES,
            save_path=plots_dir / "final_confusion_matrix.png"
        )

        # Sample Prediction Visualization (showing first batch)
        plot_sample_predictions(
            sample_images,
            sample_labels,
            y_pred_probs[:len(sample_images)],
            classes=config.CLASSES,
            save_path=plots_dir / "sample_test_results.png"
        )

        # 5. Save and Print Detailed Report
        save_detailed_report(
            y_true_idx,
            y_pred,
            classes=config.CLASSES,
            save_path=metrics_dir / "evaluation_report.txt"
        )

        # Calculate Overall Accuracy
        accuracy = np.mean(y_pred == y_true_idx)
        
        print("\n" + "="*70)
        print(f"✅ EVALUATION COMPLETE: {accuracy:.2%} Accuracy")
        print("="*70)

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()