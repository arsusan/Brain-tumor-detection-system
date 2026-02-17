#!/usr/bin/env python3
"""
Brain Tumor Categorization - Prediction Script
Updated for 4-class classification: Glioma, Meningioma, Pituitary, and No Tumor
"""

import os
import sys
import argparse
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.preprocessing import ImagePreprocessor

# Define the global class mapping
# NOTE: This order must match the training generator indices
CLASS_LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']

class BrainTumorPredictor:
    """Brain tumor categorization class"""
    
    def __init__(self, model_path: str, config: Config = None):
        """Initialize predictor with trained multiclass model"""
        import tensorflow as tf
        from tensorflow import keras
        
        self.config = config or Config()
        self.preprocessor = ImagePreprocessor(self.config)
        
        print(f"📂 Loading multiclass model from: {model_path}")
        self.model = keras.models.load_model(model_path)
        print("✅ Model loaded successfully")
        
    def predict_single(self, image_path: str) -> Dict[str, Any]:
        """Make categorization on a single image using Softmax output"""
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Preprocess image
        image = self.preprocessor.preprocess_single(image_path, augment=False)
        
        # Make prediction (Expected output shape: [1, 4])
        predictions = self.model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
        
        # Determine class using argmax (highest probability)
        predicted_idx = np.argmax(predictions)
        predicted_class = CLASS_LABELS[predicted_idx]
        confidence = float(predictions[predicted_idx])
        
        # Create a detailed probability map for all classes
        prob_map = {CLASS_LABELS[i]: float(predictions[i]) for i in range(len(CLASS_LABELS))}
        
        return {
            'image_path': image_path,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': prob_map,
            'timestamp': datetime.now().isoformat()
        }
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Make predictions on multiple images"""
        results = []
        print(f"🔍 Processing {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths):
            try:
                result = self.predict_single(image_path)
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    print(f"   Processed {i + 1}/{len(image_paths)} images...")
                    
            except Exception as e:
                print(f"⚠️ Error processing {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e),
                    'predicted_class': 'Error',
                    'confidence': 0.0
                })
        
        print(f"✅ Completed {len(results)} predictions")
        return results
    
    def predict_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """Make predictions on all images in a directory"""
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = []
        
        for root, _, files in os.walk(directory_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))
        
        if not image_paths:
            print(f"⚠️ No images found in {directory_path}")
            return []
        
        return self.predict_batch(image_paths)

    def save_predictions(self, predictions: List[Dict[str, Any]], output_file: str = "predictions.json"):
        """Save results to JSON file"""
        output_path = self.config.RESULTS_DIR / "predictions" / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        print(f"✅ Full results saved to: {output_path}")
        return output_path

    def print_predictions_summary(self, predictions: List[Dict[str, Any]]):
        """Print multiclass summary"""
        if not predictions:
            return
        
        # Initialize counts
        summary_counts = {label: 0 for label in CLASS_LABELS}
        summary_counts['Error'] = 0
        
        for p in predictions:
            summary_counts[p.get('predicted_class', 'Error')] += 1
        
        print(f"\n📊 Categorization Summary (Total: {len(predictions)}):")
        for label in CLASS_LABELS:
            print(f"   {label.capitalize()}: {summary_counts[label]}")
        
        if summary_counts['Error'] > 0:
            print(f"   Errors: {summary_counts['Error']}")
        
        valid_confidences = [p['confidence'] for p in predictions if 'confidence' in p and p['predicted_class'] != 'Error']
        if valid_confidences:
            print(f"   Average Confidence: {np.mean(valid_confidences):.2%}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Categorize Brain Tumor Types from MRI')
    parser.add_argument('--model', type=str, required=True, help='Path to trained .keras model')
    parser.add_argument('--image', type=str, help='Path to single MRI image')
    parser.add_argument('--directory', type=str, help='Path to image directory')
    parser.add_argument('--output', type=str, default='categorization_results.json', help='Output JSON filename')
    return parser.parse_args()

def main():
    print("\n" + "="*70)
    print("🧠 BRAIN TUMOR CATEGORIZATION SYSTEM - INFERENCE")
    print("="*70)
    
    args = parse_arguments()
    
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        sys.exit(1)

    try:
        predictor = BrainTumorPredictor(args.model)
        predictions = []

        if args.image:
            predictions = [predictor.predict_single(args.image)]
        elif args.directory:
            predictions = predictor.predict_directory(args.directory)
        else:
            print("❌ Please provide --image or --directory")
            sys.exit(1)

        predictor.print_predictions_summary(predictions)
        predictor.save_predictions(predictions, args.output)
        
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()