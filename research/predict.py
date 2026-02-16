#!/usr/bin/env python3
"""
Brain Tumor Detection - Prediction Script
Make predictions on new MRI images using trained model
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.preprocessing import ImagePreprocessor


class BrainTumorPredictor:
    """Brain tumor prediction class"""
    
    def __init__(self, model_path: str, config: Config = None):
        """Initialize predictor with trained model"""
        import tensorflow as tf
        from tensorflow import keras
        
        self.config = config or Config()
        self.preprocessor = ImagePreprocessor(self.config)
        
        print(f"📂 Loading model from: {model_path}")
        self.model = keras.models.load_model(model_path)
        print("✅ Model loaded successfully")
        
    def predict_single(self, image_path: str, threshold: float = 0.5) -> Dict[str, Any]:
        """Make prediction on a single image"""
        
        # Check if image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Preprocess image
        image = self.preprocessor.preprocess_single(image_path, augment=False)
        
        # Make prediction
        prediction = self.model.predict(np.expand_dims(image, axis=0), verbose=0)[0][0]
        
        # Determine class and confidence
        if prediction > threshold:
            predicted_class = "Tumor"
            confidence = float(prediction)
        else:
            predicted_class = "No Tumor"
            confidence = 1.0 - float(prediction)
        
        return {
            'image_path': image_path,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'raw_score': float(prediction),
            'threshold': threshold
        }
    
    def predict_batch(self, image_paths: List[str], threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Make predictions on multiple images"""
        
        results = []
        print(f"🔍 Processing {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths):
            try:
                result = self.predict_single(image_path, threshold)
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(image_paths)} images...")
                    
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
    
    def predict_directory(self, directory_path: str, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Make predictions on all images in a directory"""
        
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = []
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))
        
        if not image_paths:
            print(f"⚠️ No images found in {directory_path}")
            return []
        
        print(f"📂 Found {len(image_paths)} images in {directory_path}")
        return self.predict_batch(image_paths, threshold)
    
    def save_predictions(self, predictions: List[Dict[str, Any]], 
                        output_file: str = "predictions.json"):
        """Save predictions to JSON file"""
        
        output_path = self.config.RESULTS_DIR / "predictions" / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        print(f"✅ Predictions saved to: {output_path}")
        return output_path
    
    def print_predictions_summary(self, predictions: List[Dict[str, Any]]):
        """Print summary of predictions"""
        
        if not predictions:
            print("⚠️ No predictions to summarize")
            return
        
        # Count predictions by class
        tumor_count = sum(1 for p in predictions if p.get('predicted_class') == 'Tumor')
        no_tumor_count = sum(1 for p in predictions if p.get('predicted_class') == 'No Tumor')
        error_count = sum(1 for p in predictions if p.get('predicted_class') == 'Error')
        
        print(f"\n📊 Predictions Summary:")
        print(f"  Total images: {len(predictions)}")
        print(f"  Tumor predictions: {tumor_count}")
        print(f"  No Tumor predictions: {no_tumor_count}")
        
        if error_count > 0:
            print(f"  Errors: {error_count}")
        
        # Calculate average confidence
        valid_predictions = [p for p in predictions if 'confidence' in p and p['confidence'] > 0]
        if valid_predictions:
            avg_confidence = np.mean([p['confidence'] for p in valid_predictions])
            print(f"  Average confidence: {avg_confidence:.2%}")
        
        # Show sample predictions
        print(f"\n🔍 Sample Predictions (first 5):")
        for i, pred in enumerate(predictions[:5]):
            if 'error' not in pred:
                print(f"  {i+1}. {pred['image_path']}")
                print(f"     → {pred['predicted_class']} (confidence: {pred['confidence']:.2%})")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Predict Brain Tumor from MRI Images')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file')
    
    parser.add_argument('--image', type=str,
                       help='Path to single MRI image')
    
    parser.add_argument('--directory', type=str,
                       help='Path to directory containing MRI images')
    
    parser.add_argument('--batch-file', type=str,
                       help='Path to text file containing image paths')
    
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold (default: 0.5)')
    
    parser.add_argument('--output', type=str, default='predictions.json',
                       help='Output file name for predictions')
    
    return parser.parse_args()


def main():
    """Main prediction function"""
    
    print("\n" + "="*70)
    print("🧠 BRAIN TUMOR DETECTION - PREDICTION")
    print("="*70)
    
    # Parse arguments
    args = parse_arguments()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        sys.exit(1)
    
    # Check input source
    input_sources = sum([bool(args.image), bool(args.directory), bool(args.batch_file)])
    if input_sources == 0:
        print("❌ Please provide an input source: --image, --directory, or --batch-file")
        sys.exit(1)
    if input_sources > 1:
        print("❌ Please provide only one input source")
        sys.exit(1)
    
    try:
        # Initialize predictor
        config = Config()
        predictor = BrainTumorPredictor(args.model, config)
        
        # Get image paths
        image_paths = []
        
        if args.image:
            # Single image
            image_paths = [args.image]
            print(f"🔍 Processing single image: {args.image}")
            
        elif args.directory:
            # Directory of images
            print(f"📂 Processing directory: {args.directory}")
            predictions = predictor.predict_directory(args.directory, args.threshold)
            
        elif args.batch_file:
            # Batch file with image paths
            if not os.path.exists(args.batch_file):
                print(f"❌ Batch file not found: {args.batch_file}")
                sys.exit(1)
            
            with open(args.batch_file, 'r') as f:
                image_paths = [line.strip() for line in f if line.strip()]
            
            print(f"📄 Processing {len(image_paths)} images from batch file: {args.batch_file}")
        
        # Make predictions
        if image_paths:
            predictions = predictor.predict_batch(image_paths, args.threshold)
        
        # Print summary
        predictor.print_predictions_summary(predictions)
        
        # Save predictions
        output_path = predictor.save_predictions(predictions, args.output)
        
        print("\n" + "="*70)
        print("✅ PREDICTION COMPLETED SUCCESSFULLY")
        print("="*70)
        
        print(f"\n📁 Results saved to: {output_path}")
        print("\n💡 Next steps:")
        print(f"1. View predictions: cat {output_path} | head -20")
        print("2. Load predictions in Python: import json; data = json.load(open('predictions.json'))")
        
        return predictions
        
    except Exception as e:
        print(f"\n❌ Prediction failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    predictions = main()
    
    if predictions:
        print("\n🎉 Predictions completed successfully!")
    else:
        print("\n⚠️ Prediction failed.")
        sys.exit(1)