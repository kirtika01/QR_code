import os
import argparse
from data_exploration import main as explore_data
from feature_engineering import main as extract_features
from traditional_ml import main as train_traditional_ml
from deep_learning import main as train_deep_learning
from evaluate import main as evaluate_models
from deployment import QRCodeClassifier
import json

def parse_arguments():
    parser = argparse.ArgumentParser(description='QR Code Classification System')
    parser.add_argument('--mode', type=str, default='full',
                      choices=['explore', 'extract', 'train_traditional',
                              'train_deep', 'evaluate', 'predict', 'full'],
                      help='Mode of operation')
    parser.add_argument('--image', type=str,
                      help='Path to image for prediction (only for predict mode)')
    return parser.parse_args()

def check_dataset():
    """
    Verify dataset structure and files
    """
    required_dirs = ['dataset/First Print', 'dataset/Second Print']
    for directory in required_dirs:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Required directory {directory} not found")
        
        files = os.listdir(directory)
        if not any(f.endswith('.png') for f in files):
            raise FileNotFoundError(f"No PNG files found in {directory}")
    
    print("Dataset structure verified successfully")

def run_full_pipeline():
    """
    Run the complete pipeline from data exploration to model evaluation
    """
    print("\n1. Exploring and analyzing dataset...")
    explore_data()
    
    print("\n2. Extracting features...")
    extract_features()
    
    print("\n3. Training traditional ML models...")
    train_traditional_ml()
    
    print("\n4. Training deep learning model...")
    train_deep_learning()
    
    print("\n5. Evaluating models...")
    evaluate_models()
    
    print("\nPipeline completed successfully!")

def predict_image(image_path):
    """
    Make prediction on a single image using both models
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Initialize classifier
    classifier = QRCodeClassifier()
    
    # Make prediction
    result = classifier.classify_image(image_path)
    
    # Print results
    print("\nClassification Results:")
    print(json.dumps(result, indent=2))
    
    # Get performance metrics
    metrics = classifier.get_performance_metrics()
    print("\nPerformance Metrics:")
    print(json.dumps(metrics, indent=2))

def main():
    """
    Main function to orchestrate the QR code classification system
    """
    args = parse_arguments()
    
    try:
        # Verify dataset structure (except for predict mode)
        if args.mode != 'predict':
            check_dataset()
        
        # Execute based on mode
        if args.mode == 'explore':
            explore_data()
        elif args.mode == 'extract':
            extract_features()
        elif args.mode == 'train_traditional':
            train_traditional_ml()
        elif args.mode == 'train_deep':
            train_deep_learning()
        elif args.mode == 'evaluate':
            evaluate_models()
        elif args.mode == 'predict':
            if not args.image:
                raise ValueError("Image path required for predict mode")
            predict_image(args.image)
        elif args.mode == 'full':
            run_full_pipeline()
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())