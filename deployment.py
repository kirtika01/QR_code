import os
import torch
import joblib
import numpy as np
import cv2
from torchvision import transforms
from deep_learning import QRCodeCNN
import time
import json
from feature_engineering import QRFeatureExtractor

class QRCodeClassifier:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.traditional_ml_model = None
        self.deep_learning_model = None
        self.scaler = None
        self.feature_selector = None
        self.feature_extractor = QRFeatureExtractor()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Load models
        self._load_models()
        
        # Performance monitoring
        self.performance_metrics = {
            'traditional_ml': {'total_time': 0, 'calls': 0},
            'deep_learning': {'total_time': 0, 'calls': 0}
        }
    
    def _load_models(self):
        """
        Load both traditional ML and deep learning models
        """
        try:
            # Load traditional ML model
            self.traditional_ml_model = joblib.load('models/best_model.joblib')
            self.scaler = joblib.load('models/scaler.joblib')
            self.feature_selector = joblib.load('models/feature_selector.joblib')
            
            # Load deep learning model
            self.deep_learning_model = QRCodeCNN().to(self.device)
            self.deep_learning_model.load_state_dict(torch.load('best_model.pth'))
            self.deep_learning_model.eval()
            
            print("Models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for both models
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        # For traditional ML
        features = self.feature_extractor.extract_features(image_path)
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        
        # For deep learning
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor_image = self.transform(image_rgb).unsqueeze(0).to(self.device)
        
        return feature_vector, tensor_image
    
    def predict_traditional_ml(self, feature_vector):
        """
        Make prediction using traditional ML model
        """
        start_time = time.time()
        
        # Scale features
        scaled_features = self.scaler.transform(feature_vector)
        
        # Apply feature selection
        selected_features = self.feature_selector.transform(scaled_features)
        
        # Make prediction with selected features
        prediction = self.traditional_ml_model.predict(selected_features)[0]
        probability = self.traditional_ml_model.predict_proba(selected_features)[0]
        
        end_time = time.time()
        self.performance_metrics['traditional_ml']['total_time'] += (end_time - start_time)
        self.performance_metrics['traditional_ml']['calls'] += 1
        
        return prediction, probability
    
    def predict_deep_learning(self, tensor_image):
        """
        Make prediction using deep learning model
        """
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.deep_learning_model(tensor_image)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            probability = probabilities[0][prediction].item()
        
        end_time = time.time()
        self.performance_metrics['deep_learning']['total_time'] += (end_time - start_time)
        self.performance_metrics['deep_learning']['calls'] += 1
        
        return prediction, probability
    
    def classify_image(self, image_path, method='both'):
        """
        Classify QR code image using specified method
        """
        try:
            feature_vector, tensor_image = self.preprocess_image(image_path)
            results = {'filename': os.path.basename(image_path)}
            
            if method in ['traditional_ml', 'both']:
                pred_ml, prob_ml = self.predict_traditional_ml(feature_vector)
                results['traditional_ml'] = {
                    'prediction': 'Second Print' if pred_ml == 1 else 'First Print',
                    'confidence': float(max(prob_ml))
                }
            
            if method in ['deep_learning', 'both']:
                pred_dl, prob_dl = self.predict_deep_learning(tensor_image)
                results['deep_learning'] = {
                    'prediction': 'Second Print' if pred_dl == 1 else 'First Print',
                    'confidence': prob_dl
                }
            
            return results
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_performance_metrics(self):
        """
        Calculate and return performance metrics
        """
        metrics = {}
        
        for model_type, data in self.performance_metrics.items():
            if data['calls'] > 0:
                avg_time = data['total_time'] / data['calls']
                metrics[model_type] = {
                    'average_inference_time': avg_time,
                    'total_calls': data['calls']
                }
        
        return metrics

def main():
    # Initialize classifier
    classifier = QRCodeClassifier()
    
    # Test images from both directories
    test_images = [
        "dataset/First Print/input_image_active.png",
        "dataset/First Print/input_image_bottle.png",
        "dataset/First Print/input_image_canvas.png",
        "dataset/Second Print/input_image_active (1).png",
        "dataset/Second Print/input_image_bottle(1).png",
        "dataset/Second Print/input_image_canvas(1).png"
    ]
    
    print("\nTesting multiple images...")
    print("-" * 50)
    
    for image_path in test_images:
        result = classifier.classify_image(image_path)
        print(f"\nImage: {os.path.basename(image_path)}")
        print("Expected:", "Second Print" if "Second Print" in image_path else "First Print")
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            if 'traditional_ml' in result:
                print(f"Traditional ML: {result['traditional_ml']['prediction']} "
                      f"(Confidence: {result['traditional_ml']['confidence']:.2%})")
            
            if 'deep_learning' in result:
                print(f"Deep Learning: {result['deep_learning']['prediction']} "
                      f"(Confidence: {result['deep_learning']['confidence']:.2%})")
    
    # Print performance metrics
    print("\nPerformance Metrics:")
    metrics = classifier.get_performance_metrics()
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()