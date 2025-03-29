import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from deep_learning import QRCodeCNN, QRCodeDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import json
import os

class ModelEvaluator:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.results = {}
    
    def load_traditional_model(self):
        """
        Load the trained traditional ML model
        """
        model = joblib.load('models/best_model.joblib')
        scaler = joblib.load('models/scaler.joblib')
        feature_selector = joblib.load('models/feature_selector.joblib')
        return model, scaler, feature_selector
    
    def load_deep_learning_model(self):
        """
        Load the trained deep learning model
        """
        model = QRCodeCNN().to(self.device)
        model.load_state_dict(torch.load('models/best_model.pth'))
        model.eval()
        return model
    
    def evaluate_traditional_ml(self, X_test, y_test):
        """
        Evaluate traditional ML model
        """
        print("\nEvaluating Traditional ML Model...")
        print(f"Test set shape: {X_test.shape}, Labels distribution: {np.bincount(y_test)}")
        
        model, scaler, feature_selector = self.load_traditional_model()
        
        # Scale features
        X_test_scaled = scaler.transform(X_test)
        
        # Select features
        X_test_selected = feature_selector.transform(X_test_scaled)
        
        # Get predictions
        y_pred = model.predict(X_test_selected)
        y_pred_prob = model.predict_proba(X_test_selected)[:, 1]
        
        print(f"Predictions distribution: {np.bincount(y_pred)}")
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_prob)
        self.results['traditional_ml'] = metrics
        
        # Plot confusion matrix
        self._plot_confusion_matrix(y_test, y_pred, 'traditional_ml')
        
        # Plot ROC curve
        self._plot_roc_curve(y_test, y_pred_prob, 'traditional_ml')
    
    def evaluate_deep_learning(self, test_loader):
        """
        Evaluate deep learning model
        """
        print("\nEvaluating Deep Learning Model...")
        print(f"Test loader size: {len(test_loader.dataset)}")
        
        model = self.load_deep_learning_model()
        
        y_true = []
        y_pred = []
        y_pred_prob = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)
                
                _, predicted = torch.max(outputs.data, 1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                y_pred_prob.extend(probabilities[:, 1].cpu().numpy())
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        print(f"Deep Learning predictions distribution: {np.bincount(y_pred)}")
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_true, y_pred, y_pred_prob)
        self.results['deep_learning'] = metrics
        
        # Plot confusion matrix
        self._plot_confusion_matrix(y_true, y_pred, 'deep_learning')
        
        # Plot ROC curve
        self._plot_roc_curve(y_true, y_pred_prob, 'deep_learning')
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_prob):
        """
        Calculate comprehensive performance metrics
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        
        # Calculate ROC-AUC
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        # Calculate confusion matrix based metrics
        cm = confusion_matrix(y_true, y_pred)
        
        # Handle case where confusion matrix is not 2x2
        if cm.shape != (2, 2):
            print(f"Warning: Expected 2x2 confusion matrix but got shape {cm.shape}")
            print(f"Confusion matrix:\n{cm}")
            print(f"Unique values in predictions: {np.unique(y_pred)}")
            specificity = 0.0
            sensitivity = 0.0
            false_positive_rate = 0.0
            false_negative_rate = 0.0
        else:
            # Calculate metrics only if we have a proper 2x2 matrix
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            false_positive_rate = fp / (tn + fp) if (tn + fp) > 0 else 0.0
            false_negative_rate = fn / (tp + fn) if (tp + fn) > 0 else 0.0
        
        balanced_accuracy = (sensitivity + specificity) / 2
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'false_positive_rate': float(false_positive_rate),
            'false_negative_rate': float(false_negative_rate),
            'balanced_accuracy': float(balanced_accuracy),
            'confusion_matrix': cm.tolist()
        }
    
    def _plot_confusion_matrix(self, y_true, y_pred, model_type):
        """
        Plot and save confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # Handle case where confusion matrix is not 2x2
        if cm.shape != (2, 2):
            print(f"Warning: Expected 2x2 confusion matrix but got shape {cm.shape}")
            # Create a 2x2 matrix of zeros
            cm = np.zeros((2, 2))
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='.0f', 
            cmap='Blues',
            xticklabels=['First Print', 'Second Print'],
            yticklabels=['First Print', 'Second Print']
        )
        
        # Add warning text if matrix is invalid
        if np.sum(cm) == 0:
            plt.text(0.5, 0.5, 'Invalid predictions', 
                    ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.title(f'Confusion Matrix - {model_type}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'metrics/confusion_matrix_{model_type}.png')
        plt.close()
    
    def _plot_roc_curve(self, y_true, y_pred_prob, model_type):
        """
        Plot and save ROC curve
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_type}')
        plt.legend(loc="lower right")
        plt.savefig(f'metrics/roc_curve_{model_type}.png')
        plt.close()
    
    def compare_models(self):
        """
        Compare and visualize performance of both models
        """
        if not self.results:
            print("No evaluation results available")
            return
        
        # Select metrics for comparison
        metrics_to_compare = [
            'accuracy', 'precision', 'recall', 'f1_score', 
            'specificity', 'sensitivity', 'roc_auc'
        ]
        model_types = list(self.results.keys())
        
        # Create comparison plot
        plt.figure(figsize=(12, 6))
        x = np.arange(len(metrics_to_compare))
        width = 0.35
        
        for i, model_type in enumerate(model_types):
            values = [self.results[model_type][metric] for metric in metrics_to_compare]
            plt.bar(x + i*width, values, width, label=model_type)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x + width/2, metrics_to_compare, rotation=45)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('metrics/model_comparison.png')
        plt.close()
        
        # Save results to JSON with more readable format
        formatted_results = {
            'model_comparison': {},
            'individual_results': self.results
        }
        
        # Add comparative summary
        for metric in metrics_to_compare:
            formatted_results['model_comparison'][metric] = {
                model_type: self.results[model_type][metric]
                for model_type in model_types
            }
        
        # Save to JSON
        with open('metrics/evaluation_results.json', 'w') as f:
            json.dump(formatted_results, f, indent=4)

def main():
    # Create metrics directory if it doesn't exist
    os.makedirs('metrics', exist_ok=True)
    
    evaluator = ModelEvaluator()
    
    # Load test data for traditional ML
    features = np.load('features.npy')
    labels = np.load('labels.npy')
    
    # Use stratified split to maintain class balance
    _, X_test, _, y_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    # Evaluate traditional ML model
    evaluator.evaluate_traditional_ml(X_test, y_test)
    
    # Create test dataset for deep learning
    test_dataset = QRCodeDataset(transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    ]))
    
    # Use same test indices
    test_indices = range(len(X_test))  # Use same size for deep learning test set
    test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Evaluate deep learning model
    evaluator.evaluate_deep_learning(test_loader)
    
    # Compare models
    evaluator.compare_models()
    print("\nEvaluation complete! Results saved to 'metrics/evaluation_results.json'")

if __name__ == "__main__":
    main()