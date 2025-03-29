import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

class TraditionalMLPipeline:
    def __init__(self):
        self.models = {
            'svm': {
                'model': SVC(probability=True),
                'params': {
                    'C': [0.1, 0.5, 1.0, 2.0],
                    'kernel': ['rbf'],
                    'gamma': ['scale', 'auto', 0.1, 0.01]
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(class_weight='balanced'),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'knn': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                }
            }
        }
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.best_models = {}
        self.ensemble = None
        self.cv_scores = {}

    def load_data(self):
        """
        Load preprocessed features and labels
        """
        features = np.load('features.npy')
        labels = np.load('labels.npy')
        return features, labels

    def prepare_data(self, features, labels, test_size=0.2, random_state=42):
        """
        Split data, scale features and perform feature selection
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features.astype(float), labels, test_size=test_size, random_state=random_state,
            stratify=labels
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Feature selection using Random Forest with threshold
        selector = RandomForestClassifier(n_estimators=100, random_state=42)
        selector.fit(X_train_scaled, y_train)
        
        # Select features based on importance threshold
        self.feature_selector = SelectFromModel(selector, prefit=True, threshold='mean')
        X_train_selected = self.feature_selector.transform(X_train_scaled)
        
        # Save feature importances before selection
        np.save('feature_importances.npy', selector.feature_importances_)
        X_test_selected = self.feature_selector.transform(X_test_scaled)
        
        print(f"Selected {X_train_selected.shape[1]} features out of {X_train_scaled.shape[1]}")
        
        return X_train_selected, X_test_selected, y_train, y_test

    def cross_validate_model(self, model, X, y, cv=5):
        """
        Perform cross-validation
        """
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        return cv_scores.mean(), cv_scores.std()

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """
        Train and evaluate multiple models using cross-validation
        """
        results = {}
        best_models = {}
        
        # Train individual models
        for model_name, model_info in self.models.items():
            print(f"\nTraining {model_name}...")
            
            # Initialize StratifiedKFold
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            best_score = 0
            best_params = None
            best_model = None
            
            # Grid search with cross-validation
            for params in self._param_grid(model_info['params']):
                model = clone(model_info['model'])
                model.set_params(**params)
                
                # Cross-validation scores
                cv_scores = []
                for train_idx, val_idx in skf.split(X_train, y_train):
                    X_train_fold = X_train[train_idx]
                    y_train_fold = y_train[train_idx]
                    X_val_fold = X_train[val_idx]
                    y_val_fold = y_train[val_idx]
                    
                    model.fit(X_train_fold, y_train_fold)
                    score = model.score(X_val_fold, y_val_fold)
                    cv_scores.append(score)
                
                avg_score = np.mean(cv_scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_params = params
                    best_model = clone(model)
            
            # Train final model with best parameters
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            
            # Store results
            results[model_name] = {
                'model': best_model,
                'best_params': best_params,
                'cv_score': best_score,
                'test_score': best_model.score(X_test, y_test),
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            best_models[model_name] = best_model
        
        # Create and train ensemble with weights based on CV scores
        weights = [results[name]['cv_score'] for name in best_models.keys()]
        weights = np.array(weights) / sum(weights)  # Normalize weights
        
        self.ensemble = VotingClassifier(
            estimators=[(name, model) for name, model in best_models.items()],
            voting='soft',
            weights=weights
        )
        self.ensemble.fit(X_train, y_train)
        
        # Evaluate ensemble
        ensemble_pred = self.ensemble.predict(X_test)
        ensemble_proba = self.ensemble.predict_proba(X_test)[:, 1]
        
        results['ensemble'] = {
            'model': self.ensemble,
            'test_score': self.ensemble.score(X_test, y_test),
            'classification_report': classification_report(y_test, ensemble_pred),
            'confusion_matrix': confusion_matrix(y_test, ensemble_pred),
            'predictions': ensemble_pred,
            'probabilities': ensemble_proba
        }
        
        return results

    def _param_grid(self, param_dict):
        """
        Generate parameter combinations
        """
        import itertools
        keys = param_dict.keys()
        values = param_dict.values()
        for instance in itertools.product(*values):
            yield dict(zip(keys, instance))

    def plot_results(self, results, y_test):
        """
        Visualize model performance comparisons and ROC curves
        """
        # Prepare data for plotting
        model_names = list(results.keys())
        test_scores = [results[model]['test_score'] for model in model_names]
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        plt.bar(model_names, test_scores)
        plt.xlabel('Models')
        plt.ylabel('Test Accuracy')
        plt.title('Model Performance Comparison')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('metrics/model_comparison.png')
        plt.close()
        
        # Plot ROC curves
        plt.figure(figsize=(10, 6))
        for model_name in model_names:
            if 'probabilities' in results[model_name]:
                fpr, tpr, _ = roc_curve(y_test, results[model_name]['probabilities'])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('metrics/roc_curves.png')
        plt.close()
        
        # Plot confusion matrix for ensemble
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            results['ensemble']['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues'
        )
        plt.title('Confusion Matrix - Ensemble Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('metrics/confusion_matrix.png')
        plt.close()

    def save_models(self, results):
        """
        Save the ensemble model and scaler
        """
        joblib.dump(self.ensemble, 'models/best_model.joblib')
        joblib.dump(self.scaler, 'models/scaler.joblib')
        joblib.dump(self.feature_selector, 'models/feature_selector.joblib')

        # Save detailed evaluation metrics
        metrics_dict = {}
        for model_name, result in results.items():
            metrics_dict[model_name] = {
                'test_accuracy': result['test_score'],
                'cv_score': result.get('cv_score', None),
                'classification_report': result['classification_report']
            }
        with open('metrics/evaluation_metrics.json', 'w') as f:
            json.dump(metrics_dict, f, indent=4)

def main():
    # Initialize pipeline
    pipeline = TraditionalMLPipeline()
    
    # Load and prepare data
    print("Loading data...")
    features, labels = pipeline.load_data()
    
    print("Preparing data...")
    X_train, X_test, y_train, y_test = pipeline.prepare_data(features, labels)
    
    # Train and evaluate models
    print("Training models...")
    results = pipeline.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Plot results
    print("\nGenerating visualizations...")
    pipeline.plot_results(results, y_test)
    
    # Save models
    print("\nSaving models...")
    pipeline.save_models(results)
    
    # Print final results
    print("\nTraining complete!")
    for model_name, result in results.items():
        print(f"\n{model_name.upper()} Results:")
        print(f"Test Accuracy: {result['test_score']:.4f}")
        if 'cv_score' in result:
            print(f"CV Score: {result['cv_score']:.4f}")
        print("\nClassification Report:")
        print(result['classification_report'])

if __name__ == "__main__":
    main()