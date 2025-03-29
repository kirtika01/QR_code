# Machine Learning Project

This project implements both traditional machine learning and deep learning approaches for classification. It includes comprehensive data exploration, feature engineering, model training, evaluation, and deployment capabilities.

## Project Structure

```
├── data_exploration.py      # Data analysis and visualization
├── feature_engineering.py   # Feature processing and engineering
├── traditional_ml.py       # Traditional machine learning models
├── deep_learning.py        # Deep learning model implementation
├── evaluate.py             # Model evaluation and metrics
├── deployment.py           # Model deployment utilities
├── main.py                # Main execution script
├── dataset/               # Raw dataset directory
│   ├── First Print/
│   └── Second Print/
├── metrics/               # Evaluation metrics and visualizations
│   ├── confusion_matrix_deep_learning.png
│   ├── confusion_matrix_traditional_ml.png
│   ├── model_comparison.png
│   ├── roc_curve_deep_learning.png
│   └── roc_curve_traditional_ml.png
└── models/                # Trained model artifacts
    ├── best_model.pth     # Deep learning model
    ├── best_model.joblib  # Traditional ML model
    ├── scaler.joblib      # Feature scaler
    └── feature_selector.joblib  # Feature selection model
```

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Exploration
```python
python data_exploration.py
```
Analyzes and visualizes the dataset to understand data distributions and patterns.

### Feature Engineering
```python
python feature_engineering.py
```
Processes raw data and engineers features for model training.

### Model Training

Train traditional ML models:
```python
python traditional_ml.py
```

Train deep learning model:
```python
python deep_learning.py
```

### Evaluation
```python
python evaluate.py
```
Generates comprehensive evaluation metrics and visualization comparing both approaches:
- Confusion matrices
- ROC curves
- Model performance comparison

### Model Deployment
```python
python deployment.py
```
Handles model deployment and inference.

## Model Performance

The evaluation metrics and visualizations are stored in the `metrics/` directory:
- Confusion matrices for both approaches
- ROC curves showing model performance
- Direct model comparison metrics

## Files

- `features.npy`: Processed feature arrays
- `labels.npy`: Target labels
- `feature_importances.npy`: Feature importance scores
- `evaluation_results.json`: Detailed evaluation metrics

## Models

The project maintains both traditional ML and deep learning models:
- Traditional ML model saved as `models/best_model.joblib`
- Deep learning model saved as `models/best_model.pth`
- Feature preprocessing models:
  - `scaler.joblib`: Feature scaling
  - `feature_selector.joblib`: Feature selection