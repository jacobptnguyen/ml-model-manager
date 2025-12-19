import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pickle
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64

MODELS_DIR = "models"


def ensure_models_dir():
    """Create models directory if it doesn't exist"""
    os.makedirs(MODELS_DIR, exist_ok=True)


def preprocess_data(df: pd.DataFrame, target_column: str):
    """
    Preprocess CSV data: convert to numeric, handle categoricals, drop NaNs, split into train/test (80/20)
    Returns X_train, X_test, y_train, y_test, label_encoders
    """
    # Check if target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in CSV")

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Convert feature columns to numeric where possible
    # For columns that can't be converted, use label encoding
    label_encoders = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            # Try to convert to numeric first
            numeric_col = pd.to_numeric(X[col], errors='coerce')
            if numeric_col.notna().sum() == len(X[col]):
                # All values converted successfully
                X[col] = numeric_col
            else:
                # Use label encoding for categorical columns
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
        else:
            # Try to convert to numeric, coercing errors to NaN
            X[col] = pd.to_numeric(X[col], errors='coerce')

    # Convert target to numeric if needed (for classification)
    target_encoder = None
    
    # Store original target before any processing
    y_original = y.copy()
    
    # Drop rows with NaN values in features first
    mask_features = ~X.isna().any(axis=1)
    X = X[mask_features]
    y_original = y_original[mask_features]
    
    if y_original.dtype == 'object':
        # Use LabelEncoder for target to preserve mapping for inverse transform
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y_original.astype(str))
        y = pd.Series(y_encoded, index=y_original.index)
    else:
        y = pd.to_numeric(y_original, errors='coerce')
        # Drop rows with NaN in target
        mask_target = ~pd.isna(y)
        X = X[mask_target]
        y = y[mask_target]
        y_original = y_original[mask_target]

    if len(X) == 0:
        raise ValueError("No valid numeric data after preprocessing. Please check your CSV file contains numeric or encodable data.")

    # Ensure all feature columns are numeric
    X = X.astype(float)

    # Split into train and test sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )

    return X_train, X_test, y_train, y_test, label_encoders, target_encoder


def train_model(model_type: str, X_train, y_train):
    """
    Train a model with GridSearchCV for hyperparameter tuning
    Returns tuple: (fitted model, grid_search object with best_params and best_score)
    """
    if model_type == "logistic_regression":
        # Parameter grid for Logistic Regression
        # liblinear supports both l1 and l2, lbfgs only supports l2
        param_grid = [
            {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1'],
                'solver': ['liblinear']
            },
            {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l2'],
                'solver': ['liblinear', 'lbfgs']
            }
        ]
        base_model = LogisticRegression(random_state=42, max_iter=1000)
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search
        
    elif model_type == "random_forest":
        # Parameter grid for Random Forest
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search
        
    elif model_type == "xgboost":
        # Parameter grid for XGBoost
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0]
        }
        base_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_feature_importance(model, feature_columns, model_type: str):
    """
    Extract feature importance from trained model
    Returns dictionary mapping feature names to importance scores
    """
    importance_dict = {}
    
    if model_type == "random_forest":
        # Random Forest has feature_importances_ attribute
        importances = model.feature_importances_
        for i, feature in enumerate(feature_columns):
            importance_dict[feature] = float(importances[i])
            
    elif model_type == "xgboost":
        # XGBoost has feature_importances_ attribute
        importances = model.feature_importances_
        for i, feature in enumerate(feature_columns):
            importance_dict[feature] = float(importances[i])
            
    elif model_type == "logistic_regression":
        # For Logistic Regression, use absolute value of coefficients
        # Average across all classes for multi-class
        if len(model.coef_.shape) > 1:
            # Multi-class: average absolute coefficients across classes
            importances = np.mean(np.abs(model.coef_), axis=0)
        else:
            # Binary: use absolute coefficients
            importances = np.abs(model.coef_[0])
        
        for i, feature in enumerate(feature_columns):
            importance_dict[feature] = float(importances[i])
    
    # Normalize to sum to 1
    total = sum(importance_dict.values())
    if total > 0:
        importance_dict = {k: v / total for k, v in importance_dict.items()}
    
    return importance_dict


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model and return metrics: accuracy, precision, recall, ROC-AUC
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    # ROC-AUC requires binary classification or multi-class with probabilities
    try:
        if len(np.unique(y_test)) == 2:
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
    except Exception:
        roc_auc = 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "roc_auc": float(roc_auc)
    }


def save_model(model, filename: str):
    """Save model to disk using pickle"""
    ensure_models_dir()
    filepath = os.path.join(MODELS_DIR, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    return filepath


def load_model(filename: str):
    """Load model from disk"""
    filepath = os.path.join(MODELS_DIR, filename)
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model


def preprocess_prediction_input(input_dict: dict, feature_columns: list, label_encoders: dict):
    """
    Preprocess a single prediction input in original format to match training preprocessing
    input_dict: Dictionary with original column names and values (can be strings or numbers)
    feature_columns: List of feature column names in order
    label_encoders: Dictionary mapping column names to LabelEncoder objects (already reconstructed)
    Returns: numpy array of preprocessed values in correct order
    """
    import pandas as pd
    
    # Process each column the same way as during training
    processed_values = []
    for col in feature_columns:
        if col not in input_dict:
            raise ValueError(f"Missing column: {col}")
        
        value = input_dict[col]
        
        # If column was label encoded during training
        if col in label_encoders:
            le = label_encoders[col]
            # Convert to string and transform
            try:
                encoded_value = le.transform([str(value)])[0]
                processed_values.append(float(encoded_value))
            except ValueError as e:
                available = ', '.join(le.classes_.astype(str))
                raise ValueError(f"Invalid value '{value}' for {col}. Valid options: {available}")
        else:
            # Try to convert to numeric
            if isinstance(value, str):
                numeric_value = pd.to_numeric(value, errors='coerce')
                if pd.isna(numeric_value):
                    raise ValueError(f"Cannot convert {col} value '{value}' to numeric")
            else:
                numeric_value = float(value)
            
            processed_values.append(float(numeric_value))
    
    return np.array(processed_values).reshape(1, -1)


def predict(model, X_input):
    """
    Make prediction on input data
    X_input should be a list or array of feature values
    """
    if isinstance(X_input, list):
        X_input = np.array(X_input).reshape(1, -1)
    else:
        X_input = np.array(X_input).reshape(1, -1)

    prediction = model.predict(X_input)[0]
    return str(prediction)


def generate_metrics_plot(accuracy, precision, recall, roc_auc):
    """
    Generate a matplotlib bar chart of performance metrics
    Returns base64-encoded image string
    """
    # Filter out None values and prepare data
    metrics = []
    labels = []
    colors = []
    
    if accuracy is not None:
        metrics.append(accuracy * 100)
        labels.append('Accuracy')
        colors.append('#28a745')
    if precision is not None:
        metrics.append(precision * 100)
        labels.append('Precision')
        colors.append('#007bff')
    if recall is not None:
        metrics.append(recall * 100)
        labels.append('Recall')
        colors.append('#ffc107')
    if roc_auc is not None:
        metrics.append(roc_auc * 100)
        labels.append('ROC-AUC')
        colors.append('#dc3545')
    
    if not metrics:
        return None
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, metrics, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Add value labels on top of bars
    for bar, value in zip(bars, metrics):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Customize the plot
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, max(metrics) * 1.15 if metrics else 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return f"data:image/png;base64,{img_base64}"


def generate_feature_importance_plot(feature_importance: dict):
    """
    Generate a matplotlib horizontal bar chart of feature importance
    Returns base64-encoded image string
    """
    if not feature_importance:
        return None
    
    # Sort features by importance (descending)
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    # Take top 15 features to avoid overcrowding
    top_features = sorted_features[:15]
    
    features = [f[0] for f in top_features]
    importances = [f[1] * 100 for f in top_features]  # Convert to percentage
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.4)))
    bars = ax.barh(features, importances, color='#007bff', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, importances)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{value:.2f}%',
                ha='left', va='center', fontsize=10, fontweight='bold', pad=5)
    
    # Customize the plot
    ax.set_xlabel('Importance (%)', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, max(importances) * 1.2 if importances else 100)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Invert y-axis to show highest importance at top
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return f"data:image/png;base64,{img_base64}"

