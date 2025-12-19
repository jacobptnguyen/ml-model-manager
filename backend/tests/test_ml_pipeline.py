import pytest
import pandas as pd
import numpy as np
from ml_pipeline import preprocess_data, train_model, evaluate_model, save_model, load_model, predict
import os
import tempfile
import shutil


@pytest.fixture
def sample_data():
    """Create sample CSV data for testing"""
    data = {
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        'target': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    }
    return pd.DataFrame(data)


def test_preprocess_data(sample_data):
    """Test data preprocessing"""
    X_train, X_test, y_train, y_test = preprocess_data(sample_data, 'target')
    
    assert len(X_train) == 8  # 80% of 10
    assert len(X_test) == 2   # 20% of 10
    assert len(y_train) == 8
    assert len(y_test) == 2
    assert 'target' not in X_train.columns
    assert 'target' not in X_test.columns


def test_train_model(sample_data):
    """Test model training"""
    X_train, X_test, y_train, y_test = preprocess_data(sample_data, 'target')
    
    model, grid_search = train_model('logistic_regression', X_train, y_train)
    assert model is not None
    assert grid_search is not None
    assert hasattr(grid_search, 'best_params_')
    assert hasattr(grid_search, 'best_score_')
    
    model_rf, _ = train_model('random_forest', X_train, y_train)
    assert model_rf is not None
    
    model_xgb, _ = train_model('xgboost', X_train, y_train)
    assert model_xgb is not None


def test_evaluate_model(sample_data):
    """Test model evaluation"""
    X_train, X_test, y_train, y_test = preprocess_data(sample_data, 'target')
    model, _ = train_model('logistic_regression', X_train, y_train)
    
    metrics = evaluate_model(model, X_test, y_test)
    
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'roc_auc' in metrics
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1


def test_save_and_load_model(sample_data):
    """Test saving and loading models"""
    X_train, X_test, y_train, y_test = preprocess_data(sample_data, 'target')
    model, _ = train_model('logistic_regression', X_train, y_train)
    
    # Save model
    filename = save_model(model, 'test_model.pkl')
    assert os.path.exists(filename)
    
    # Load model
    loaded_model = load_model('test_model.pkl')
    assert loaded_model is not None
    
    # Clean up
    os.remove(filename)


def test_predict(sample_data):
    """Test prediction"""
    X_train, X_test, y_train, y_test = preprocess_data(sample_data, 'target')
    model, _ = train_model('logistic_regression', X_train, y_train)
    
    # Make prediction
    input_data = [5, 10]
    prediction = predict(model, input_data)
    assert prediction is not None
    assert isinstance(prediction, str)

