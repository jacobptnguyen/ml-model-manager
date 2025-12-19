from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class TrainRequest(BaseModel):
    model_type: str  # logistic_regression, random_forest, xgboost
    model_name: str
    target_column: str


class TrainResponse(BaseModel):
    model_id: int
    model_name: str
    model_type: str
    accuracy: float
    precision: float
    recall: float
    roc_auc: float

    class Config:
        from_attributes = True


class ModelResponse(BaseModel):
    id: int
    name: str
    type: str
    dataset_name: Optional[str] = None
    feature_columns: Optional[str] = None  # JSON string of feature column names
    sample_data: Optional[str] = None  # JSON string of sample feature values
    label_encoders: Optional[str] = None  # JSON string of label encoder mappings
    target_encoder: Optional[str] = None  # JSON string of target column label encoder classes
    feature_importance: Optional[str] = None  # JSON string of feature importance scores
    best_params: Optional[str] = None  # JSON string of best hyperparameters from grid search
    best_cv_score: Optional[float] = None  # Best cross-validation score from grid search
    accuracy: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    roc_auc: Optional[float]
    created_at: datetime
    metrics_plot: Optional[str] = None  # Base64-encoded image
    feature_importance_plot: Optional[str] = None  # Base64-encoded feature importance chart

    class Config:
        from_attributes = True


class PredictRequest(BaseModel):
    model_id: int
    input_data: dict[str, str | float]  # Dictionary of column names to original values (strings or numbers)


class PredictResponse(BaseModel):
    prediction: str
    model_id: int
    timestamp: datetime


class ModelUpdate(BaseModel):
    name: str

