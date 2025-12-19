from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base


class Model(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)  # logistic_regression, random_forest, xgboost
    model_filename = Column(String, nullable=False)  # Filename of saved model
    dataset_name = Column(String, nullable=True)  # Original CSV filename
    feature_columns = Column(Text, nullable=True)  # JSON string of feature column names
    sample_data = Column(Text, nullable=True)  # JSON string of sample feature values for placeholders
    label_encoders = Column(Text, nullable=True)  # JSON string mapping column names to label encoder classes
    target_encoder = Column(Text, nullable=True)  # JSON string of target column label encoder classes
    feature_importance = Column(Text, nullable=True)  # JSON string mapping feature names to importance scores
    best_params = Column(Text, nullable=True)  # JSON string of best hyperparameters from grid search
    best_cv_score = Column(Float, nullable=True)  # Best cross-validation score from grid search
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    roc_auc = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    predictions = relationship("Prediction", back_populates="model")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    input_data = Column(Text, nullable=False)
    prediction = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    model = relationship("Model", back_populates="predictions")

