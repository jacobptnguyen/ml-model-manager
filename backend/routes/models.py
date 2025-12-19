from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from database import get_db
from models import Model
from schemas import ModelResponse, ModelUpdate
from ml_pipeline import generate_metrics_plot
import json

router = APIRouter()


@router.get("/models", response_model=List[ModelResponse])
async def get_models(db: Session = Depends(get_db)):
    """Get all trained models"""
    models = db.query(Model).order_by(Model.created_at.desc()).all()
    return models


@router.get("/models/{model_id}", response_model=ModelResponse)
async def get_model(model_id: int, db: Session = Depends(get_db)):
    """Get a specific model by ID"""
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Generate metrics plot (always generate)
    metrics_plot = generate_metrics_plot(
        model.accuracy,
        model.precision,
        model.recall,
        model.roc_auc
    )
    
    # Create response with plot
    response = ModelResponse(
        id=model.id,
        name=model.name,
        type=model.type,
        dataset_name=model.dataset_name,
        feature_columns=model.feature_columns,
        sample_data=model.sample_data,
        label_encoders=model.label_encoders,
        target_encoder=model.target_encoder,
        feature_importance=model.feature_importance,
        best_params=model.best_params,
        best_cv_score=model.best_cv_score,
        accuracy=model.accuracy,
        precision=model.precision,
        recall=model.recall,
        roc_auc=model.roc_auc,
        created_at=model.created_at,
        metrics_plot=metrics_plot,
        feature_importance_plot=None
    )
    
    return response


@router.patch("/models/{model_id}", response_model=ModelResponse)
async def update_model(model_id: int, model_update: ModelUpdate, db: Session = Depends(get_db)):
    """Update a model's name"""
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if model_update.name:
        model.name = model_update.name
    
    db.commit()
    db.refresh(model)
    
    return ModelResponse(
        id=model.id,
        name=model.name,
        type=model.type,
        dataset_name=model.dataset_name,
        feature_columns=model.feature_columns,
        sample_data=model.sample_data,
        label_encoders=model.label_encoders,
        target_encoder=model.target_encoder,
        feature_importance=model.feature_importance,
        best_params=model.best_params,
        best_cv_score=model.best_cv_score,
        accuracy=model.accuracy,
        precision=model.precision,
        recall=model.recall,
        roc_auc=model.roc_auc,
        created_at=model.created_at,
        metrics_plot=None,
        feature_importance_plot=None
    )


@router.delete("/models/{model_id}")
async def delete_model(model_id: int, db: Session = Depends(get_db)):
    """Delete a model"""
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        # Delete associated predictions first (if any)
        from models import Prediction
        db.query(Prediction).filter(Prediction.model_id == model_id).delete()
        
        # Delete the model
        db.delete(model)
        db.commit()
        
        return {"message": "Model deleted successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting model: {str(e)}")

