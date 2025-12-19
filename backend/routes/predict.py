from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from models import Model, Prediction
from schemas import PredictRequest, PredictResponse
from ml_pipeline import load_model, predict, preprocess_prediction_input
from sklearn.preprocessing import LabelEncoder
import numpy as np
import json
from datetime import datetime
import os

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
async def make_prediction(
    request: PredictRequest,
    db: Session = Depends(get_db)
):
    """Make a prediction using a trained model"""
    
    # Get model from database
    model_record = db.query(Model).filter(Model.id == request.model_id).first()
    if not model_record:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        # Load model using filename stored in database
        model = load_model(model_record.model_filename)

        # Get feature columns and label encoders
        if not model_record.feature_columns:
            raise HTTPException(status_code=400, detail="Model missing feature columns information")
        
        feature_columns = json.loads(model_record.feature_columns)
        
        # Reconstruct label encoders from stored data
        label_encoders = {}
        if model_record.label_encoders:
            label_encoders_data = json.loads(model_record.label_encoders)
            for col, classes in label_encoders_data.items():
                le = LabelEncoder()
                le.classes_ = np.array(classes)
                label_encoders[col] = le

        # Preprocess input data from original format
        preprocessed_input = preprocess_prediction_input(
            request.input_data,
            feature_columns,
            label_encoders
        )

        # Make prediction
        prediction_encoded = predict(model, preprocessed_input)
        
        # Decode prediction back to original format if target encoder exists
        prediction_value = prediction_encoded
        if model_record.target_encoder:
            try:
                target_classes = json.loads(model_record.target_encoder)
                if target_classes:
                    prediction_int = int(prediction_encoded)
                    if 0 <= prediction_int < len(target_classes):
                        prediction_value = target_classes[prediction_int]
            except Exception:
                # If decoding fails, use the encoded value
                pass

        # Save prediction to database (store original decoded value)
        prediction_record = Prediction(
            model_id=request.model_id,
            input_data=json.dumps(request.input_data),
            prediction=prediction_value
        )
        db.add(prediction_record)
        db.commit()

        return PredictResponse(
            prediction=prediction_value,
            model_id=request.model_id,
            timestamp=prediction_record.timestamp
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

