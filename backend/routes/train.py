from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np
import json
from database import get_db
from models import Model
from schemas import TrainResponse
from ml_pipeline import preprocess_data, train_model, evaluate_model, save_model, get_feature_importance, generate_feature_importance_plot
import uuid
import os

router = APIRouter()


@router.post("/train", response_model=TrainResponse)
async def train(
    file: UploadFile = File(...),
    model_type: str = Form(...),
    model_name: str = Form(...),
    target_column: str = Form(...),
    db: Session = Depends(get_db)
):
    """Train a model from uploaded CSV file"""
    
    # Validate model type
    valid_model_types = ["logistic_regression", "random_forest", "xgboost"]
    if model_type not in valid_model_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model type. Must be one of: {', '.join(valid_model_types)}"
        )

    # Validate file type - check both filename and content type
    if not file.filename or not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV file (.csv extension required)")
    
    # Check content type if provided
    if file.content_type and file.content_type not in ['text/csv', 'application/csv', 'text/plain', 'application/vnd.ms-excel']:
        # Some browsers/systems may not send correct MIME type, so we'll still try to read it
        # but log a warning
        pass

    try:
        # Read CSV file - this will fail if file is not actually CSV
        df = pd.read_csv(file.file)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")

        # Get original sample data from first valid row BEFORE preprocessing (to preserve text values)
        # This will be used for placeholders in the prediction form
        original_feature_columns = [col for col in df.columns if col != target_column]
        original_sample_data = {}
        
        # Find first row with at least some valid data in feature columns
        for idx in range(len(df)):
            sample_row = df.iloc[idx]
            # Check if this row has any valid (non-NaN) values in feature columns
            valid_cols = [col for col in original_feature_columns if not pd.isna(sample_row[col])]
            if len(valid_cols) > 0:
                # Collect all valid values from this row
                for col in original_feature_columns:
                    value = sample_row[col]
                    if not pd.isna(value):
                        # Keep original value type (string, number, etc.)
                        if isinstance(value, (int, float, np.integer, np.floating)):
                            original_sample_data[col] = float(value)
                        else:
                            original_sample_data[col] = str(value)
                break

        # Preprocess data
        X_train, X_test, y_train, y_test, label_encoders, target_encoder = preprocess_data(df, target_column)

        # Get feature column names (after preprocessing, columns are in order)
        feature_columns = list(X_train.columns)

        # Store sample data with original values (text preserved)
        sample_data = json.dumps(original_sample_data)
        
        # Store label encoders mapping (column name -> list of classes for that encoder)
        # This allows us to recreate the encoding during prediction
        label_encoders_data = {}
        for col, le in label_encoders.items():
            # Store the classes (unique values) for each encoder
            label_encoders_data[col] = le.classes_.tolist()
        label_encoders_json = json.dumps(label_encoders_data)
        
        # Store target encoder mapping for decoding predictions
        target_encoder_data = None
        if target_encoder is not None:
            target_encoder_data = target_encoder.classes_.tolist()
        target_encoder_json = json.dumps(target_encoder_data) if target_encoder_data else None

        # Train model (returns model and grid_search object)
        model, grid_search = train_model(model_type, X_train, y_train)

        # Extract grid search results
        best_params = grid_search.best_params_
        best_cv_score = grid_search.best_score_
        best_params_json = json.dumps(best_params)

        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)

        # Calculate feature importance
        feature_importance = get_feature_importance(model, feature_columns, model_type)
        feature_importance_json = json.dumps(feature_importance)

        # Save model to disk
        model_filename = f"{model_name}_{uuid.uuid4().hex[:8]}.pkl"
        save_model(model, model_filename)

        # Save model metadata to database
        db_model = Model(
            name=model_name,
            type=model_type,
            model_filename=model_filename,
            dataset_name=file.filename,  # Store original CSV filename
            feature_columns=json.dumps(feature_columns),  # Store feature column names as JSON
            sample_data=sample_data,  # Store sample values for placeholders
            label_encoders=label_encoders_json,  # Store label encoder mappings
            target_encoder=target_encoder_json,  # Store target encoder for decoding predictions
            feature_importance=feature_importance_json,  # Store feature importance
            best_params=best_params_json,  # Store best hyperparameters from grid search
            best_cv_score=best_cv_score,  # Store best cross-validation score
            accuracy=metrics["accuracy"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            roc_auc=metrics["roc_auc"]
        )
        db.add(db_model)
        db.commit()
        db.refresh(db_model)

        return TrainResponse(
            model_id=db_model.id,
            model_name=db_model.name,
            model_type=db_model.type,
            accuracy=db_model.accuracy,
            precision=db_model.precision,
            recall=db_model.recall,
            roc_auc=db_model.roc_auc
        )

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty or contains no data")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}. Please ensure the file is a valid CSV file.")
    except UnicodeDecodeError as e:
        raise HTTPException(status_code=400, detail=f"File encoding error: {str(e)}. Please ensure the CSV file uses UTF-8 encoding.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")

