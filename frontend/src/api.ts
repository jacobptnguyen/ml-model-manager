import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: false, // Set to false for CORS (credentials not needed for this app)
  timeout: 300000, // 5 minutes timeout for model training
});

export interface TrainRequest {
  file: File;
  modelType: string;
  modelName: string;
  targetColumn: string;
}

export interface TrainResponse {
  model_id: number;
  model_name: string;
  model_type: string;
  accuracy: number;
  precision: number;
  recall: number;
  roc_auc: number;
}

export interface Model {
  id: number;
  name: string;
  type: string;
  dataset_name: string | null;
  feature_columns: string | null;  // JSON string of feature column names
  sample_data: string | null;  // JSON string of sample feature values
  label_encoders: string | null;  // JSON string of label encoder mappings
  target_encoder: string | null;  // JSON string of target column label encoder classes
  feature_importance: string | null;  // JSON string of feature importance scores
  best_params: string | null;  // JSON string of best hyperparameters from grid search
  best_cv_score: number | null;  // Best cross-validation score from grid search
  accuracy: number | null;
  precision: number | null;
  recall: number | null;
  roc_auc: number | null;
  created_at: string;
  metrics_plot?: string | null;  // Base64-encoded image
  feature_importance_plot?: string | null;  // Base64-encoded feature importance chart
}

export interface PredictRequest {
  model_id: number;
  input_data: Record<string, string | number>;  // Dictionary of column names to original values
}

export interface PredictResponse {
  prediction: string;
  model_id: number;
  timestamp: string;
}

export const uploadAndTrain = async (request: TrainRequest): Promise<TrainResponse> => {
  const formData = new FormData();
  formData.append('file', request.file);
  formData.append('model_type', request.modelType);
  formData.append('model_name', request.modelName);
  formData.append('target_column', request.targetColumn);

  const response = await api.post<TrainResponse>('/api/train', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export const getModels = async (): Promise<Model[]> => {
  const response = await api.get<Model[]>('/api/models');
  return response.data;
};

export const getModel = async (modelId: number): Promise<Model> => {
  const response = await api.get<Model>(`/api/models/${modelId}`);
  return response.data;
};

export const predict = async (request: PredictRequest): Promise<PredictResponse> => {
  const response = await api.post<PredictResponse>('/api/predict', request);
  return response.data;
};

export const updateModel = async (modelId: number, name: string): Promise<Model> => {
  const response = await api.patch<Model>(`/api/models/${modelId}`, { name });
  return response.data;
};

export const deleteModel = async (modelId: number): Promise<void> => {
  await api.delete(`/api/models/${modelId}`);
};

