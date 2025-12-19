import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { getModel, predict, PredictRequest } from '../api';

const ModelDetail = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [model, setModel] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [featureInputs, setFeatureInputs] = useState<Record<string, string>>({});
  const [validationErrors, setValidationErrors] = useState<Record<string, string>>({});
  const [hasAttemptedSubmit, setHasAttemptedSubmit] = useState(false);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [predictError, setPredictError] = useState<string | null>(null);

  useEffect(() => {
    const fetchModel = async () => {
      if (!id) return;
      
      setLoading(true);
      setError(null);
      try {
        const data = await getModel(parseInt(id));
        setModel(data);
        
        // Initialize feature inputs if feature columns are available
        if (data.feature_columns) {
          try {
            const columns = JSON.parse(data.feature_columns);
            const initialInputs: Record<string, string> = {};
            columns.forEach((col: string) => {
              initialInputs[col] = '';
            });
            setFeatureInputs(initialInputs);
          } catch (e) {
            // If parsing fails, feature_columns might be in wrong format
            console.error('Failed to parse feature columns:', e);
          }
        }
      } catch (err: any) {
        setError(err.response?.data?.detail || 'Failed to fetch model');
      } finally {
        setLoading(false);
      }
    };

    fetchModel();
  }, [id]);

  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!id || !model) return;

    setHasAttemptedSubmit(true);
    setPredictError(null);
    setPrediction(null);

    // Validate all inputs
    let labelEncoders: Record<string, string[]> = {};
    let sampleData: Record<string, string | number> = {};
    
    if (model.label_encoders) {
      try {
        labelEncoders = JSON.parse(model.label_encoders);
      } catch (e) {
        // If parsing fails, assume no label encoders
      }
    }
    
    if (model.sample_data) {
      try {
        sampleData = JSON.parse(model.sample_data);
      } catch (e) {
        // If parsing fails, continue without sample data
      }
    }

    // Validate all fields
    const errors: Record<string, string> = {};
    let featureColumns: string[] = [];
    if (model.feature_columns) {
      try {
        featureColumns = JSON.parse(model.feature_columns);
      } catch (e) {
        setPredictError('Invalid feature columns data');
        return;
      }
    }

    for (const col of featureColumns) {
      const value = featureInputs[col];
      if (!value || value.trim() === '') {
        errors[col] = 'This field is required';
        continue;
      }
      
      const error = validateInput(col, value, labelEncoders, sampleData);
      if (error) {
        errors[col] = error;
      }
    }

    setValidationErrors(errors);

    // If there are validation errors, don't proceed
    if (Object.keys(errors).length > 0) {
      return;
    }

    setPredicting(true);

    try {
      // Build input data dictionary with original format values
      const inputData: Record<string, string | number> = {};
      for (const col of featureColumns) {
        const value = featureInputs[col];
        
        // If column has label encoder, keep as string (categorical)
        if (col in labelEncoders) {
          inputData[col] = value.trim();
        } else {
          // For numeric columns, convert to number
          const numValue = parseFloat(value.trim());
          inputData[col] = numValue;
        }
      }

      const request: PredictRequest = {
        model_id: parseInt(id),
        input_data: inputData,
      };

      const response = await predict(request);
      setPrediction(response.prediction);
    } catch (err: any) {
      setPredictError(err.response?.data?.detail || err.message || 'Failed to make prediction');
    } finally {
      setPredicting(false);
    }
  };

  const validateInput = (column: string, value: string, labelEncoders: Record<string, string[]>, sampleData: Record<string, string | number>): string | null => {
    if (!value || value.trim() === '') {
      return null; // Empty is okay, will be caught by required attribute
    }

    // Check if column is categorical
    if (column in labelEncoders) {
      const validOptions = labelEncoders[column];
      if (!validOptions.includes(value.trim())) {
        // Show only one example
        const example = validOptions[0];
        return `Invalid value. Example: "${example}"`;
      }
    } else {
      // For numeric columns, check if it's a valid number
      const numValue = parseFloat(value.trim());
      if (isNaN(numValue)) {
        // Show example from sample data if available
        const example = sampleData[column];
        if (example !== undefined && typeof example === 'number') {
          return `Must be a number. Example: ${example.toFixed(2)}`;
        }
        return 'Must be a number';
      }
    }

    return null; // Valid
  };

  const handleFeatureInputChange = (column: string, value: string) => {
    setFeatureInputs(prev => ({
      ...prev,
      [column]: value
    }));
    // Clear validation error for this field when user types
    setValidationErrors(prev => ({
      ...prev,
      [column]: ''
    }));
  };

  const hasValidationErrors = (): boolean => {
    return Object.values(validationErrors).some(error => error !== '');
  };

  const allFieldsFilled = (): boolean => {
    if (!model?.feature_columns) return false;
    try {
      const columns = JSON.parse(model.feature_columns);
      return columns.every((col: string) => featureInputs[col] && featureInputs[col].trim() !== '');
    } catch {
      return false;
    }
  };

  const formatMetric = (value: number | null) => {
    if (value === null) return 'N/A';
    return (value * 100).toFixed(2) + '%';
  };

  const formatModelType = (type: string) => {
    return type
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  if (loading) {
    return <div style={{ padding: '2rem', textAlign: 'center' }}>Loading...</div>;
  }

  if (error || !model) {
    return (
      <div style={{ padding: '2rem' }}>
        <div style={{ padding: '1rem', backgroundColor: '#f8d7da', color: '#721c24', borderRadius: '4px' }}>
          {error || 'Model not found'}
        </div>
        <button
          onClick={() => navigate('/')}
          style={{
            marginTop: '1rem',
            padding: '0.5rem 1rem',
            backgroundColor: '#6c757d',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
          }}
        >
          ← Back
        </button>
      </div>
    );
  }

  return (
    <div style={{ maxWidth: '1400px', margin: '0 auto', padding: '2rem' }}>
      <button
        onClick={() => navigate('/')}
        style={{
          marginBottom: '2rem',
          padding: '0.5rem 1rem',
          backgroundColor: 'transparent',
          color: '#6c757d',
          border: '1px solid #6c757d',
          borderRadius: '4px',
          cursor: 'pointer',
        }}
      >
        ← Back
      </button>

      {/* Two-column layout for desktop */}
      <style>{`
        @media (max-width: 1023px) {
          .model-detail-grid {
            grid-template-columns: 1fr !important;
          }
          .sticky-prediction {
            position: relative !important;
            top: 0 !important;
          }
        }
      `}</style>
      <div className="model-detail-grid" style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))',
        gap: '2rem',
        alignItems: 'start'
      }}>
        {/* Left Column: Model Information and Metrics */}
        <div>
          <h1 style={{ marginBottom: '2rem' }}>{model.name}</h1>
          
          <div style={{ marginBottom: '2rem' }}>
            <h2>Model Information</h2>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginTop: '1rem' }}>
              <div>
                <strong>Dataset:</strong> {model.dataset_name || 'N/A'}
              </div>
              <div>
                <strong>Type:</strong> {formatModelType(model.type)}
              </div>
            </div>
          </div>

          <div style={{ marginBottom: '2rem' }}>
            <h2>Performance Metrics</h2>
            
        {/* Matplotlib Visualization */}
        {model.metrics_plot && (
          <div style={{ marginTop: '1rem', marginBottom: '2rem', textAlign: 'center' }}>
            <img 
              src={model.metrics_plot} 
              alt="Performance Metrics Chart" 
              style={{ 
                maxWidth: '100%', 
                height: 'auto',
                borderRadius: '8px',
                boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
              }} 
            />
          </div>
        )}

        {/* Grid Search Results */}
        {model.best_params && (
          <div style={{ marginTop: '1rem', marginBottom: '2rem' }}>
            <h3 style={{ marginBottom: '1rem', fontSize: '1.1rem' }}>Hyperparameters (Grid Search Results)</h3>
            <div style={{ 
              backgroundColor: 'white', 
              borderRadius: '4px', 
              padding: '1rem',
              border: '1px solid #dee2e6'
            }}>
              {model.best_cv_score !== null && (
                <div style={{ marginBottom: '1rem', paddingBottom: '1rem', borderBottom: '1px solid #dee2e6' }}>
                  <strong style={{ color: '#495057' }}>Best CV Score:</strong>
                  <span style={{ marginLeft: '0.5rem', fontSize: '1.1rem', fontWeight: 'bold', color: '#28a745' }}>
                    {formatMetric(model.best_cv_score)}
                  </span>
                </div>
              )}
              <div>
                <strong style={{ color: '#495057', display: 'block', marginBottom: '0.5rem' }}>Best Parameters:</strong>
                <div style={{ 
                  backgroundColor: '#f8f9fa', 
                  padding: '0.75rem', 
                  borderRadius: '4px',
                  fontFamily: 'monospace',
                  fontSize: '0.9rem'
                }}>
                  {(() => {
                    try {
                      const params = JSON.parse(model.best_params);
                      return Object.entries(params).map(([key, value]) => (
                        <div key={key} style={{ marginBottom: '0.25rem' }}>
                          <span style={{ color: '#6c757d' }}>{key}:</span>{' '}
                          <span style={{ color: '#212529', fontWeight: '500' }}>
                            {value === null ? 'None' : String(value)}
                          </span>
                        </div>
                      ));
                    } catch (e) {
                      return <div style={{ color: '#dc3545' }}>Error parsing parameters</div>;
                    }
                  })()}
                </div>
              </div>
            </div>
          </div>
        )}

            {/* Classification Report Style Table */}
            <div style={{ marginTop: '1rem' }}>
              <h3 style={{ marginBottom: '1rem', fontSize: '1.1rem' }}>Classification Report</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse', backgroundColor: 'white', borderRadius: '4px', overflow: 'hidden' }}>
            <thead>
              <tr style={{ backgroundColor: '#343a40', color: 'white' }}>
                <th style={{ padding: '0.75rem', textAlign: 'left', border: '1px solid #dee2e6' }}>Metric</th>
                <th style={{ padding: '0.75rem', textAlign: 'center', border: '1px solid #dee2e6' }}>Score</th>
              </tr>
            </thead>
            <tbody>
              <tr style={{ borderBottom: '1px solid #dee2e6' }}>
                <td style={{ padding: '0.75rem', border: '1px solid #dee2e6', fontWeight: '500' }}>Accuracy</td>
                <td style={{ padding: '0.75rem', textAlign: 'center', border: '1px solid #dee2e6', fontWeight: 'bold' }}>
                  {formatMetric(model.accuracy)}
                </td>
              </tr>
              <tr style={{ borderBottom: '1px solid #dee2e6' }}>
                <td style={{ padding: '0.75rem', border: '1px solid #dee2e6', fontWeight: '500' }}>Precision</td>
                <td style={{ padding: '0.75rem', textAlign: 'center', border: '1px solid #dee2e6', fontWeight: 'bold' }}>
                  {formatMetric(model.precision)}
                </td>
              </tr>
              <tr style={{ borderBottom: '1px solid #dee2e6' }}>
                <td style={{ padding: '0.75rem', border: '1px solid #dee2e6', fontWeight: '500' }}>Recall</td>
                <td style={{ padding: '0.75rem', textAlign: 'center', border: '1px solid #dee2e6', fontWeight: 'bold' }}>
                  {formatMetric(model.recall)}
                </td>
              </tr>
              <tr>
                <td style={{ padding: '0.75rem', border: '1px solid #dee2e6', fontWeight: '500' }}>ROC-AUC</td>
                <td style={{ padding: '0.75rem', textAlign: 'center', border: '1px solid #dee2e6', fontWeight: 'bold' }}>
                  {formatMetric(model.roc_auc)}
                </td>
              </tr>
            </tbody>
          </table>
            </div>
          </div>
        </div>

        {/* Right Column: Prediction Form (Sticky) */}
        <div className="sticky-prediction" style={{ 
          position: 'sticky',
          top: '2rem',
          alignSelf: 'start'
        }}>
          <div style={{
            backgroundColor: '#f8f9fa',
            padding: '2rem',
            borderRadius: '8px',
            border: '1px solid #dee2e6',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
          }}>
            <h2 style={{ marginBottom: '1.5rem' }}>Make Prediction</h2>
            <style>{`
              input[type="number"]::-webkit-inner-spin-button,
              input[type="number"]::-webkit-outer-spin-button {
                -webkit-appearance: none;
                margin: 0;
              }
              input[type="number"] {
                -moz-appearance: textfield;
              }
            `}</style>
        <style>{`
          input[type="number"]::-webkit-inner-spin-button,
          input[type="number"]::-webkit-outer-spin-button {
            -webkit-appearance: none;
            margin: 0;
          }
          input[type="number"] {
            -moz-appearance: textfield;
          }
        `}</style>
        {model.feature_columns ? (
          <form onSubmit={handlePredict} style={{ marginTop: '1rem' }}>
            <div style={{ display: 'grid', gap: '1rem', marginBottom: '1.5rem' }}>
              {(() => {
                try {
                  const columns = JSON.parse(model.feature_columns);
                  let sampleData: Record<string, string | number> = {};
                  if (model.sample_data) {
                    try {
                      sampleData = JSON.parse(model.sample_data);
                    } catch (e) {
                      // If sample_data parsing fails, continue without it
                    }
                  }
                  
                  let labelEncoders: Record<string, string[]> = {};
                  if (model.label_encoders) {
                    try {
                      labelEncoders = JSON.parse(model.label_encoders);
                    } catch (e) {
                      // If parsing fails, assume no label encoders
                    }
                  }
                  
                  return columns.map((column: string) => {
                    const sampleValue = sampleData[column];
                    const isCategorical = column in labelEncoders;
                    const availableOptions = isCategorical ? labelEncoders[column] : null;
                    let placeholder = `Enter value for ${column}`;
                    
                    // Build placeholder with valid options for categorical columns
                    if (isCategorical && availableOptions) {
                      placeholder = `e.g., "${availableOptions[0]}"`;
                    } else if (sampleValue !== undefined && sampleValue !== null) {
                      if (typeof sampleValue === 'string') {
                        placeholder = `e.g., "${sampleValue}"`;
                      } else if (typeof sampleValue === 'number') {
                        placeholder = `e.g., ${sampleValue.toFixed(2)}`;
                      }
                    }
                    
                    const hasError = validationErrors[column] && validationErrors[column] !== '';
                    
                    return (
                      <div key={column}>
                        <label 
                          htmlFor={`feature-${column}`} 
                          style={{ 
                            display: 'block', 
                            marginBottom: '0.5rem',
                            fontWeight: '500',
                            fontSize: '0.95rem'
                          }}
                        >
                          {column}:
                        </label>
                        <input
                          type={isCategorical ? "text" : "number"}
                          step={isCategorical ? undefined : "any"}
                          id={`feature-${column}`}
                          value={featureInputs[column] || ''}
                          onChange={(e) => handleFeatureInputChange(column, e.target.value)}
                          placeholder={placeholder}
                          style={{ 
                            width: '100%', 
                            padding: '0.75rem', 
                            fontSize: '1rem',
                            border: hasError ? '2px solid #dc3545' : '1px solid #dee2e6',
                            borderRadius: '4px',
                            boxSizing: 'border-box',
                            backgroundColor: hasError ? '#fff5f5' : 'white'
                          }}
                          required
                        />
                        {hasError && (
                          <div style={{ 
                            marginTop: '0.25rem', 
                            fontSize: '0.85rem', 
                            color: '#dc3545'
                          }}>
                            {validationErrors[column]}
                          </div>
                        )}
                      </div>
                    );
                  });
                } catch (e) {
                  return (
                    <div style={{ padding: '1rem', backgroundColor: '#f8d7da', color: '#721c24', borderRadius: '4px' }}>
                      Unable to load feature columns. Please use comma-separated values format.
                    </div>
                  );
                }
              })()}
            </div>
            <button
              type="submit"
              disabled={predicting || !allFieldsFilled()}
              style={{
                padding: '0.75rem 1.5rem',
                backgroundColor: (predicting || !allFieldsFilled()) ? '#ccc' : '#007bff',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: (predicting || !allFieldsFilled()) ? 'not-allowed' : 'pointer',
                fontSize: '1rem',
                fontWeight: '500',
              }}
            >
              {predicting ? 'Predicting...' : 'Predict'}
            </button>
            {hasAttemptedSubmit && (hasValidationErrors() || !allFieldsFilled()) && (
              <div style={{ 
                marginTop: '0.5rem', 
                fontSize: '0.9rem', 
                color: '#856404',
                fontStyle: 'italic'
              }}>
                {!allFieldsFilled() && 'Please fill in all fields. '}
                {hasValidationErrors() && 'Please fix validation errors before predicting.'}
              </div>
            )}
          </form>
        ) : (
          <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#fff3cd', color: '#856404', borderRadius: '4px' }}>
            Feature columns not available for this model. This model was created before feature column tracking was added.
          </div>
        )}

            {predictError && (
              <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#f8d7da', color: '#721c24', borderRadius: '4px' }}>
                {predictError}
              </div>
            )}

            {prediction !== null && (
              <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#d4edda', color: '#155724', borderRadius: '4px' }}>
                <strong>Prediction:</strong> {prediction}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelDetail;

