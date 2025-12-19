import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { uploadAndTrain } from '../api';

const Upload = () => {
  const [file, setFile] = useState<File | null>(null);
  const [modelType, setModelType] = useState('logistic_regression');
  const [modelName, setModelName] = useState('');
  const [targetColumn, setTargetColumn] = useState('');
  const [columnNames, setColumnNames] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [parsing, setParsing] = useState(false);
  const [trainingStartTime, setTrainingStartTime] = useState<number | null>(null);
  const [elapsedTime, setElapsedTime] = useState<number>(0);
  const navigate = useNavigate();

  const parseCSVColumns = async (file: File): Promise<string[]> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const text = e.target?.result as string;
          // Remove BOM if present
          const cleanText = text.replace(/^\uFEFF/, '');
          // Get first line (headers) - handle both \n and \r\n
          const firstLine = cleanText.split(/\r?\n/)[0];
          
          // Parse CSV line properly handling quoted fields
          const columns: string[] = [];
          let current = '';
          let inQuotes = false;
          
          for (let i = 0; i < firstLine.length; i++) {
            const char = firstLine[i];
            
            if (char === '"') {
              if (inQuotes && firstLine[i + 1] === '"') {
                // Escaped quote
                current += '"';
                i++; // Skip next quote
              } else {
                // Toggle quote state
                inQuotes = !inQuotes;
              }
            } else if (char === ',' && !inQuotes) {
              // End of column
              columns.push(current.trim());
              current = '';
            } else {
              current += char;
            }
          }
          
          // Add last column
          if (current.length > 0 || firstLine.endsWith(',')) {
            columns.push(current.trim());
          }
          
          // Filter out empty columns and clean up
          const cleanedColumns = columns
            .map(col => col.replace(/^"|"$/g, '').trim())
            .filter(col => col.length > 0);
          
          if (cleanedColumns.length === 0) {
            reject(new Error('No columns found in CSV file'));
          } else {
            resolve(cleanedColumns);
          }
        } catch (err) {
          reject(err);
        }
      };
      reader.onerror = () => reject(new Error('Failed to read file'));
      // Only read first 2KB to get headers (in case of very long header line)
      reader.readAsText(file.slice(0, 2048));
    });
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0] || null;
    setFile(selectedFile);
    setTargetColumn(''); // Reset target column when file changes
    setError(null);

    if (selectedFile && selectedFile.name.endsWith('.csv')) {
      setParsing(true);
      try {
        const columns = await parseCSVColumns(selectedFile);
        setColumnNames(columns);
        if (columns.length > 0) {
          // Auto-select first column as default (user can change)
          setTargetColumn(columns[0]);
        }
      } catch (err) {
        setError('Failed to parse CSV file. Please ensure it is a valid CSV.');
        setColumnNames([]);
      } finally {
        setParsing(false);
      }
    } else {
      setColumnNames([]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!file) {
      setError('Please select a CSV file');
      return;
    }
    
    // Validate file type on frontend
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setError('Please select a valid CSV file (.csv extension required)');
      return;
    }
    
    if (!modelName.trim()) {
      setError('Please enter a model name');
      return;
    }
    
    if (!targetColumn.trim()) {
      setError('Please enter the target column name');
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(false);
    setElapsedTime(0);
    const startTime = Date.now();
    setTrainingStartTime(startTime);

    // Update elapsed time every second
    let intervalId: NodeJS.Timeout | null = null;
    
    try {
      intervalId = setInterval(() => {
        setElapsedTime(Math.floor((Date.now() - startTime) / 1000));
      }, 1000);

      const response = await uploadAndTrain({
        file,
        modelType,
        modelName,
        targetColumn,
      });
      
      if (intervalId) {
        clearInterval(intervalId);
      }
      setSuccess(true);
      setTimeout(() => {
        navigate('/dashboard');
      }, 2000);
    } catch (err: any) {
      if (intervalId) {
        clearInterval(intervalId);
      }
      setError(err.response?.data?.detail || 'Failed to train model');
    } finally {
      setLoading(false);
      setTrainingStartTime(null);
      if (intervalId) {
        clearInterval(intervalId);
      }
    }
  };

  return (
    <div style={{ maxWidth: '600px', margin: '0 auto', padding: '2rem' }}>
      <h1>Train New Model</h1>
      
      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <div>
          <label htmlFor="file">CSV File:</label>
          <input
            type="file"
            id="file"
            accept=".csv"
            onChange={handleFileChange}
            required
            style={{ marginTop: '0.5rem', width: '100%' }}
          />
          {parsing && (
            <div style={{ marginTop: '0.5rem', color: '#666', fontSize: '0.9rem' }}>
              Parsing CSV...
            </div>
          )}
        </div>

        <div>
          <label htmlFor="targetColumn">Target Column:</label>
          {columnNames.length > 0 ? (
            <select
              id="targetColumn"
              value={targetColumn}
              onChange={(e) => setTargetColumn(e.target.value)}
              required
              style={{ marginTop: '0.5rem', width: '100%', padding: '0.5rem' }}
            >
              {columnNames.map((col) => (
                <option key={col} value={col}>
                  {col}
                </option>
              ))}
            </select>
          ) : (
            <input
              type="text"
              id="targetColumn"
              value={targetColumn}
              onChange={(e) => setTargetColumn(e.target.value)}
              placeholder="Upload a CSV file to see column options"
              required
              disabled
              style={{ 
                marginTop: '0.5rem', 
                width: '100%', 
                padding: '0.5rem',
                backgroundColor: '#f5f5f5',
                cursor: 'not-allowed'
              }}
            />
          )}
          {columnNames.length > 0 && (
            <div style={{ marginTop: '0.25rem', color: '#666', fontSize: '0.85rem' }}>
              Found {columnNames.length} column{columnNames.length !== 1 ? 's' : ''} in CSV
            </div>
          )}
        </div>

        <div>
          <label htmlFor="modelType">Model Type:</label>
          <select
            id="modelType"
            value={modelType}
            onChange={(e) => setModelType(e.target.value)}
            style={{ marginTop: '0.5rem', width: '100%', padding: '0.5rem' }}
          >
            <option value="logistic_regression">Logistic Regression</option>
            <option value="random_forest">Random Forest</option>
            <option value="xgboost">XGBoost</option>
          </select>
        </div>

        <div>
          <label htmlFor="modelName">Model Name:</label>
          <input
            type="text"
            id="modelName"
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            placeholder="e.g., Iris Classifier"
            required
            style={{ marginTop: '0.5rem', width: '100%', padding: '0.5rem' }}
          />
        </div>

        <button
          type="submit"
          disabled={loading}
          style={{
            padding: '0.75rem',
            backgroundColor: loading ? '#ccc' : '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: loading ? 'not-allowed' : 'pointer',
            fontSize: '1rem',
          }}
        >
          {loading ? `Training... (${Math.floor(elapsedTime / 60)} min ${elapsedTime % 60} sec)` : 'Train Model'}
        </button>
      </form>

      {error && (
        <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#f8d7da', color: '#721c24', borderRadius: '4px' }}>
          {error}
        </div>
      )}

      {success && (
        <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#d4edda', color: '#155724', borderRadius: '4px' }}>
          Model trained successfully! Redirecting to dashboard...
        </div>
      )}
    </div>
  );
};

export default Upload;

