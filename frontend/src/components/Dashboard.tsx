import { useState, useEffect, useRef } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { getModels, Model, updateModel, deleteModel } from '../api';

const Dashboard = () => {
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editName, setEditName] = useState('');
  const [menuOpenId, setMenuOpenId] = useState<number | null>(null);
  const [deleteConfirmId, setDeleteConfirmId] = useState<number | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();

  const fetchModels = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getModels();
      setModels(data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to fetch models');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchModels();
  }, []);

  // Close menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setMenuOpenId(null);
      }
    };

    if (menuOpenId !== null) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [menuOpenId]);

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString();
  };

  const formatAccuracy = (accuracy: number | null) => {
    if (accuracy === null) return 'N/A';
    return (accuracy * 100).toFixed(2) + '%';
  };

  const formatModelType = (type: string) => {
    return type
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  const handleRowClick = (modelId: number, event: React.MouseEvent) => {
    // Don't navigate if clicking on the menu button
    if ((event.target as HTMLElement).closest('.menu-button') || 
        (event.target as HTMLElement).closest('.menu-dropdown')) {
      return;
    }
    navigate(`/model/${modelId}`);
  };

  const handleEdit = (model: Model) => {
    setEditingId(model.id);
    setEditName(model.name);
    setMenuOpenId(null);
  };

  const handleSaveEdit = async (modelId: number) => {
    if (!editName.trim()) {
      setError('Model name cannot be empty');
      return;
    }

    try {
      await updateModel(modelId, editName.trim());
      setEditingId(null);
      await fetchModels();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to update model');
    }
  };

  const handleCancelEdit = () => {
    setEditingId(null);
    setEditName('');
  };

  const handleDeleteClick = (modelId: number) => {
    setDeleteConfirmId(modelId);
    setMenuOpenId(null);
  };

  const handleDeleteConfirm = async () => {
    if (!deleteConfirmId) return;

    try {
      await deleteModel(deleteConfirmId);
      setError(null); // Clear any previous errors
      setDeleteConfirmId(null);
      await fetchModels();
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || err.message || 'Failed to delete model';
      setError(errorMessage);
      // Don't close modal on error - let user see the error and try again or cancel
      console.error('Delete error:', err);
    }
  };

  const handleDeleteCancel = () => {
    setDeleteConfirmId(null);
    setError(null); // Clear error when canceling
  };

  const toggleMenu = (modelId: number, event: React.MouseEvent) => {
    event.stopPropagation();
    setMenuOpenId(menuOpenId === modelId ? null : modelId);
  };

  if (loading) {
    return <div style={{ padding: '2rem', textAlign: 'center' }}>Loading...</div>;
  }

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '2rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
        <h1>Models</h1>
        <button
          onClick={() => navigate('/upload')}
          style={{
            padding: '0.5rem 1rem',
            backgroundColor: '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '1rem',
          }}
        >
          Train Model
        </button>
      </div>

      {error && (
        <div style={{ marginBottom: '1rem', padding: '1rem', backgroundColor: '#f8d7da', color: '#721c24', borderRadius: '4px' }}>
          {error}
        </div>
      )}

      {models.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '2rem' }}>
          <p>No models trained yet. <Link to="/upload">Train your first model</Link></p>
        </div>
      ) : (
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ backgroundColor: '#f8f9fa' }}>
              <th style={{ padding: '1rem', textAlign: 'left', borderBottom: '2px solid #dee2e6' }}>Model Name</th>
              <th style={{ padding: '1rem', textAlign: 'left', borderBottom: '2px solid #dee2e6' }}>Dataset</th>
              <th style={{ padding: '1rem', textAlign: 'left', borderBottom: '2px solid #dee2e6' }}>Type</th>
              <th style={{ padding: '1rem', textAlign: 'left', borderBottom: '2px solid #dee2e6' }}>Accuracy</th>
              <th style={{ padding: '1rem', textAlign: 'left', borderBottom: '2px solid #dee2e6' }}>Created</th>
              <th style={{ padding: '1rem', textAlign: 'left', borderBottom: '2px solid #dee2e6', width: '50px' }}></th>
            </tr>
          </thead>
          <tbody>
            {models.map((model) => (
              <tr
                key={model.id}
                onClick={(e) => handleRowClick(model.id, e)}
                style={{
                  borderBottom: '1px solid #dee2e6',
                  cursor: 'pointer',
                  transition: 'background-color 0.2s',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = '#f8f9fa';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = 'transparent';
                }}
              >
                <td style={{ padding: '1rem' }}>
                  {editingId === model.id ? (
                    <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                      <input
                        type="text"
                        value={editName}
                        onChange={(e) => setEditName(e.target.value)}
                        onClick={(e) => e.stopPropagation()}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') {
                            handleSaveEdit(model.id);
                          } else if (e.key === 'Escape') {
                            handleCancelEdit();
                          }
                        }}
                        style={{
                          padding: '0.25rem 0.5rem',
                          border: '1px solid #007bff',
                          borderRadius: '4px',
                          fontSize: '1rem',
                        }}
                        autoFocus
                      />
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleSaveEdit(model.id);
                        }}
                        style={{
                          padding: '0.25rem 0.5rem',
                          backgroundColor: '#28a745',
                          color: 'white',
                          border: 'none',
                          borderRadius: '4px',
                          cursor: 'pointer',
                          fontSize: '0.875rem',
                        }}
                      >
                        Save
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleCancelEdit();
                        }}
                        style={{
                          padding: '0.25rem 0.5rem',
                          backgroundColor: '#6c757d',
                          color: 'white',
                          border: 'none',
                          borderRadius: '4px',
                          cursor: 'pointer',
                          fontSize: '0.875rem',
                        }}
                      >
                        Cancel
                      </button>
                    </div>
                  ) : (
                    model.name
                  )}
                </td>
                <td style={{ padding: '1rem' }}>{model.dataset_name || 'N/A'}</td>
                <td style={{ padding: '1rem' }}>{formatModelType(model.type)}</td>
                <td style={{ padding: '1rem' }}>{formatAccuracy(model.accuracy)}</td>
                <td style={{ padding: '1rem' }}>{formatDate(model.created_at)}</td>
                <td style={{ padding: '1rem', position: 'relative' }}>
                  <button
                    className="menu-button"
                    onClick={(e) => toggleMenu(model.id, e)}
                    style={{
                      background: 'none',
                      border: 'none',
                      cursor: 'pointer',
                      fontSize: '1.25rem',
                      padding: '0.25rem 0.5rem',
                      color: '#6c757d',
                    }}
                    title="More options"
                  >
                    ⋮
                  </button>
                  {menuOpenId === model.id && (
                    <div
                      ref={menuRef}
                      className="menu-dropdown"
                      style={{
                        position: 'absolute',
                        right: '1rem',
                        top: '2.5rem',
                        backgroundColor: 'white',
                        border: '1px solid #dee2e6',
                        borderRadius: '4px',
                        boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
                        zIndex: 1000,
                        minWidth: '150px',
                      }}
                      onClick={(e) => e.stopPropagation()}
                    >
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleEdit(model);
                        }}
                        style={{
                          width: '100%',
                          padding: '0.75rem 1rem',
                          textAlign: 'left',
                          background: 'none',
                          border: 'none',
                          cursor: 'pointer',
                          fontSize: '0.9rem',
                          color: '#333',
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.backgroundColor = '#f8f9fa';
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.backgroundColor = 'transparent';
                        }}
                      >
                        Edit Name
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteClick(model.id);
                        }}
                        style={{
                          width: '100%',
                          padding: '0.75rem 1rem',
                          textAlign: 'left',
                          background: 'none',
                          border: 'none',
                          borderTop: '1px solid #dee2e6',
                          cursor: 'pointer',
                          fontSize: '0.9rem',
                          color: '#dc3545',
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.backgroundColor = '#f8f9fa';
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.backgroundColor = 'transparent';
                        }}
                      >
                        Delete
                      </button>
                    </div>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {/* Delete Confirmation Modal */}
      {deleteConfirmId && (() => {
        const modelToDelete = models.find(m => m.id === deleteConfirmId);
        return (
          <div
            style={{
              position: 'fixed',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              backgroundColor: 'rgba(0, 0, 0, 0.5)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              zIndex: 2000,
            }}
            onClick={handleDeleteCancel}
          >
            <div
              style={{
                backgroundColor: 'white',
                padding: '2rem',
                borderRadius: '8px',
                maxWidth: '400px',
                width: '90%',
                boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
              }}
              onClick={(e) => e.stopPropagation()}
            >
              <h2 style={{ marginTop: 0, marginBottom: '1rem' }}>Delete Model</h2>
              <p style={{ marginBottom: '2rem', color: '#666' }}>
                Are you sure you want to delete <strong>{modelToDelete?.name || 'this model'}</strong>? This action cannot be undone.
              </p>
              {error && deleteConfirmId && (
                <div style={{ 
                  marginBottom: '1rem', 
                  padding: '0.75rem', 
                  backgroundColor: '#f8d7da', 
                  color: '#721c24', 
                  borderRadius: '4px',
                  fontSize: '0.9rem'
                }}>
                  {error}
                </div>
              )}
              <div style={{ display: 'flex', gap: '1rem', justifyContent: 'flex-end' }}>
                <button
                  onClick={handleDeleteCancel}
                  style={{
                    padding: '0.5rem 1.5rem',
                    backgroundColor: '#6c757d',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontSize: '1rem',
                  }}
                >
                  Cancel
                </button>
                <button
                  onClick={handleDeleteConfirm}
                  style={{
                    padding: '0.5rem 1.5rem',
                    backgroundColor: '#dc3545',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontSize: '1rem',
                  }}
                >
                  Delete
                </button>
              </div>
            </div>
          </div>
        );
      })()}
    </div>
  );
};

export default Dashboard;
