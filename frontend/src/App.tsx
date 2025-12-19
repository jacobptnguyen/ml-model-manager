import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Upload from './components/Upload';
import Dashboard from './components/Dashboard';
import ModelDetail from './components/ModelDetail';
import './App.css';

function App() {
  return (
    <Router>
      <div>
        <nav style={{
          backgroundColor: '#343a40',
          padding: '1.5rem',
          marginBottom: '2rem',
        }}>
          <div style={{
            maxWidth: '1200px',
            margin: '0 auto',
            display: 'flex',
            gap: '2rem',
            alignItems: 'center',
          }}>
            <Link
              to="/"
              style={{
                color: 'white',
                textDecoration: 'none',
                fontSize: '1.75rem',
                fontWeight: 'bold',
              }}
            >
              ML Model Manager
            </Link>
          </div>
        </nav>

        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/upload" element={<Upload />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/model/:id" element={<ModelDetail />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;

