import React, { useState } from 'react';
import { buildWatchlist, runRecognitionTest, tuneThreshold, buildCrimeContext, evaluateLandmarksAutomation, evaluateRecognitionAutomation, evaluateReconstructionAutomation } from '../api';

const AdminControls = ({ isAdmin }) => {
  const [loading, setLoading] = useState({
    watchlist: false,
    test: false,
    threshold: false,
    crime: false,
    evalLandmarks: false,
    evalRecognition: false,
    evalReconstruction: false,
  });
  const [results, setResults] = useState({
    watchlist: null,
    test: null,
    threshold: null,
    crime: null,
    evalLandmarks: null,
    evalRecognition: null,
    evalReconstruction: null,
  });
  const [error, setError] = useState(null);
  const [threshold, setThreshold] = useState(0.5);
  const [maxSamples, setMaxSamples] = useState(10);
  const [forceFilter, setForceFilter] = useState('');

  if (!isAdmin) {
    return null;
  }

  const handleBuildWatchlist = async () => {
    setLoading(prev => ({ ...prev, watchlist: true }));
    setError(null);
    try {
      const response = await buildWatchlist();
      setResults(prev => ({ ...prev, watchlist: response }));
    } catch (err) {
      setError(`Error building watchlist: ${err.message}`);
    } finally {
      setLoading(prev => ({ ...prev, watchlist: false }));
    }
  };

  const handleRunTest = async () => {
    setLoading(prev => ({ ...prev, test: true }));
    setError(null);
    try {
      const response = await runRecognitionTest(threshold);
      setResults(prev => ({ ...prev, test: response }));
    } catch (err) {
      setError(`Error running recognition test: ${err.message}`);
    } finally {
      setLoading(prev => ({ ...prev, test: false }));
    }
  };

  const handleTuneThreshold = async () => {
    setLoading(prev => ({ ...prev, threshold: true }));
    setError(null);
    try {
      const response = await tuneThreshold(maxSamples);
      setResults(prev => ({ ...prev, threshold: response }));
      
      // If successful, update the threshold value with the suggested one
      if (response?.ok && response?.details?.suggested_threshold) {
        setThreshold(response.details.suggested_threshold);
      }
    } catch (err) {
      setError(`Error tuning threshold: ${err.message}`);
    } finally {
      setLoading(prev => ({ ...prev, threshold: false }));
    }
  };

  const handleBuildCrimeContext = async () => {
    setLoading(prev => ({ ...prev, crime: true }));
    setError(null);
    try {
      const response = await buildCrimeContext(forceFilter || null);
      setResults(prev => ({ ...prev, crime: response }));
    } catch (err) {
      setError(`Error building crime context: ${err.message}`);
    } finally {
      setLoading(prev => ({ ...prev, crime: false }));
    }
  };

  const handleEvalLandmarks = async () => {
    setLoading(prev => ({ ...prev, evalLandmarks: true }));
    setError(null);
    try {
      const response = await evaluateLandmarksAutomation();
      setResults(prev => ({ ...prev, evalLandmarks: response }));
    } catch (err) {
      setError(`Error evaluating landmarks: ${err?.response?.data?.detail || err.message}`);
    } finally {
      setLoading(prev => ({ ...prev, evalLandmarks: false }));
    }
  };

  const handleEvalRecognition = async () => {
    setLoading(prev => ({ ...prev, evalRecognition: true }));
    setError(null);
    try {
      const response = await evaluateRecognitionAutomation();
      setResults(prev => ({ ...prev, evalRecognition: response }));
    } catch (err) {
      setError(`Error evaluating recognition: ${err?.response?.data?.detail || err.message}`);
    } finally {
      setLoading(prev => ({ ...prev, evalRecognition: false }));
    }
  };

  const handleEvalReconstruction = async () => {
    setLoading(prev => ({ ...prev, evalReconstruction: true }));
    setError(null);
    try {
      const response = await evaluateReconstructionAutomation();
      setResults(prev => ({ ...prev, evalReconstruction: response }));
    } catch (err) {
      setError(`Error evaluating reconstruction: ${err?.response?.data?.detail || err.message}`);
    } finally {
      setLoading(prev => ({ ...prev, evalReconstruction: false }));
    }
  };

  return (
    <div className="admin-controls-container">
      <h2>Admin Controls</h2>

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      <div className="admin-controls-grid">
        <div className="control-card">
          <h3>Watchlist Management</h3>
          <button 
            onClick={handleBuildWatchlist} 
            disabled={loading.watchlist}
            className="primary-button"
          >
            {loading.watchlist ? 'Building...' : 'Build Watchlist'}
          </button>
          {results.watchlist && (
            <div className="results-box">
              <h4>Result:</h4>
              <pre>{JSON.stringify(results.watchlist, null, 2)}</pre>
            </div>
          )}
        </div>

        <div className="control-card">
          <h3>Recognition Test</h3>
          <div className="form-group">
            <label>Threshold:</label>
            <input 
              type="number" 
              min="0" 
              max="1" 
              step="0.01"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
            />
          </div>
          <button 
            onClick={handleRunTest} 
            disabled={loading.test}
            className="primary-button"
          >
            {loading.test ? 'Testing...' : 'Run Recognition Test'}
          </button>
          {results.test && (
            <div className="results-box">
              <h4>Result:</h4>
              <pre>{JSON.stringify(results.test, null, 2)}</pre>
            </div>
          )}
        </div>

        <div className="control-card">
          <h3>Auto Threshold Tuning</h3>
          <div className="form-group">
            <label>Max Samples:</label>
            <input 
              type="number" 
              min="1" 
              max="50" 
              step="1"
              value={maxSamples}
              onChange={(e) => setMaxSamples(parseInt(e.target.value))}
            />
          </div>
          <button 
            onClick={handleTuneThreshold} 
            disabled={loading.threshold}
            className="primary-button"
          >
            {loading.threshold ? 'Tuning...' : 'Auto-Tune Threshold'}
          </button>
          {results.threshold && (
            <div className="results-box">
              <h4>Result:</h4>
              <pre>{JSON.stringify(results.threshold, null, 2)}</pre>
            </div>
          )}
        </div>

        <div className="control-card">
          <h3>Crime Context</h3>
          <div className="form-group">
            <label>Force Filter (Optional):</label>
            <input 
              type="text" 
              value={forceFilter}
              onChange={(e) => setForceFilter(e.target.value)}
              placeholder="e.g. metropolitan"
            />
          </div>
          <button 
            onClick={handleBuildCrimeContext} 
            disabled={loading.crime}
            className="primary-button"
          >
            {loading.crime ? 'Building...' : 'Build Crime Context'}
          </button>
          {results.crime && (
            <div className="results-box">
              <h4>Result:</h4>
              <pre>{JSON.stringify(results.crime, null, 2)}</pre>
            </div>
          )}
        </div>

        <div className="control-card">
          <h3>Evaluate Landmarks</h3>
          <button 
            onClick={handleEvalLandmarks}
            disabled={loading.evalLandmarks}
            className="primary-button"
          >
            {loading.evalLandmarks ? 'Running...' : 'Run Landmark Evaluation'}
          </button>
          {results.evalLandmarks && (
            <div className="results-box">
              <h4>Result:</h4>
              <pre>{JSON.stringify(results.evalLandmarks, null, 2)}</pre>
            </div>
          )}
        </div>

        <div className="control-card">
          <h3>Evaluate Recognition</h3>
          <button 
            onClick={handleEvalRecognition}
            disabled={loading.evalRecognition}
            className="primary-button"
          >
            {loading.evalRecognition ? 'Running...' : 'Run Recognition Evaluation'}
          </button>
          {results.evalRecognition && (
            <div className="results-box">
              <h4>Result:</h4>
              <pre>{JSON.stringify(results.evalRecognition, null, 2)}</pre>
            </div>
          )}
        </div>

        <div className="control-card">
          <h3>Evaluate Reconstruction</h3>
          <button 
            onClick={handleEvalReconstruction}
            disabled={loading.evalReconstruction}
            className="primary-button"
          >
            {loading.evalReconstruction ? 'Running...' : 'Run Reconstruction Evaluation'}
          </button>
          {results.evalReconstruction && (
            <div className="results-box">
              <h4>Result:</h4>
              <pre>{JSON.stringify(results.evalReconstruction, null, 2)}</pre>
            </div>
          )}
        </div>
      </div>

      <style jsx>{`
        .admin-controls-container {
          padding: 20px;
          background: rgba(255, 255, 255, 0.05);
          backdrop-filter: blur(15px);
          border-radius: 20px;
          margin: 20px 0;
          border: 1px solid rgba(255, 255, 255, 0.1);
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        .admin-controls-container h2 {
          color: rgba(192, 192, 192, 0.9);
          font-size: 2rem;
          font-weight: 300;
          margin-bottom: 1rem;
          text-shadow: 0 0 10px rgba(128, 128, 128, 0.3);
        }
        
        .admin-controls-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
          gap: 20px;
          margin-top: 20px;
        }
        
        .control-card {
          background: rgba(255, 255, 255, 0.05);
          backdrop-filter: blur(10px);
          border-radius: 16px;
          padding: 20px;
          box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
          border: 1px solid rgba(255, 255, 255, 0.1);
          transition: all 0.3s ease;
        }
        
        .control-card:hover {
          transform: translateY(-3px);
          box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
          border-color: rgba(255, 255, 255, 0.2);
        }
        
        .control-card h3 {
          color: rgba(192, 192, 192, 0.9);
          font-size: 1.3rem;
          font-weight: 400;
          margin-bottom: 15px;
          text-shadow: 0 0 5px rgba(128, 128, 128, 0.2);
        }
        
        .form-group {
          margin-bottom: 15px;
        }
        
        .form-group label {
          display: block;
          margin-bottom: 5px;
          color: rgba(192, 192, 192, 0.8);
          font-weight: 500;
        }
        
        .form-group input {
          width: 100%;
          padding: 10px 12px;
          border: 1px solid rgba(128, 128, 128, 0.3);
          border-radius: 8px;
          background: rgba(128, 128, 128, 0.1);
          color: rgba(192, 192, 192, 0.9);
          font-size: 1rem;
          backdrop-filter: blur(5px);
        }
        
        .form-group input:focus {
          outline: none;
          border-color: rgba(128, 128, 128, 0.6);
          box-shadow: 0 0 10px rgba(128, 128, 128, 0.3);
        }
        
        .primary-button {
          background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
          color: rgba(192, 192, 192, 0.9);
          border: none;
          padding: 12px 20px;
          border-radius: 10px;
          cursor: pointer;
          font-weight: 500;
          width: 100%;
          transition: all 0.3s ease;
          box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
          border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .primary-button:hover {
          transform: translateY(-2px);
          box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
          background: linear-gradient(45deg, #764ba2 0%, #667eea 100%);
        }
        
        .primary-button:disabled {
          background: rgba(255, 255, 255, 0.1);
          cursor: not-allowed;
          box-shadow: none;
          transform: none;
        }
        
        .results-box {
          margin-top: 15px;
          background: rgba(255, 255, 255, 0.05);
          padding: 15px;
          border-radius: 10px;
          font-size: 12px;
          max-height: 200px;
          overflow-y: auto;
          border: 1px solid rgba(255, 255, 255, 0.1);
          backdrop-filter: blur(5px);
        }
        
        .results-box h4 {
          color: rgba(192, 192, 192, 0.9);
          margin: 0 0 10px 0;
          font-size: 1rem;
        }
        
        .results-box pre {
          color: rgba(192, 192, 192, 0.9);
          margin: 0;
        }
        
        .error-message {
          background: rgba(255, 107, 107, 0.1);
          color: #ff6b6b;
          padding: 15px;
          border-radius: 10px;
          margin-bottom: 15px;
          border: 1px solid rgba(255, 107, 107, 0.3);
          backdrop-filter: blur(10px);
        }
      `}</style>
    </div>
  );
};

export default AdminControls;