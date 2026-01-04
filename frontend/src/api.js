// --- Model Evaluation ---
// export const evaluateModel = async (modelId, modelType = 'custom', buffaloVariant) => {
//   const params = { model_type: modelType };
//   if (modelType === 'buffalo' && buffaloVariant) params.buffalo_variant = buffaloVariant;
//   const response = await apiClient.get(`/models/${modelId}/evaluate`, { params });
//   return response.data;
// };

export const evaluateModelFile = async (modelPath) => {
  const response = await apiClient.get(`/models/evaluate/file/${encodeURIComponent(modelPath)}`);
  return response.data;
};

// --- Buffalo Comparison ---
export const compareBuffaloModels = async () => {
  const response = await apiClient.post('/models/compare/buffalo');
  return response.data;
};

// --- Run Benchmarks ---
export const runBenchmarks = async () => {
  const response = await apiClient.post('/models/benchmarks/run');
  return response.data;
};
// --- Evidence Upload & Verification ---
export const uploadEvidence = async (file, metadata = {}) => {
  const formData = new FormData();
  formData.append('file', file);
  Object.entries(metadata).forEach(([key, value]) => formData.append(key, value));
  const response = await apiClient.post('/evidence', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

export const listEvidence = async (skip = 0, limit = 100) => {
  const response = await apiClient.get('/evidence', { params: { skip, limit } });
  return response.data;
};

export const downloadEvidence = async (evidenceId) => {
  const response = await apiClient.get(`/evidence/${evidenceId}`, {
    responseType: 'blob',
  });
  return response;
};

export const deleteEvidence = async (evidenceId) => {
  const response = await apiClient.delete(`/evidence/${evidenceId}`);
  return response.data;
};

export const verifyEvidence = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await apiClient.post('/evidence/verify', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

// --- Case Management ---
export const listCases = async () => {
  const response = await apiClient.get('/cases');
  return response.data;
};

export const createCase = async (caseData) => {
  const response = await apiClient.post('/cases', caseData);
  return response.data;
};

export const getCase = async (caseId) => {
  const response = await apiClient.get(`/cases/${caseId}`);
  return response.data;
};

// --- Predictive Intelligence ---
export const getPredictiveHotspots = async (payload) => {
  const response = await apiClient.post('/predictive/hotspots', payload);
  return response.data;
};

export const auditPrediction = async (payload) => {
  const response = await apiClient.post('/predictive/audit', payload);
  return response.data;
};

// --- Blockchain Evidence Ledger ---
export const verifyBlockchainEvidence = async (payload) => {
  const response = await apiClient.post('/evidence-ledger/verify', payload);
  return response.data;
};

// --- Audit Trail ---
export const getAuditTrail = async (caseId) => {
  const response = await apiClient.get(`/audit/${caseId}`);
  return response.data;
};

// --- Model Registry ---
export const listModels = async () => {
  const response = await apiClient.get('/models');
  return response.data;
};

export const listAvailableModels = async () => {
  const response = await apiClient.get('/models/list');
  return response.data;
};

export const activateModel = async (modelData) => {
  const response = await apiClient.post('/models/activate', modelData);
  return response.data;
};

import axios from 'axios';
import { io } from 'socket.io-client';
import { mockLogin } from './utils/mockAuth';

// Enable mock mode for development if backend is not available
const USE_MOCK = process.env.REACT_APP_USE_MOCK === 'true';

const API_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:8000/api/v1';
const WS_URL = process.env.REACT_APP_WS_URL || 'ws://127.0.0.1:8000';

// Initialize socket connection
export const socket = io(WS_URL, {
  autoConnect: false,
  path: '/ws/socket.io',
});

const apiClient = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const login = async (username, password) => {
  // Use mock login if backend is not available or in development
  if (USE_MOCK) {
    return mockLogin(username, password);
  }

  try {
    const params = new URLSearchParams();
    params.append('username', username);
    params.append('password', password);

    const response = await apiClient.post('/auth/token', params, {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
    });

    if (response.data.access_token) {
      localStorage.setItem('authToken', response.data.access_token);
    }
    return response.data;
  } catch (error) {
    console.warn('Backend login failed, falling back to mock authentication:', error.message);
    return mockLogin(username, password);
  }
};


export const logout = () => {
    localStorage.removeItem('authToken');
};

apiClient.interceptors.request.use(config => {
    const token = localStorage.getItem('authToken');
    if (token) {
        config.headers['Authorization'] = `Bearer ${token}`;
        console.log('Added auth token to request:', config.url);
    } else {
        console.warn('No auth token found for request:', config.url);
    }
    return config;
}, error => {
    console.error('Request interceptor error:', error);
    return Promise.reject(error);
});

// Add response interceptor for error handling
apiClient.interceptors.response.use(
    response => {
        // Any status code within the range of 2xx
        return response;
    },
    error => {
        if (error.response) {
            // The request was made and the server responded with a status code
            // that falls out of the range of 2xx
            console.error('Response error:', error.response.status, error.response.data);

            // Only handle 401 Unauthorized errors for critical endpoints
            // Let individual functions handle their own auth errors with fallbacks
            if (error.response.status === 401) {
                console.warn('Authentication error detected - letting individual functions handle it');
                // Don't automatically redirect to login - let functions handle fallbacks
            }
        } else if (error.request) {
            // The request was made but no response was received
            console.error('No response received:', error.request);
        } else {
            // Something happened in setting up the request that triggered an Error
            console.error('Request setup error:', error.message);
        }

        // Pass the error along to the component/function
        return Promise.reject(error);
    }
);

export const reconstructFace = async (file, caseId = 'web-upload') => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('case_id', caseId);  // Required by backend

  const response = await apiClient.post('/reconstruct', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response;
};

import { mockPredictions, mockAlerts } from './utils/mockData';

export const getPredictions = async (windowDays = 365) => {
  try {
    console.log(`Fetching national daily predictions with window of ${windowDays} days`);
    const response = await apiClient.get(`/analytics/predictions?window=${windowDays}`);
    // Check if we got valid predictions data
    if (response.data && response.data.predictions && response.data.predictions.length > 0) {
      console.log(`✅ Got ${response.data.predictions.length} real national daily predictions from backend`);
      return response.data;
    } else {
      console.warn('⚠️ Backend returned empty predictions - no forecasting data available');
      return { predictions: [] };
    }
  } catch (error) {
    console.warn('❌ Failed to fetch predictions from backend:', error.message);
    return { predictions: [] };
  }
};

export const recognizeFace = async (file, caseId = 'web-upload') => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('case_id', caseId);  // Required by backend

  const response = await apiClient.post('/recognize', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response;
};

export const generateReport = async (reportPayload) => {
  const response = await apiClient.post('/report', reportPayload, {
    headers: { 'Content-Type': 'application/json' },
    responseType: 'blob', // Important for file downloads
  });
  return response;
};

export const buildWatchlist = async () => {
  const response = await apiClient.post('/models/watchlist/build');
  return response.data;
};

export const runRecognitionTest = async (threshold = 0.5) => {
  const response = await apiClient.post(`/models/demo/run?threshold=${threshold}`);
  return response.data;
};

export const tuneThreshold = async (maxSamples = 10) => {
  const response = await apiClient.post(`/models/tune-threshold?max_samples=${maxSamples}`);
  return response.data;
};

export const buildCrimeContext = async (force = null) => {
  const formData = new FormData();
  if (force) {
    formData.append('force', force);
  }
  const response = await apiClient.post('/analytics/crime/context', formData);
  return response.data;
};

// --- Crime Analytics (reads) ---
export const getCrimeMonthlyTrends = async (from, to) => {
  const response = await apiClient.get('/crime/forces/monthly', {
    params: { from, to },
  });
  return response.data;
};

export const getCrimeLatestHotspots = async (force) => {
  const params = {};
  if (force) params.force = force;
  const response = await apiClient.get('/crime/hotspots/latest', { params });
  return response.data;
};

export const getCrimeLsoaSeries = async (lsoa) => {
  const response = await apiClient.get('/crime/lsoa/series', {
    params: { lsoa },
  });
  return response.data;
};

export const getCrimeSummary = async () => {
  const response = await apiClient.get('/crime/summary');
  return response.data;
};

export const getCrimeAggregatedTrends = async () => {
  const response = await apiClient.get('/crime/forces/monthly/all');
  return response.data;
};

export const listCrimeForces = async () => {
  const response = await apiClient.get('/crime/forces');
  return response.data;
};

export const listCrimeLsoas = async (force) => {
  const params = {};
  if (force) params.force = force;
  const response = await apiClient.get('/crime/lsoas', { params });
  return response.data;
};

// --- Evaluation Automation Endpoints (Admin) ---
export const evaluateLandmarksAutomation = async () => {
  const response = await apiClient.post('/models/evaluate/landmarks');
  return response.data;
};

export const evaluateReconstructionAutomation = async () => {
  const response = await apiClient.post('/models/evaluate/reconstruction');
  return response.data;
};

export const evaluateRecognitionAutomation = async () => {
  const response = await apiClient.post('/models/evaluate/recognition');
  return response.data;
};

export default apiClient;
