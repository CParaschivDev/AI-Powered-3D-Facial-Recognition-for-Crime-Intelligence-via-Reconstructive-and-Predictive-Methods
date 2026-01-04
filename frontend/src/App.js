import React, { useContext, useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, Link, useLocation } from 'react-router-dom';
import Upload from './pages/Upload.js';
import Dashboard from './pages/Dashboard.js';
import Login from './pages/Login.js';
import EvidenceManager from './components/EvidenceManager';
import { AuthProvider, AuthContext } from './context/AuthProvider.js';
import SplashScreen from './components/splash/SplashScreen';
import './App.css';
import CrimeAnalytics from './pages/CrimeAnalytics';

function RequireAuth({ children }) {
  const { auth } = useContext(AuthContext);
  console.log('RequireAuth check:', auth);
  
  if (!auth?.token) {
    console.log('No token found, redirecting to login');
    return <Navigate to="/login" replace />;
  }
  
  console.log('Auth token found, rendering protected content');
  return children;
}

function AppContent() {
  const [showSplash, setShowSplash] = useState(true);
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Dashboard' },
    { path: '/upload', label: 'Upload' },
    { path: '/evidence', label: 'Evidence' },
    { path: '/analytics', label: 'Analytics' },
  ];
  return (
    <div className="App">
      <header className="App-header">
        <div className="header-content">
          <h1 className="app-title">An AI-Powered 3D Facial Recognition Framework for Crime Intelligence Using Reconstructive and Predictive Techniques</h1>
          <nav className="modern-nav">
            <div className="nav-container">
              {navItems.map((item) => (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`nav-link ${location.pathname === item.path ? 'active' : ''}`}
                >
                  <span className="nav-label">{item.label}</span>
                </Link>
              ))}
            </div>
          </nav>
        </div>
      </header>
      <main>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/upload" element={<RequireAuth><Upload /></RequireAuth>} />
          <Route path="/evidence" element={<RequireAuth><EvidenceManager /></RequireAuth>} />
          <Route path="/analytics" element={<RequireAuth><CrimeAnalytics /></RequireAuth>} />
          <Route path="/" element={<RequireAuth><Dashboard /></RequireAuth>} />
        </Routes>
      </main>
      <footer className="modern-footer">
        <div className="footer-content">
          <div className="footer-main">
            <div className="footer-brand">
              <div className="footer-icon">🔒</div>
              <div className="footer-text">
                <h3>Confidential Law Enforcement Tool</h3>
                <p>Advanced AI-Powered Facial Recognition System</p>
              </div>
            </div>
            <div className="footer-meta">
              <div className="footer-status">
                <span className="status-dot"></span>
                <span>System Operational</span>
              </div>
              <div className="footer-author">
                <span>Cristian-Constantin Paraschiv, University of Wales Trinity Saint David</span>
              </div>
            </div>
          </div>
          <div className="footer-bottom">
            <div className="footer-links">
              <span>© 2025 AI Crime Intelligence Platform</span>
              <span>•</span>
              <span>Version 4.0</span>
              <span>•</span>
              <span>Secure & Encrypted</span>
            </div>
            <div className="footer-badge">
              <span className="badge">🛡️ CLASSIFIED</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

function App() {
  return (
    <Router>
      <AuthProvider>
        <AppContent />
      </AuthProvider>
    </Router>
  );
}

export default App;
