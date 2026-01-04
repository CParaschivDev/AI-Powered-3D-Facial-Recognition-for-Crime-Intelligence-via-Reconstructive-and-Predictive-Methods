import React, { useState, useEffect, useContext } from 'react';
import { AuthContext } from '../context/AuthProvider';
import { mockAlerts } from '../utils/mockData';

// Get WebSocket URL from environment or use default
const WS_URL = process.env.REACT_APP_WS_URL || "ws://127.0.0.1:8000";

function LiveAlerts() {
  const { user } = useContext(AuthContext);
  const [alerts, setAlerts] = useState([]);
  
  useEffect(() => {
    // First load mock alerts immediately
    setAlerts(mockAlerts);

    if (!user || !user.token) return;
    
    // Use native WebSocket connection
    const wsUrl = process.env.REACT_APP_WS_URL || 'ws://127.0.0.1:8000';
    // Append the alerts endpoint and token as query parameter
    const ws = new WebSocket(`${wsUrl}/api/v1/ws/alerts?token=${user.token}`);

    ws.onopen = () => {
      console.log('WebSocket connected');
    };

    ws.onerror = (err) => {
      console.error('WebSocket error:', err);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setAlerts((prevAlerts) => [...prevAlerts, data]);
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    return () => {
      ws.close();
    };
  }, [user]);

  return (
    <div className="live-alerts">
      <h4>🚨 Live Alerts</h4>
      <div className="alerts-list">
        {alerts.map((alert, i) => (
          <div key={i} className="alert-item">
            <span className="alert-icon">🚨</span>
            <span className="alert-text">
              High-confidence match for <strong>{alert.id}</strong> at {alert.camera || alert.location} ({alert.time})
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default LiveAlerts;
