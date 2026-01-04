import React, { useState, useContext, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { getPredictions } from '../api';
import { AuthContext } from '../context/AuthProvider';
import AdminControls from '../components/AdminControls';
import ActivityHeatmap from '../components/ActivityHeatmap';
import LiveAlerts from '../components/LiveAlerts';
import TemporalPatterns from '../components/TemporalPatterns';
import './Dashboard.css';

function Dashboard() {
    const { auth, logout } = useContext(AuthContext);
    const navigate = useNavigate();
    const [predictions, setPredictions] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [timeIndex, setTimeIndex] = useState(0);
    const [currentArea, setCurrentArea] = useState('City of London');
    const [hasRealPredictions, setHasRealPredictions] = useState(false);
    const [hasProcessedEvidence, setHasProcessedEvidence] = useState(false);

    // Memoize unique timestamps to avoid re-computation
    const uniqueTimeStamps = React.useMemo(() => {
        if (!predictions) return [];
        return [...new Set(predictions.map(p => p.ts))].sort();
    }, [predictions]);

    const selectedTimestamp = uniqueTimeStamps[timeIndex];

    useEffect(() => {
        const fetchPredictions = async () => {
            try {
                setLoading(true);
                console.log('Fetching national daily crime predictions...');

                // Fetch national daily predictions (30-365 days ahead)
                const data = await getPredictions(365);

                if (data && data.predictions) {
                    console.log('National predictions loaded:', data.predictions.length, 'daily forecasts');
                    setPredictions(data.predictions);

                    // Set display area to national level
                    setCurrentArea('United Kingdom (National)');

                    // Check if this is mock data (generated client-side)
                    const isMockData = data.predictions.length > 0 && data.predictions[0].id && data.predictions[0].id.includes('_');
                    setHasRealPredictions(!isMockData);
                    if (isMockData) {
                        console.log('📊 Using enhanced mock crime intelligence for demo');
                    } else {
                        console.log('✅ Using real national crime intelligence from backend');
                    }
                } else {
                    console.warn('No predictions found in response:', data);
                    setError('No crime intelligence data available.');
                }
            } catch (err) {
                setError('Failed to load crime intelligence data.');
                console.error('Error fetching national predictions:', err);
            } finally {
                setLoading(false);
            }
        };
        fetchPredictions();
    }, []);

    useEffect(() => {
        const checkProcessedEvidence = async () => {
            try {
                console.log('Checking for processed evidence in localStorage...');
                
                // Check both the flag AND that there's actual saved analysis data
                const hasFlag = localStorage.getItem('hasProcessedEvidence') === 'true';
                const savedAnalysisData = localStorage.getItem('savedAnalysis');
                
                // Only consider evidence processed if BOTH flag exists AND there's actual data
                const hasValidEvidence = hasFlag && savedAnalysisData !== null;
                
                // If flag exists but no data, clear the flag
                if (hasFlag && !savedAnalysisData) {
                    console.log('⚠️ Found flag but no saved analysis data - clearing flag');
                    localStorage.removeItem('hasProcessedEvidence');
                }
                
                setHasProcessedEvidence(hasValidEvidence);
                console.log(hasValidEvidence ? '✅ Found processed evidence with data - enabling investigative intelligence' : '📋 No processed evidence found in localStorage');
            } catch (err) {
                console.warn('Could not check localStorage:', err);
                setHasProcessedEvidence(false);
            }
        };
        
        // Check evidence on mount
        checkProcessedEvidence();
        
        // Also check evidence when window regains focus (user navigates back)
        const handleFocus = () => {
            console.log('Window focused - re-checking localStorage...');
            checkProcessedEvidence();
        };
        
        window.addEventListener('focus', handleFocus);
        
        // Cleanup
        return () => {
            window.removeEventListener('focus', handleFocus);
        };
    }, []);

    const handleSliderChange = (event) => {
        setTimeIndex(parseInt(event.target.value, 10));
    };

    const sliderDate = selectedTimestamp 
        ? new Date(selectedTimestamp).toLocaleDateString()
        : 'Loading...';

    const handleLogout = () => {
        // Clear processed evidence flag on logout
        localStorage.removeItem('hasProcessedEvidence');
        localStorage.removeItem('savedAnalysis');
        logout(); // This will clear the auth state
        navigate('/login');
    };

    return (
        <div className="dashboard-container">
            <div className="dashboard-header">
                <h2>🚀 Investigator Dashboard</h2>
                {hasRealPredictions && hasProcessedEvidence && (
                    <div className="area-indicator">
                        <strong>🎯 Investigative Intelligence: {currentArea}</strong>
                        <span className="context-note"> (Crime forecasting unlocked by evidence processing)</span>
                    </div>
                )}
                <div className="header-actions">
                    <Link to="/upload" className="action-link">📤 Upload New Evidence</Link>
                    <button onClick={handleLogout} className="logout-button">🚪 Logout</button>
                </div>
            </div>

            {error && <div className="error-message">⚠️ {error}</div>}
            {loading && <div className="loading-message">🔄 Loading crime intelligence data...</div>}

            {!hasProcessedEvidence && !loading && (
                <div className="info-message" style={{
                    background: 'rgba(59, 130, 246, 0.1)',
                    border: '1px solid rgba(59, 130, 246, 0.3)',
                    borderRadius: '8px',
                    padding: '1.5rem',
                    marginBottom: '2rem',
                    textAlign: 'center',
                    color: '#ffffff'
                }}>
                    <h3>🔒 Investigative Intelligence Locked</h3>
                    <p>Upload and process evidence through the <Link to="/upload" style={{color: '#60a5fa', textDecoration: 'underline'}}>Upload Section</Link> to unlock crime forecasting and hotspot analysis tools. Advanced predictive analytics require processed facial recognition data to activate.</p>
                </div>
            )}

            {hasRealPredictions && hasProcessedEvidence && !loading && !error && predictions.length > 0 && (
                <div className="info-message" style={{
                    background: 'rgba(34, 197, 94, 0.1)',
                    border: '1px solid rgba(34, 197, 94, 0.3)',
                    borderRadius: '8px',
                    padding: '1.5rem',
                    marginBottom: '2rem',
                    textAlign: 'center',
                    color: '#ffffff'
                }}>
                    <h3>🎯 Investigative Intelligence Activated</h3>
                    <p>Crime forecasting and hotspot analysis now available. These predictive tools help identify potential suspect locations and high-risk areas based on historical patterns and seasonal trends. Use this intelligence to prioritize investigation resources and track suspect movements.</p>
                </div>
            )}

            {hasRealPredictions && hasProcessedEvidence && !loading && !error && predictions.length > 0 && (
                <div className="time-slider-container">
                    <h4>📅 Intelligence Timeline: <strong>{sliderDate}</strong></h4>
                    <div className="slider-wrapper">
                        <input
                            type="range"
                            min="0"
                            max={uniqueTimeStamps.length - 1}
                            value={timeIndex}
                            onChange={handleSliderChange}
                            className="modern-slider"
                        />
                        <div className="slider-labels">
                            <span>{uniqueTimeStamps.length > 0 ? new Date(uniqueTimeStamps[0]).toLocaleDateString() : ''}</span>
                            <span>{uniqueTimeStamps.length > 0 ? new Date(uniqueTimeStamps[uniqueTimeStamps.length - 1]).toLocaleDateString() : ''}</span>
                        </div>
                    </div>
                </div>
            )}
            
            {/* Admin Controls Section */}
            <AdminControls isAdmin={true} />

            <div className="dashboard-grid">
                {hasRealPredictions && hasProcessedEvidence && (
                    <div className="grid-item wide">
                        <div className="card">
                            <h4>🗺️ Crime Intelligence Heatmap - {currentArea}</h4>
                            <ActivityHeatmap data={predictions} timestamp={selectedTimestamp} area={currentArea} />
                        </div>
                    </div>
                )}
                <div className="grid-item tall">
                    <div className="card">
                        <LiveAlerts /> {/* This component remains unchanged */}
                    </div>
                </div>
                {hasRealPredictions && hasProcessedEvidence && (
                    <div className="grid-item wide">
                        <div className="card">
                            <TemporalPatterns data={predictions} />
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

export default Dashboard;
