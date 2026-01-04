import React, { useState, useContext, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { login } from '../api';
import { AuthContext } from '../context/AuthProvider';
import './Login.css';

function Login() {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const { auth, setAuth } = useContext(AuthContext);
    const navigate = useNavigate();
    const location = useLocation();

    // Check if user is already logged in
    useEffect(() => {
        if (auth?.token) {
            console.log("Already authenticated, redirecting to dashboard");
            navigate('/');
        }
    }, [auth, navigate]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsLoading(true);
        setError('');

        try {
            console.log('Login attempt with:', username);
            const response = await login(username, password);
            console.log('Login response:', response);

            if (response && response.access_token) {
                setAuth({ token: response.access_token, user: username });

                // Force reload to ensure the app recognizes the auth state change
                setTimeout(() => {
                    navigate('/', { replace: true });
                }, 500);
            } else {
                setError('Invalid response format from server');
            }
        } catch (err) {
            console.error('Login error:', err);
            // Make sure we don't pass objects directly as error messages
            const errorMessage = typeof err?.response?.data?.detail === 'string'
                ? err.response.data.detail
                : 'Login failed. Please check your credentials.';
            setError(errorMessage);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="login-container">
            <div className="login-form-container">
                <h2 className="login-title">Officer Login</h2>
                <form className="login-form" onSubmit={handleSubmit}>
                    <div className="form-group">
                        <label className="form-label" htmlFor="username">Username</label>
                        <input
                            id="username"
                            className="form-input"
                            type="text"
                            placeholder="Enter your username"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            required
                        />
                    </div>
                    <div className="form-group">
                        <label className="form-label" htmlFor="password">Password</label>
                        <input
                            id="password"
                            className="form-input"
                            type="password"
                            placeholder="Enter your password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                        />
                    </div>
                    <button type="submit" className="login-button" disabled={isLoading}>
                        {isLoading ? 'Logging in...' : 'Login'}
                    </button>
                </form>
                {error && <p className="error-message">{String(error)}</p>}
                {isLoading && <p className="loading-message">Authenticating...</p>}

                <div className="system-info">
                    <p><strong>API Endpoint:</strong> {process.env.REACT_APP_API_URL || 'http://127.0.0.1:8000/api/v1'}</p>
                    <p><strong>Demo Credentials:</strong></p>
                    <p>Username: officer | Password: password</p>
                    <p>Username: admin | Password: admin_password</p>
                </div>
            </div>
        </div>
    );
}

export default Login;
