import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './SplashScreen.css';

const SplashScreen = ({ onComplete }) => {
  const [fadeOut, setFadeOut] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    // Start fade out after 2.5 seconds
    const fadeTimer = setTimeout(() => {
      setFadeOut(true);
    }, 2500);

    // Navigate after fade out animation completes (0.5 seconds)
    const navigateTimer = setTimeout(() => {
      if (onComplete) {
        onComplete();
      } else {
        navigate('/login');
      }
    }, 3000);

    // Clean up timers if component unmounts
    return () => {
      clearTimeout(fadeTimer);
      clearTimeout(navigateTimer);
    };
  }, [navigate, onComplete]);

  return (
    <div className={`splash-screen ${fadeOut ? 'fade-out' : ''}`}>
      <div className="splash-content">
        <img 
          src={`${process.env.PUBLIC_URL}/logo.svg`} 
          alt="Police Shield Logo" 
          className="splash-logo" 
          onError={(e) => {
            console.error("Error loading logo:", e);
            e.target.onerror = null;
            e.target.src = `${process.env.PUBLIC_URL}/logo192.png`;
          }}
        />
        
        <h1 className="splash-title">3D Face Reconstruction System</h1>
        <p className="splash-subtitle">for Police Investigations</p>
        
        <div className="face-mesh-container">
          <img 
            src={`${process.env.PUBLIC_URL}/face-mesh.svg`} 
            alt="3D Face Mesh" 
            className="face-mesh-image"
            onError={(e) => {
              console.error("Error loading face mesh:", e);
            }}
          />
        </div>
        
        <div className="loading-dots">
          <div className="dot"></div>
          <div className="dot"></div>
          <div className="dot"></div>
        </div>
      </div>
    </div>
  );
};

export default SplashScreen;