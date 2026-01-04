import React from 'react';
import { MapContainer, TileLayer, CircleMarker, Tooltip } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
// This is a simplified heatmap. For a true heatmap layer, you might use a plugin
// like 'react-leaflet-heatmap-layer-v3'. For simplicity, we use CircleMarkers.

// Dummy mapping for visualization. In a real system, areas would have geo-coordinates.
const crimeTypeToLocationOffset = {
    'Theft': { lat: 0.001, lng: -0.001 },
    'Burglary': { lat: -0.001, lng: 0.001 },
    'Assault': { lat: 0.001, lng: 0.001 },
    'Vehicle crime': { lat: -0.001, lng: -0.001 },
    'Public order': { lat: 0, lng: 0.002 },
    'Other crime': { lat: 0, lng: -0.002 },
};

const AREA_CENTERS = {
    'City of London': { lat: 51.515, lng: -0.09 },
    'Westminster': { lat: 51.497, lng: -0.135 },
    'Camden': { lat: 51.529, lng: -0.125 },
    'Tower Hamlets': { lat: 51.509, lng: -0.005 },
    'Islington': { lat: 51.538, lng: -0.102 },
    'Hackney': { lat: 51.545, lng: -0.055 }
};

function simpleScale(value, domain, range) {
    const [d_min, d_max] = domain;
    const [r_min, r_max] = range;
    if (d_max === d_min) return r_min;
    const ratio = (value - d_min) / (d_max - d_min);
    return r_min + ratio * (r_max - r_min);
}

function ActivityHeatmap({ data, timestamp, area = 'City of London' }) {
  const pointsForTimestamp = data.filter(p => p.ts === timestamp);
  const areaCenter = AREA_CENTERS[area] || AREA_CENTERS['City of London'];

  if (!pointsForTimestamp || pointsForTimestamp.length === 0) {
    return <div style={{ padding: '1rem', textAlign: 'center', color: '#ffffff', fontSize: '1.1rem' }}>🌍 Move the time slider to see predictions on the map.</div>;
  }

  const maxIntensity = Math.max(...data.map(p => p.yhat), 1);

  return (
    <MapContainer center={[areaCenter.lat, areaCenter.lng]} zoom={14} style={{ height: '100%', width: '100%' }}>
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
      />
      {pointsForTimestamp.map((point, idx) => {
        const offset = crimeTypeToLocationOffset[point.crime_type] || { lat: 0, lng: 0 };
        const position = [areaCenter.lat + offset.lat, areaCenter.lng + offset.lng];
        const radius = simpleScale(point.yhat, [0, maxIntensity], [5, 25]);
        const opacity = simpleScale(point.yhat, [0, maxIntensity], [0.3, 0.8]);

        return (
          <CircleMarker key={idx} center={position} radius={radius} pathOptions={{ color: 'red', fillColor: 'red', fillOpacity: opacity }}>
            <Tooltip>
              {point.crime_type}: Predicted incidents: {point.yhat !== undefined && point.yhat !== null ? point.yhat.toFixed(2) : 'N/A'}
            </Tooltip>
          </CircleMarker>
        );
      })}
    </MapContainer>
  );
}

export default ActivityHeatmap;
