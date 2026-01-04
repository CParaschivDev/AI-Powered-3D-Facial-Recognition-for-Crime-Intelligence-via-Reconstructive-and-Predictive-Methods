import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area } from 'recharts';

const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#0088FE', '#00C49F'];

function TemporalPatterns({ data }) {
  if (!data || data.length === 0) {
    return (
      <div className="temporal-patterns">
        <h4>Predicted Crime Trends</h4>
        <p>No prediction data available.</p>
      </div>
    );
  }

  // Process data for recharts: group by timestamp
  const chartData = data.reduce((acc, { ts, crime_type, yhat, yhat_lower, yhat_upper }) => {
    const date = new Date(ts).toLocaleDateString();
    let entry = acc.find(item => item.date === date);
    if (!entry) {
      entry = { date };
      acc.push(entry);
    }
    entry[crime_type] = yhat;
    // For confidence interval band
    entry[`${crime_type}_ci`] = [yhat_lower, yhat_upper];
    return acc;
  }, []);

  const crimeTypes = [...new Set(data.map(p => p.crime_type))];

  return (
    <div className="temporal-patterns">
      <h4>📈 Predicted Crime Trends</h4>
      <ResponsiveContainer width="100%" height={250}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis label={{ value: 'Predicted Incidents', angle: -90, position: 'insideLeft' }} />
          <Tooltip />
          <Legend />
          {crimeTypes.map((crimeType, index) => ([
            <Area
              key={`${crimeType}-ci`}
              type="monotone"
              dataKey={`${crimeType}_ci`}
              stroke="none"
              fill={COLORS[index % COLORS.length]}
              fillOpacity={0.2}
            />,
            <Line
              key={crimeType}
              type="monotone"
              dataKey={crimeType}
              stroke={COLORS[index % COLORS.length]}
              activeDot={{ r: 8 }} />
          ]))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default TemporalPatterns;
