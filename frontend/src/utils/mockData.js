// Mock data for development purposes
export const mockPredictions = {
  "predictions": [
    {
      "ts": "2025-08-01T09:00:00Z",
      "crime_type": "Theft",
      "yhat": 2.5,
      "yhat_lower": 1.8,
      "yhat_upper": 3.2
    },
    {
      "ts": "2025-08-15T10:30:00Z",
      "crime_type": "Assault",
      "yhat": 1.2,
      "yhat_lower": 0.9,
      "yhat_upper": 1.5
    },
    {
      "ts": "2025-09-01T14:15:00Z",
      "crime_type": "Burglary",
      "yhat": 3.1,
      "yhat_lower": 2.5,
      "yhat_upper": 3.7
    },
    {
      "ts": "2025-09-15T18:45:00Z",
      "crime_type": "Theft",
      "yhat": 4.2,
      "yhat_lower": 3.8,
      "yhat_upper": 4.6
    },
    {
      "ts": "2025-10-01T22:30:00Z",
      "crime_type": "Assault",
      "yhat": 2.8,
      "yhat_lower": 2.1,
      "yhat_upper": 3.5
    },
    {
      "ts": "2025-10-15T12:00:00Z",
      "crime_type": "Vehicle crime",
      "yhat": 1.8,
      "yhat_lower": 1.3,
      "yhat_upper": 2.3
    },
    {
      "ts": "2025-11-01T16:20:00Z",
      "crime_type": "Public order",
      "yhat": 3.5,
      "yhat_lower": 2.9,
      "yhat_upper": 4.1
    }
  ]
};

export const mockAlerts = [
  { id: 'SUS-34', camera: 'Cam-14', time: '22:08:48', confidence: 0.94 },
  { id: 'SUS-45', camera: 'Cam-13', time: '22:08:40', confidence: 0.92 },
  { id: 'SUS-70', camera: 'Cam-15', time: '22:08:32', confidence: 0.88 },
  { id: 'SUS-98', camera: 'Cam-7', time: '22:08:24', confidence: 0.86 },
  { id: 'SUS-98', camera: 'Cam-15', time: '22:08:16', confidence: 0.81 }
];