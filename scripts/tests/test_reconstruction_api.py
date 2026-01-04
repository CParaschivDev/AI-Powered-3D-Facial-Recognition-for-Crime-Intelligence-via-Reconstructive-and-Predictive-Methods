import requests
import json

url = 'http://localhost:8000/models/evaluate/file/logs/reconstruction/reconstruction_model.pth'
try:
    response = requests.get(url)
    print(f'Status: {response.status_code}')
    if response.status_code == 200:
        data = response.json()
        print('Success! Parsed JSON:')
        print(f'Model type: {data.get("model_info", {}).get("evaluation_type")}')
        print(f'MSE: {data.get("mse")}')
        print(f'MAE: {data.get("mae")}')
        print(f'R2: {data.get("r2")}')
        print(f'Num samples: {data.get("num_samples")}')
    else:
        print(f'Error: {response.text}')
except Exception as e:
    print(f'Exception: {e}')