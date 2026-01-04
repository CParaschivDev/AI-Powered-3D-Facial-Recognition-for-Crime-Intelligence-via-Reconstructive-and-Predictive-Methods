import requests

response = requests.get('http://127.0.0.1:8000/api/v1/models/evaluate/file/logs/reconstruction/reconstruction_model.pth')
print('Status Code:', response.status_code)
if response.status_code == 200:
    print('Success!')
    data = response.json()
    print(f'Model type: {data.get("model_info", {}).get("evaluation_type")}')
    print(f'MSE: {data.get("mse")}')
    print(f'MAE: {data.get("mae")}')
    print(f'R2: {data.get("r2")}')
else:
    print('Error Response:')
    print(response.text)