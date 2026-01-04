import requests
import json
import time
import sys
import os
from backend.core.process_utils import run_detached_process

# Start the server
print("Starting server...")
server_process = run_detached_process([
    sys.executable, "-c", 
    "from backend.api.main import app; import uvicorn; uvicorn.run(app, host='127.0.0.1', port=8000, log_level='info')"
], cwd=os.getcwd())

# Wait for server to start by polling the docs endpoint (avoid fixed sleeps)
print("Waiting for server to start (polling /docs)...")
max_wait = 30
started = False
for i in range(max_wait):
    if server_process.poll() is not None:
        print(f"Server process exited early with code {server_process.returncode}")
        sys.exit(1)
    try:
        # Justification: Timeout ensures bounded wait time (timeout=1).
        r = requests.get('http://localhost:8000/docs', timeout=1)
        if r.status_code == 200:
            started = True
            break
    except Exception:
        pass
    time.sleep(1)

if not started:
    print("Server did not start within timeout; check logs or increase wait time")
    try:
        server_process.terminate()
    except Exception:
        pass
    sys.exit(1)

print("Server appears to be running...")

try:
    # Test the models endpoint
    # Justification: Timeout ensures bounded wait time (timeout=3).
    response = requests.get('http://localhost:8000/api/v1/models', timeout=3)
    if response.status_code == 200:
        models = response.json()
        print('Models endpoint working:')
        print(json.dumps(models, indent=2))
    else:
        print(f'Models endpoint failed: {response.status_code}')

    # Test evaluation results for recognition
    # Justification: Timeout ensures bounded wait time (timeout=3).
    response = requests.get('http://localhost:8000/api/v1/models/evaluate/recognition/results', timeout=3)
    if response.status_code == 200:
        results = response.json()
        print('\nRecognition evaluation results:')
        for dataset, metrics in results.items():
            print(f'Dataset: {dataset}')
            print(f'F1 Score: {metrics.get("f1_macro", "N/A")}')
            print(f'Accuracy: {metrics.get("accuracy", "N/A")}')
            cm = metrics.get('confusion_matrix', [])
            print(f'Confusion Matrix shape: {len(cm)}x{len(cm[0]) if cm else 0}')
            print()
    else:
        print(f'Recognition results endpoint failed: {response.status_code}')

    # Test evaluation results for reconstruction
    # Justification: Timeout ensures bounded wait time (timeout=3).
    response = requests.get('http://localhost:8000/api/v1/models/evaluate/reconstruction/results', timeout=3)
    if response.status_code == 200:
        results = response.json()
        print('\nReconstruction evaluation results:')
        for dataset, metrics in results.items():
            print(f'Dataset: {dataset}')
            print(f'MSE: {metrics.get("mse", "N/A")}')
            print()
    else:
        print(f'Reconstruction results endpoint failed: {response.status_code}')

finally:
    # Stop the server
    print("Stopping server...")
    try:
        server_process.terminate()
        server_process.wait(timeout=10)
    except Exception:
        try:
            server_process.kill()
        except Exception:
            pass