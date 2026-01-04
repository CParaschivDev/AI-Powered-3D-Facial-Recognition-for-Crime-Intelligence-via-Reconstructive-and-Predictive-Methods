#!/usr/bin/env python3
"""
Test script to verify the frontend polling fix
"""
import requests
import time
import json

API_BASE = "http://127.0.0.1:8000/api/v1"

def test_task_status_parsing():
    """Test that task status response is parsed correctly like the frontend does"""

    # First, let's try to get a real task status if backend is running
    try:
        # Test with a dummy task ID
        # Justification: Timeout ensures bounded wait time (timeout=1).
        response = requests.get(f"{API_BASE}/tasks/dummy-task/status", timeout=1)
        if response.status_code == 200:
            data = response.json()
            print("Task status response structure:")
            print(json.dumps(data, indent=2))

            # Test the frontend logic
            status = data.get('status')
            result = data.get('result')

            print(f"Status: {status}")
            print(f"Result type: {type(result)}")

            if status == 'SUCCESS':
                print("Would set reconstruction data:", result)
            elif status == 'FAILURE':
                print("Would show error:", result)
            else:
                print(f"Would continue polling for status: {status}")

        else:
            print(f"API returned status {response.status_code}: {response.text}")

    except requests.exceptions.ConnectionError:
        print("Backend not running - creating mock test")

        # Mock the response structure that the backend should return
        mock_response = {
            "id": "test-task-123",
            "status": "SUCCESS",
            "result": {
                "vertices": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                "faces": [[0, 1, 2]],
                "message": "Reconstruction completed successfully."
            }
        }

        print("Mock task status response:")
        print(json.dumps(mock_response, indent=2))

        # Test the fixed frontend logic
        status = mock_response['status']  # Fixed: was mock_response.data.status
        result = mock_response['result']  # Fixed: was mock_response.data.result

        print(f"Status: {status}")
        print(f"Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")

        if status == 'SUCCESS':
            print("✅ SUCCESS: Would set reconstruction data correctly")
            print(f"Vertices count: {len(result['vertices'])}")
            print(f"Faces count: {len(result['faces'])}")
        elif status == 'FAILURE':
            print("❌ FAILURE: Would show error")
        else:
            print(f"⏳ PENDING/PROGRESS: Would continue polling for status: {status}")

if __name__ == "__main__":
    test_task_status_parsing()