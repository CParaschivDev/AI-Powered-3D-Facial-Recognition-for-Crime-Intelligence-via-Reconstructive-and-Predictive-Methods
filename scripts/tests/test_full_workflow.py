#!/usr/bin/env python3
"""
Test script to upload an image and monitor reconstruction task
"""
import requests
import time
import json
import os

# Backend URL
BASE_URL = "http://localhost:8000"

def test_reconstruction():
    print("🧪 Testing 3D face reconstruction workflow...")

    # Check if backend is running by trying docs endpoint
    try:
        # Justification: Timeout ensures bounded wait time (timeout=1).
        response = requests.get(f"{BASE_URL}/docs", timeout=1)
        if response.status_code != 200:
            print("❌ Backend not responding")
            return
        print("✅ Backend is running")
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")
        return

    # Find a test image
    test_image_path = None
    for root, dirs, files in os.walk("Data"):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_image_path = os.path.join(root, file)
                break
        if test_image_path:
            break

    if not test_image_path:
        print("❌ No test image found in Data directory")
        return

    print(f"📸 Using test image: {test_image_path}")

    # Upload image for reconstruction
    try:
        with open(test_image_path, 'rb') as f:
            files = {'file': f}
            data = {'case_id': 'test_case_001'}
            # Justification: Timeout ensures bounded wait time (timeout=5).
            response = requests.post(
                f"{BASE_URL}/api/v1/reconstruct",
                files=files,
                data=data,
                timeout=5,
            )

        if response.status_code not in [200, 202]:
            print(f"❌ Upload failed: {response.status_code} - {response.text}")
            return

        result = response.json()
        print(f"✅ Upload successful: {result}")

        task_id = result.get('task_id')
        if not task_id:
            print("❌ No task_id in response")
            return

        print(f"🔄 Monitoring task: {task_id}")

        # Poll for status
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                # Justification: Timeout ensures bounded wait time (timeout=1).
                status_response = requests.get(f"{BASE_URL}/api/v1/tasks/{task_id}/status", timeout=1)
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    status = status_data.get('status')
                    print(f"📊 Attempt {attempt+1}/{max_attempts}: Status = {status}")

                    if status == 'SUCCESS':
                        print("🎉 Reconstruction completed successfully!")
                        mesh_data = status_data.get('result', {})
                        print(f"📐 Mesh data keys: {list(mesh_data.keys()) if mesh_data else 'None'}")
                        return True
                    elif status == 'FAILURE':
                        print(f"❌ Reconstruction failed: {status_data.get('error')}")
                        return False
                    elif status in ['PENDING', 'PROGRESS']:
                        time.sleep(2)  # Wait 2 seconds before next check
                    else:
                        print(f"⚠️ Unknown status: {status}")
                        time.sleep(2)
                else:
                    print(f"❌ Status check failed: {status_response.status_code}")
                    time.sleep(2)

            except Exception as e:
                print(f"❌ Error checking status: {e}")
                time.sleep(2)

        print("⏰ Timeout: Task did not complete within expected time")
        return False

    except Exception as e:
        print(f"❌ Upload error: {e}")
        return False

if __name__ == "__main__":
    test_reconstruction()