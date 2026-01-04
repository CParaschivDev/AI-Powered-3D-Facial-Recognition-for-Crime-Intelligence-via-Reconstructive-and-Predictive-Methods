#!/usr/bin/env python3
"""
Test the YOLO API endpoint directly
"""
import requests
import base64
import cv2
import numpy as np

def test_api_endpoint():
    # Load test image - try a larger image from the gallery
    img_path = "Data/QMUL-SurvFace/Face_Identification_Test_Set/gallery/1006_cam2_1.jpg"
    img = cv2.imread(img_path)

    if img is None:
        print("Could not load test image")
        return

    print(f"Loaded image: {img.shape}")

    # Convert to bytes
    success, encoded_img = cv2.imencode('.jpg', img)
    if not success:
        print("Could not encode image")
        return

    img_bytes = encoded_img.tobytes()

    # Prepare form data
    files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
    data = {
        'camera_id': 'test_camera',
        'location_id': 'test_location',
        'frame_id': '123'
    }

    try:
        # Make API call
        response = requests.post(
            'http://localhost:8000/api/v1/streams/process-frame',
            files=files,
            data=data
        )

        print(f"API Response Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("API Response:")
            print(f"  - Persons: {len(result.get('persons', []))}")
            print(f"  - Weapons: {len(result.get('weapons', []))}")
            print(f"  - Camera: {result.get('camera_id')}")
            print(f"  - Location: {result.get('location_id')}")
            print(f"  - Image Quality: {result.get('image_quality', {})}")

            if result.get('persons'):
                for i, person in enumerate(result['persons']):
                    print(f"    Person {i+1}: conf={person['confidence']:.2f}")
        else:
            print(f" API Error: {response.text}")

    except requests.exceptions.ConnectionError:
        print("Cannot connect to backend server. Is it running?")
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_api_endpoint()