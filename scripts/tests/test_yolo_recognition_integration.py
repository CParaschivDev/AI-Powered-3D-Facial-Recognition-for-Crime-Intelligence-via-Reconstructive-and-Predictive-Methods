#!/usr/bin/env python3
"""
Test script to verify YOLO integration in recognition endpoint
"""
import requests
import json
import os
import sys

# Backend URL
BASE_URL = "http://localhost:8000"

def test_recognition_with_yolo():
    print("🧪 Testing recognition endpoint with YOLO integration...")

    # Check if backend is running
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=2)
        if response.status_code != 200:
            print("❌ Backend not responding")
            return False
        print("✅ Backend is running")
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")
        return False

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
        return False

    print(f"📸 Using test image: {test_image_path}")

    # Test recognition endpoint
    try:
        with open(test_image_path, 'rb') as f:
            files = {'file': f}
            data = {
                'case_id': 'test_yolo_integration',
                'location': 'test_location',
                'yolo_conf_threshold': '0.1'
            }
            response = requests.post(
                f"{BASE_URL}/api/v1/recognize",
                files=files,
                data=data,
                timeout=30,  # Increased timeout for YOLO + recognition processing
            )

        print(f"📡 Response status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("✅ Recognition successful!")
            print(f"🎯 Best match ID: {result.get('best_id')}")
            print(f"📊 Cosine score: {result.get('cosine_score')}")
            print(f"⚖️ Verdict: {result.get('verdict')}")

            # Check YOLO results
            yolo_results = result.get('yolo_results', {})
            if yolo_results:
                persons = yolo_results.get('persons', [])
                weapons = yolo_results.get('weapons', [])
                total_detections = yolo_results.get('total_detections', 0)

                print(f"🔍 YOLO Results:")
                print(f"   👥 Persons detected: {len(persons)}")
                print(f"   🔫 Weapons detected: {len(weapons)}")
                print(f"   📈 Total detections: {total_detections}")

                if persons:
                    print("   📋 Person details:")
                    for i, person in enumerate(persons):
                        bbox = person.get('bbox', [])
                        conf = person.get('confidence', 0)
                        recognition = person.get('recognition', {})
                        print(f"      Person {i+1}: bbox={bbox}, conf={conf:.3f}")
                        if recognition:
                            pred_class = recognition.get('pred_class')
                            pred_conf = recognition.get('pred_conf')
                            print(f"         Recognition: ID={pred_class}, conf={pred_conf:.3f}")

                if weapons:
                    print("   🔫 Weapon details:")
                    for i, weapon in enumerate(weapons):
                        bbox = weapon.get('bbox', [])
                        conf = weapon.get('confidence', 0)
                        weapon_type = weapon.get('weapon_type', 'unknown')
                        print(f"      Weapon {i+1}: {weapon_type}, bbox={bbox}, conf={conf:.3f}")

                return True
            else:
                print("❌ No YOLO results found in response")
                return False
        else:
            print(f"❌ Recognition failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Error during recognition test: {e}")
        return False

if __name__ == "__main__":
    success = test_recognition_with_yolo()
    sys.exit(0 if success else 1)