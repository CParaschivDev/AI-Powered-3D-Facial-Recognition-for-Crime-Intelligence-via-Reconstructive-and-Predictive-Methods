#!/usr/bin/env python3
"""
Test script for YOLO + Recognition integration
"""
import cv2
import json
from pathlib import Path

# Add backend to path
import sys
sys.path.append('.')

from backend.services.crime_scene_processor import process_frame

def main():
    # Test with an actor face image that might have more context
    img_path = "Data/actor_faces/Aaron_Eckhart/Aaron_Eckhart_105_83.jpeg"
    frame = cv2.imread(img_path)

    if frame is None:
        print(f"Could not load image: {img_path}")
        return

    print(f"Loaded image: {frame.shape}")

    # Process the frame
    record = process_frame(
        frame_bgr=frame,
        frame_id=1,
        camera_id="TestCam1",
        timestamp="2025-12-02T12:00:00Z",
        location_id="TestLocation"
    )

    # Create logs directory
    Path("logs/inference").mkdir(parents=True, exist_ok=True)

    # Save result
    with open("logs/inference/yolo_actor_test_record.json", "w") as f:
        json.dump(record, f, indent=2)

    print("Saved logs/inference/yolo_actor_test_record.json")
    print("Record summary:")
    print(f"- Persons detected: {len(record['persons'])}")
    print(f"- Weapons detected: {len(record['weapons'])}")

    for i, person in enumerate(record['persons']):
        print(f"Person {i+1}: bbox={person['bbox']}, conf={person['confidence']:.2f}")
        rec = person['recognition']
        if 'error' not in rec:
            print(f"  Recognition: class={rec['pred_class']}, conf={rec['pred_conf']:.2f}")
        else:
            print(f"  Recognition error: {rec['error']}")

    for i, weapon in enumerate(record['weapons']):
        print(f"Weapon {i+1}: type={weapon['type']}, bbox={weapon['bbox']}, conf={weapon['confidence']:.2f}")

if __name__ == "__main__":
    main()