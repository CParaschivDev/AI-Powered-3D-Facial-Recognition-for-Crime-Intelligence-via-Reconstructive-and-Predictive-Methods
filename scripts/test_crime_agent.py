#!/usr/bin/env python3
"""
Comprehensive test of the Crime Intelligence Agent
"""
import cv2
import json
from pathlib import Path

# Add backend to path
import sys
sys.path.append('.')

from backend.services.crime_intelligence_agent import CrimeIntelligenceAgent

def main():
    agent = CrimeIntelligenceAgent()

    # Test 1: Process a frame
    print("=== Test 1: Processing CCTV Frame ===")
    img_path = "Data/actor_faces/Aaron_Eckhart/Aaron_Eckhart_105_83.jpeg"
    frame = cv2.imread(img_path)

    if frame is None:
        print(f"Could not load image: {img_path}")
        return

    print(f"Loaded image: {frame.shape}")

    # Process frame
    record = agent.process_cctv_frame(
        frame_bgr=frame,
        frame_id=1,
        camera_id="TestCam1",
        timestamp="2025-12-02T12:00:00Z",
        location_id="Downtown"
    )

    print("Processing complete!")
    print(f"- Storage success: {record['storage_success']}")
    print(f"- Persons detected: {len(record['persons'])}")
    print(f"- Weapons detected: {len(record['weapons'])}")

    # Test 2: Query events
    print("\n=== Test 2: Querying Events ===")
    events = agent.query_events(camera_id="TestCam1")
    print(f"Found {len(events)} events for TestCam1")

    for i, event in enumerate(events):
        print(f"Event {i}: {len(event['persons'])} persons, {len(event['weapons'])} weapons")

    # Test 3: Test individual components
    print("\n=== Test 3: Testing Individual Components ===")

    # Test object detection
    detections = agent.detect_objects(frame)
    print(f"YOLO detected {len(detections)} objects:")
    for det in detections:
        print(f"  - {det['label']}: {det['confidence']:.2f}")

    # Test face recognition (using person crop)
    if detections:
        # Get first person detection
        person_det = next((d for d in detections if d['label'] == 'person'), None)
        if person_det:
            x1, y1, x2, y2 = [int(v) for v in person_det['bbox']]
            person_crop = frame[y1:y2, x1:x2]
            face_result = agent.recognise_face(person_crop)
            print(f"Face recognition result: class={face_result.get('pred_class', 'N/A')}, conf={face_result.get('pred_conf', 0):.2f}")

    # Test 4: Save comprehensive results
    print("\n=== Test 4: Saving Results ===")
    Path("logs/inference").mkdir(parents=True, exist_ok=True)

    results = {
        "frame_processing": record,
        "query_results": events,
        "component_tests": {
            "yolo_detections": len(detections),
            "face_recognition_available": bool(detections)
        }
    }

    with open("logs/inference/agent_test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("Saved comprehensive test results to logs/inference/agent_test_results.json")
    print("\nCrime Intelligence Agent integration test complete!")

if __name__ == "__main__":
    main()