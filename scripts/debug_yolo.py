#!/usr/bin/env python3
"""
Debug YOLO detection on different images
"""
import cv2
import os
from pathlib import Path

# Add backend to path
import sys
sys.path.append('.')

from backend.services.yolo_detector import YoloWeaponDetector

def test_yolo_on_image(image_path, description):
    print(f"\n=== Testing {description} ===")
    print(f"Image: {image_path}")

    if not os.path.exists(image_path):
        print("❌ Image not found")
        return

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Failed to load image")
        return

    print(f"✅ Image loaded: {image.shape}")

    # Test YOLO detection
    detector = YoloWeaponDetector(conf_threshold=0.1)  # Very low threshold for testing
    detections = detector.detect(image)

    print(f"🔍 YOLO detections: {len(detections)}")
    for i, det in enumerate(detections):
        print(f"  {i+1}. {det['label']}: {det['confidence']:.3f} @ {det['bbox']}")

    if len(detections) == 0:
        print("⚠️  No detections found - trying even lower threshold")
        detector_low = YoloWeaponDetector(conf_threshold=0.01)
        detections_low = detector_low.detect(image)
        print(f"🔍 With 0.01 threshold: {len(detections_low)} detections")
        for i, det in enumerate(detections_low):
            print(f"  {i+1}. {det['label']}: {det['confidence']:.3f} @ {det['bbox']}")

def main():
    print("🔧 YOLO Detection Debug Tool")

    # Test on different images
    test_images = [
        ("backend/benchmarks/data/sample_face.jpg", "Sample Face (small crop)"),
        ("Data/actor_faces/Aaron_Eckhart/Aaron_Eckhart_105_83.jpeg", "Aaron Eckhart (full face)"),
    ]

    # Find more images
    data_dir = Path("Data/actor_faces")
    if data_dir.exists():
        for actor_dir in data_dir.iterdir():
            if actor_dir.is_dir():
                for img_file in actor_dir.iterdir():
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        test_images.append((str(img_file), f"{actor_dir.name} - {img_file.name}"))
                        break  # Just test one image per actor
                if len(test_images) >= 5:  # Limit to 5 test images
                    break

    for img_path, description in test_images:
        test_yolo_on_image(img_path, description)

if __name__ == "__main__":
    main()