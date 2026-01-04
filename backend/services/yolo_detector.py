# yolo_detector.py
from ultralytics import YOLO
import numpy as np

TARGET_CLASSES = {"person", "knife", "baseball bat"}

class YoloWeaponDetector:
    def __init__(self, model_name: str = "yolov8s.pt", conf_threshold: float = 0.3):
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold

    def detect(self, image):
        """
        image: np.ndarray (BGR, as from cv2.imread or cv2.VideoCapture)
        returns: list of dicts with {label, confidence, bbox}
        """
        results = self.model(image, conf=self.conf_threshold)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = results.names[cls_id]
            conf = float(box.conf[0])

            if label not in TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()

            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })

        return detections