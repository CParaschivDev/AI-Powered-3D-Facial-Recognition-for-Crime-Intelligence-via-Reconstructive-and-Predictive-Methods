# crime_scene_processor.py
import cv2
import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np
from pathlib import Path

from backend.services.yolo_detector import YoloWeaponDetector
from training.recognition_train import RecognitionNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- load recognition model ---
def load_recognition_model(ckpt_path: str, num_classes: int, embedding_size: int = 512, feature_dim: int = 256):
    model = RecognitionNet(num_classes=num_classes, embedding_size=embedding_size, feature_dim=feature_dim)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    
    # Handle different checkpoint formats
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    
    model.to(DEVICE)
    model.eval()
    return model

REC_CHECKPOINT = "./logs/recognition/recognition_model_best.pth"
REC_NUM_CLASSES = 530  # from training summary
REC_FEATURE_DIM = 256  # from training summary

rec_model = load_recognition_model(REC_CHECKPOINT, REC_NUM_CLASSES, feature_dim=REC_FEATURE_DIM)

# match your training transform as closely as possible
rec_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

yolo = YoloWeaponDetector(conf_threshold=0.1)  # Lower threshold for face detection

def _crop_person_upper(img_bgr, bbox, top_fraction=0.5, margin=0.1):
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    dw = int((x2 - x1) * margin)
    dh = int((y2 - y1) * margin)

    x1 = max(0, x1 - dw)
    x2 = min(w - 1, x2 + dw)
    y_mid = y1 + int((y2 - y1) * top_fraction)
    y1 = max(0, y1 - dh)
    y2 = min(h - 1, y_mid)

    return img_bgr[y1:y2, x1:x2]

def recognise_face_from_person_crop(person_bgr):
    face_bgr = _crop_person_upper(person_bgr, [0, 0, person_bgr.shape[1], person_bgr.shape[0]])
    if face_bgr.size == 0:
        return {"error": "empty_face_crop"}

    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(face_rgb)
    x = rec_transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        embedding = rec_model(x, return_embedding=True)
        logits = rec_model(x)

        probs = torch.softmax(logits, dim=1)
        pred_idx = int(torch.argmax(probs, dim=1)[0])
        pred_conf = float(torch.max(probs, dim=1)[0])

    return {
        "pred_class": pred_idx,
        "pred_conf": pred_conf,
        "embedding": embedding.cpu().numpy().flatten().tolist()
    }

def process_frame(frame_bgr, frame_id=None, camera_id=None, timestamp=None, location_id=None, yolo_conf_threshold=None, min_detection_conf=None, min_bbox_area_ratio=None):
    # Use provided threshold or default
    conf_threshold = yolo_conf_threshold if yolo_conf_threshold is not None else 0.1
    min_conf = min_detection_conf if min_detection_conf is not None else 0.01
    min_area_ratio = min_bbox_area_ratio if min_bbox_area_ratio is not None else 0.01

    # Create detector with the specified threshold
    from backend.services.yolo_detector import YoloWeaponDetector
    detector = YoloWeaponDetector(conf_threshold=conf_threshold)
    detections = detector.detect(frame_bgr)

    # Filter out detections below minimum confidence (likely noise)
    detections = [d for d in detections if d['confidence'] >= min_conf]

    # Additional filtering for bounding box size
    image_area = frame_bgr.shape[0] * frame_bgr.shape[1]
    detections = [d for d in detections if (d['bbox'][2] - d['bbox'][0]) * (d['bbox'][3] - d['bbox'][1]) / image_area >= min_area_ratio]

    persons = []
    weapons = []

    for det in detections:
        label = det["label"]
        bbox = det["bbox"]
        conf = det["confidence"]

        if label == "person":
            x1, y1, x2, y2 = [int(v) for v in bbox]
            person_crop = frame_bgr[y1:y2, x1:x2]
            rec_info = recognise_face_from_person_crop(person_crop)
            persons.append({
                "bbox": bbox,
                "confidence": conf,
                "recognition": rec_info
            })

        elif label in ("knife", "baseball bat"):
            weapons.append({
                "type": label,
                "bbox": bbox,
                "confidence": conf
            })

    return {
        "frame_id": frame_id,
        "camera_id": camera_id,
        "timestamp": timestamp,
        "location_id": location_id,
        "persons": persons,
        "weapons": weapons
    }