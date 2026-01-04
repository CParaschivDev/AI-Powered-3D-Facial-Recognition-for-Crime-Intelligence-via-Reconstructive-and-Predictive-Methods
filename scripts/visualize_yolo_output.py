#!/usr/bin/env python3
"""
Visualise the JSON output from scripts/test_yolo_integration.py
- reads logs/inference/yolo_actor_test_record.json
- draws person + weapon bboxes and confidences onto the same image
- writes docs/figures/yolo_actor_test_vis.png

Run:
python scripts/test_yolo_integration.py    # generates logs/inference/yolo_actor_test_record.json
python scripts/visualize_yolo_output.py
"""
import json
import os
import sys
import cv2

# Ensure project root is on sys.path so imports like `backend.*` work when
# running the script directly (python scripts/visualize_yolo_output.py).
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

IN_JSON = os.path.join('logs', 'inference', 'yolo_actor_test_record.json')
OUT_DIR = os.path.join('docs', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# Path used by the test script; change if needed
IMG_PATH = "Data/actor_faces/Aaron_Eckhart/Aaron_Eckhart_105_83.jpeg"

if not os.path.exists(IN_JSON):
    raise SystemExit(f"Missing JSON output: {IN_JSON}. Run scripts/test_yolo_integration.py first.")

with open(IN_JSON, 'r') as f:
    record = json.load(f)

img = cv2.imread(IMG_PATH)
if img is None:
    raise SystemExit(f"Could not load image at {IMG_PATH}")

# Prepare overlay for translucent boxes
overlay = img.copy()
alpha = 0.35

# Simple color map
COLORS = {
    'person': (200, 120, 40),      # orange-ish (BGR)
    'knife': (30, 30, 200),        # red-ish
    'baseball bat': (20, 160, 20)  # green-ish
}

drawn_classes = set()

def draw_label(img, text, x, y, bg_color, text_color=(255,255,255)):
    """Draw label with filled background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    pad = 4
    cv2.rectangle(img, (x, y - h - pad), (x + w + pad, y + 2), bg_color, -1)
    cv2.putText(img, text, (x + 2, y - 2), font, scale, text_color, thickness, cv2.LINE_AA)

def draw_conf_bar(img, x1, y2, x2, conf, color):
    """Draw a small confidence bar below bbox (left-aligned)."""
    bar_h = 6
    total_w = min(80, x2 - x1)
    filled_w = int(total_w * conf)
    y = min(img.shape[0]-1, y2 + 8)
    cv2.rectangle(img, (x1, y), (x1 + total_w, y + bar_h), (50,50,50), -1)
    cv2.rectangle(img, (x1, y), (x1 + filled_w, y + bar_h), color, -1)

# Draw persons
for p in record.get('persons', []):
    x1, y1, x2, y2 = [int(v) for v in p['bbox']]
    conf = float(p.get('confidence', 0.0))
    color = COLORS.get('person', (255,0,0))

    # translucent fill on overlay
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

    # strong border on original
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    label = f"Person {conf:.2f}"
    draw_label(img, label, x1, max(12, y1+15), color)
    draw_conf_bar(img, x1, y2, x2, conf, color)

    rec = p.get('recognition', {})
    if 'pred_class' in rec:
        rec_label = f"ID:{rec['pred_class']} {rec['pred_conf']:.2f}"
        draw_label(img, rec_label, x1, min(y2 + 32, img.shape[0]-6), color)

    drawn_classes.add('person')

# Draw weapons
for w in record.get('weapons', []):
    x1, y1, x2, y2 = [int(v) for v in w['bbox']]
    conf = float(w.get('confidence', 0.0))
    label_type = w.get('type', 'weapon')
    color = COLORS.get(label_type, (0,0,255))

    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = f"{label_type} {conf:.2f}"
    draw_label(img, label, x1, max(12, y1+15), color)
    draw_conf_bar(img, x1, y2, x2, conf, color)

    drawn_classes.add(label_type)

# Blend overlay
cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

# Legend
legend_x, legend_y = 12, 12
line_h = 20
cv2.rectangle(img, (legend_x-6, legend_y-6), (legend_x+200, legend_y + 120), (240,240,240), -1)
cv2.putText(img, "Legend:", (legend_x, legend_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10,10,10), 1)
ly = legend_y + 28
for i, (cls, col) in enumerate(COLORS.items()):
    if cls not in drawn_classes:
        continue
    cv2.rectangle(img, (legend_x+4, ly + i*line_h), (legend_x+20, ly + 12 + i*line_h), col, -1)
    cv2.putText(img, f"{cls}", (legend_x+28, ly + 12 + i*line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10,10,10), 1)

# Save full annotated image (resized for easier viewing if very large)
MAX_WIDTH = 1200
save_full = img.copy()
if save_full.shape[1] > MAX_WIDTH:
    scale = MAX_WIDTH / save_full.shape[1]
    save_full = cv2.resize(save_full, (int(save_full.shape[1]*scale), int(save_full.shape[0]*scale)), interpolation=cv2.INTER_AREA)

full_out = os.path.join(OUT_DIR, 'yolo_actor_test_vis.png')
cv2.imwrite(full_out, save_full)

# Save a zoomed crop around all detections (for clearer screenshot)
all_boxes = []
for p in record.get('persons', []):
    all_boxes.append([int(v) for v in p['bbox']])
for w in record.get('weapons', []):
    all_boxes.append([int(v) for v in w['bbox']])

if all_boxes:
    x1s = [b[0] for b in all_boxes]
    y1s = [b[1] for b in all_boxes]
    x2s = [b[2] for b in all_boxes]
    y2s = [b[3] for b in all_boxes]
    bx1, by1, bx2, by2 = min(x1s), min(y1s), max(x2s), max(y2s)
    pad = int(0.12 * max(bx2 - bx1, by2 - by1))
    bx1 = max(0, bx1 - pad)
    by1 = max(0, by1 - pad)
    bx2 = min(img.shape[1]-1, bx2 + pad)
    by2 = min(img.shape[0]-1, by2 + pad)

    crop = img[by1:by2, bx1:bx2]
    # Resize crop to reasonable width for screenshot
    CROP_W = 900
    if crop.shape[1] > CROP_W:
        scale = CROP_W / crop.shape[1]
        crop = cv2.resize(crop, (int(crop.shape[1]*scale), int(crop.shape[0]*scale)), interpolation=cv2.INTER_AREA)

    crop_out = os.path.join(OUT_DIR, 'yolo_actor_test_vis_zoom.png')
    cv2.imwrite(crop_out, crop)
    print(f"Saved zoomed visualization to {crop_out}")
else:
    print("No detections to create zoomed crop.")

print(f"Saved full annotation to {full_out}")
