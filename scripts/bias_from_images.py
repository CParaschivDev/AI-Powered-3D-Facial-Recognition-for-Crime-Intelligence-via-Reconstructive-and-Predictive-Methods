#!/usr/bin/env python3
"""
Bias evaluation from raw images using ONLY the trained model.
NO embeddings - loads images directly and runs through RecognitionNet.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import glob
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.bias_monitor import BiasMonitor
from backend.database.dependencies import SessionLocal
from backend.database.models import PredictiveLog
import json


class RecognitionNet(nn.Module):
    """Recognition model architecture (must match training)"""
    def __init__(self, num_classes, embedding_size=512, feature_dim=256):
        super(RecognitionNet, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((112, 112))
        self.embedding_layer = nn.Linear(64 * 112 * 112, embedding_size)
        
        # Feature selector layer
        if feature_dim is not None and feature_dim < embedding_size:
            self.feature_selector_layer = nn.Linear(embedding_size, feature_dim)
            classifier_input_dim = feature_dim
        else:
            self.feature_selector_layer = None
            classifier_input_dim = embedding_size
        
        self.classifier = nn.Linear(classifier_input_dim, num_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        embedding = self.embedding_layer(x)
        
        if self.feature_selector_layer is not None:
            features = self.feature_selector_layer(embedding)
        else:
            features = embedding
        
        logits = self.classifier(features)
        return logits


def load_facescrub_images(data_dir, split_ratio=0.8):
    """
    Load FaceScrub images from actor_faces and actress_faces.
    
    Returns:
        image_paths, labels, group_labels (0=male, 1=female)
    """
    print("\n[1/3] Loading FaceScrub dataset structure...")
    
    actor_dir = os.path.join(data_dir, "actor_faces")
    actress_dir = os.path.join(data_dir, "actress_faces")
    
    image_paths = []
    labels = []
    group_labels = []
    
    label_map = {}
    current_label = 0
    
    # Load actors (male = 0)
    print("  Loading actor images...")
    for identity_dir in sorted(os.listdir(actor_dir)):
        identity_path = os.path.join(actor_dir, identity_dir)
        if not os.path.isdir(identity_path):
            continue
        
        if identity_dir not in label_map:
            label_map[identity_dir] = current_label
            current_label += 1
        
        for img_file in glob.glob(os.path.join(identity_path, "*")):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append(img_file)
                labels.append(label_map[identity_dir])
                group_labels.append(0)  # male
    
    # Load actresses (female = 1)
    print("  Loading actress images...")
    for identity_dir in sorted(os.listdir(actress_dir)):
        identity_path = os.path.join(actress_dir, identity_dir)
        if not os.path.isdir(identity_path):
            continue
        
        if identity_dir not in label_map:
            label_map[identity_dir] = current_label
            current_label += 1
        
        for img_file in glob.glob(os.path.join(identity_path, "*")):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append(img_file)
                labels.append(label_map[identity_dir])
                group_labels.append(1)  # female
    
    print(f"  ✓ Loaded {len(image_paths)} images from {len(label_map)} identities")
    print(f"  ✓ Males: {sum(1 for g in group_labels if g == 0)} images")
    print(f"  ✓ Females: {sum(1 for g in group_labels if g == 1)} images")
    
    return image_paths, np.array(labels), np.array(group_labels)


def run_model_inference(model, image_paths, device='cpu'):
    """
    Run model inference on images.
    
    Returns:
        predictions array
    """
    print("\n[2/3] Running model inference on images...")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    predictions = []
    batch_size = 32
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            if i % 500 == 0:
                print(f"  Progress: {i}/{len(image_paths)}")
            
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            
            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = transform(img)
                    batch_images.append(img_tensor)
                except Exception as e:
                    print(f"  ⚠️  Error loading {img_path}: {e}")
                    batch_images.append(torch.zeros(3, 224, 224))
            
            if batch_images:
                batch_tensor = torch.stack(batch_images).to(device)
                logits = model(batch_tensor)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(preds)
    
    print(f"  ✓ Generated {len(predictions)} predictions from MODEL")
    return np.array(predictions)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Bias Evaluation from Raw Images")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.pth)")
    parser.add_argument("--data-dir", type=str, default="./Data", help="Root data directory")
    parser.add_argument("--output-dir", type=str, default="./logs/bias", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("BIAS MONITORING - From Raw Images Using Model ONLY")
    print("=" * 70)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"\n✓ Loading model from {args.model}")
    checkpoint = torch.load(args.model, map_location=args.device, weights_only=False)
    num_classes = checkpoint.get('num_classes', 530)
    
    model = RecognitionNet(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()
    print(f"✓ Model loaded: {num_classes} classes")
    
    # Load images
    image_paths, labels, group_labels = load_facescrub_images(args.data_dir)
    
    # Run inference
    predictions = run_model_inference(model, image_paths, args.device)
    
    # Compute bias metrics
    print("\n[3/3] Computing bias metrics...")
    monitor = BiasMonitor(threshold=0.05)
    
    group_names = {0: "Actors (Male)", 1: "Actresses (Female)"}
    report = monitor.generate_bias_report(
        predictions=predictions,
        labels=labels,
        group_labels=group_labels,
        model_name="RecognitionNet",
        group_names=group_names
    )
    
    # Save report
    output_path = os.path.join(args.output_dir, "bias_report_from_images.json")
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Log to database
    try:
        db = SessionLocal()
        log_entry = PredictiveLog(
            model_name=report['model_name'],
            prediction_type='bias_audit',
            prediction_data=json.dumps(report),
            bias_score=report['overall_bias_score'],
            is_flagged=report['is_flagged'],
            audit_status='pending'
        )
        db.add(log_entry)
        db.commit()
        print(f"✓ Logged to database (ID: {log_entry.id})")
        db.close()
    except Exception as e:
        print(f"⚠️  Database logging failed: {e}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("BIAS REPORT SUMMARY (FROM RAW IMAGES)")
    print("=" * 70)
    print(f"Model: {report['model_name']}")
    print(f"Total Samples: {report['total_samples']}")
    print(f"Demographic Groups: {report['num_groups']}")
    print(f"\nBias Score: {report['overall_bias_score']:.4f}")
    print(f"Threshold: {report['threshold']:.4f}")
    print(f"Flagged: {'⚠️  YES' if report['is_flagged'] else '✓ NO'}")
    
    print("\nGroup Performance:")
    for group_name, metrics in report['fairness_metrics']['accuracy_gap']['group_metrics'].items():
        print(f"  {group_name}:")
        print(f"    Accuracy: {metrics['accuracy']:.4f}")
        print(f"    F1 Score: {metrics['f1']:.4f}")
        print(f"    Samples: {metrics['sample_count']}")
    
    print(f"\n📊 Recommendation:")
    print(f"  {report['recommendation']}")
    print(f"\n✓ Report saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
