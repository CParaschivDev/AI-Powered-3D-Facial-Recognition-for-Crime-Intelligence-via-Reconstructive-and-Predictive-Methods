#!/usr/bin/env python3
"""
Bias evaluation using the trained RecognitionNet model's classifier.
This loads the model and runs predictions through the classifier layer.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.bias_monitor import BiasMonitor
from backend.database.dependencies import SessionLocal
from backend.database.models import PredictiveLog
from datetime import datetime, timezone
import json


class RecognitionNet(nn.Module):
    """Recognition model architecture (must match training)"""
    def __init__(self, num_classes, embedding_size=512):
        super(RecognitionNet, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((112, 112))
        self.embedding_layer = nn.Linear(64 * 112 * 112, embedding_size)
        self.classifier = nn.Linear(embedding_size, num_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        embedding = self.embedding_layer(x)
        logits = self.classifier(embedding)
        return logits


def get_model_predictions(model_path, embeddings_path):
    """
    Load model and get predictions using the classifier layer.
    
    Args:
        model_path: Path to trained model checkpoint
        embeddings_path: Path to test embeddings
        
    Returns:
        predictions array
    """
    print(f"\n✓ Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    num_classes = checkpoint.get('num_classes', 530)
    
    model = RecognitionNet(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded: {num_classes} classes")
    print(f"✓ Loading embeddings from {embeddings_path}")
    
    embeddings = np.load(embeddings_path)
    embeddings_tensor = torch.from_numpy(embeddings).float()
    
    print(f"✓ Running classifier on {len(embeddings)} embeddings...")
    
    predictions = []
    batch_size = 256
    
    with torch.no_grad():
        for i in range(0, len(embeddings_tensor), batch_size):
            if i % 1000 == 0:
                print(f"  Progress: {i}/{len(embeddings_tensor)}")
            
            batch = embeddings_tensor[i:i+batch_size]
            logits = model.classifier(batch)
            preds = torch.argmax(logits, dim=1).numpy()
            predictions.extend(preds)
    
    print(f"✓ Generated {len(predictions)} predictions using MODEL CLASSIFIER\n")
    return np.array(predictions)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Bias Evaluation Using Trained Model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.pth)")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to test embeddings")
    parser.add_argument("--labels", type=str, required=True, help="Path to test labels")
    parser.add_argument("--data-dir", type=str, default="./Data", help="Root data directory")
    parser.add_argument("--output-dir", type=str, default="./logs/bias", help="Output directory")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("BIAS MONITORING - Using Trained Model Classifier")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load labels
    labels = np.load(args.labels)
    
    # Get predictions from model
    predictions = get_model_predictions(args.model, args.embeddings)
    
    # Create gender indices
    actor_indices_path = os.path.join(args.output_dir, "actor_indices.npy")
    actress_indices_path = os.path.join(args.output_dir, "actress_indices.npy")
    
    if not os.path.exists(actor_indices_path):
        from scripts.automate_bias_evaluation import create_gender_split_indices
        create_gender_split_indices(args.data_dir, args.output_dir)
    
    actor_indices = np.load(actor_indices_path)
    actress_indices = np.load(actress_indices_path)
    
    # Create group labels
    group_labels = np.zeros(len(labels), dtype=int)
    group_labels[actress_indices] = 1
    
    # Compute bias metrics
    print("Computing bias metrics...")
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
    output_path = os.path.join(args.output_dir, "bias_report_model.json")
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
    print("BIAS REPORT SUMMARY")
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
