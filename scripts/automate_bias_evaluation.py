#!/usr/bin/env python3
"""
Automated bias evaluation for facial recognition models.
Integrates with existing evaluation pipeline to compute fairness metrics.
"""

import sys
import os
import numpy as np
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.bias_monitor import BiasMonitor, evaluate_facescrub_gender_bias
from backend.database.dependencies import SessionLocal
from backend.database.models import PredictiveLog
from datetime import datetime, timezone


def create_gender_split_indices(data_dir: str, output_dir: str = "./logs/bias"):
    """
    Create index files for actor/actress split from FaceScrub dataset structure.
    
    Args:
        data_dir: Root data directory containing actor_faces and actress_faces
        output_dir: Directory to save index files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    actor_dir = os.path.join(data_dir, "actor_faces")
    actress_dir = os.path.join(data_dir, "actress_faces")
    
    # Count images in each directory
    actor_count = 0
    actress_count = 0
    
    if os.path.exists(actor_dir):
        for identity_dir in os.listdir(actor_dir):
            identity_path = os.path.join(actor_dir, identity_dir)
            if os.path.isdir(identity_path):
                actor_count += len([f for f in os.listdir(identity_path) if f.endswith(('.jpg', '.png'))])
    
    if os.path.exists(actress_dir):
        for identity_dir in os.listdir(actress_dir):
            identity_path = os.path.join(actress_dir, identity_dir)
            if os.path.isdir(identity_path):
                actress_count += len([f for f in os.listdir(identity_path) if f.endswith(('.jpg', '.png'))])
    
    total = actor_count + actress_count
    
    # Create indices (first N are actors, rest are actresses based on dataset loading order)
    # This assumes dataset loader processes actor_faces before actress_faces
    actor_indices = np.arange(actor_count)
    actress_indices = np.arange(actor_count, total)
    
    # Save indices
    actor_indices_path = os.path.join(output_dir, "actor_indices.npy")
    actress_indices_path = os.path.join(output_dir, "actress_indices.npy")
    
    np.save(actor_indices_path, actor_indices)
    np.save(actress_indices_path, actress_indices)
    
    print(f"✓ Actor indices saved: {actor_indices_path} ({len(actor_indices)} samples)")
    print(f"✓ Actress indices saved: {actress_indices_path} ({len(actress_indices)} samples)")
    
    return actor_indices_path, actress_indices_path


def log_bias_to_database(bias_report: dict, db_session=None):
    """
    Log bias metrics to predictive_logs table for audit trail.
    
    Args:
        bias_report: Bias report dictionary
        db_session: Optional database session (creates new if not provided)
    """
    close_session = False
    if db_session is None:
        db_session = SessionLocal()
        close_session = True
    
    try:
        # Extract key metrics
        fairness_metrics = bias_report["fairness_metrics"]["accuracy_gap"]
        
        # Create log entry
        log_entry = PredictiveLog(
            model_name=bias_report["model_name"],
            prediction_type="bias_audit",
            prediction_data=json.dumps(bias_report),
            timestamp=datetime.now(timezone.utc),
            bias_score=bias_report["overall_bias_score"],
            is_flagged=bias_report["is_flagged"],
            auditor_id=None,  # Can be set if running with user context
            audit_status="pending" if bias_report["is_flagged"] else "approved",
            audit_notes=bias_report["recommendation"]
        )
        
        db_session.add(log_entry)
        db_session.commit()
        
        print(f"✓ Bias report logged to database (ID: {log_entry.id})")
        return log_entry.id
    
    except Exception as e:
        print(f"✗ Failed to log bias report to database: {e}")
        db_session.rollback()
        return None
    
    finally:
        if close_session:
            db_session.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated Bias Evaluation for Recognition Models")
    parser.add_argument("--data-dir", type=str, default="./Data", help="Root data directory")
    parser.add_argument("--embeddings", type=str, default="./logs/recognition/test_embeddings.npy", 
                       help="Path to test embeddings")
    parser.add_argument("--labels", type=str, default="./logs/recognition/test_labels.npy", 
                       help="Path to test labels")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model (.pth)")
    parser.add_argument("--test-images", type=str, default=None, help="Path to test images directory")
    parser.add_argument("--output-dir", type=str, default="./logs/bias", help="Output directory for reports")
    parser.add_argument("--threshold", type=float, default=0.05, help="Bias threshold (default 5%)")
    parser.add_argument("--skip-database", action="store_true", help="Skip logging to database")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BIAS MONITORING - Automated Fairness Evaluation")
    print("=" * 60)
    
    # Step 1: Create gender split indices if they don't exist
    print("\n[1/4] Creating gender split indices...")
    actor_indices_path = os.path.join(args.output_dir, "actor_indices.npy")
    actress_indices_path = os.path.join(args.output_dir, "actress_indices.npy")
    
    if not os.path.exists(actor_indices_path) or not os.path.exists(actress_indices_path):
        actor_indices_path, actress_indices_path = create_gender_split_indices(
            args.data_dir, args.output_dir
        )
    else:
        print(f"✓ Using existing indices: {actor_indices_path}, {actress_indices_path}")
    
    # Step 2: Check if embeddings exist
    print("\n[2/4] Checking for evaluation embeddings...")
    if not os.path.exists(args.embeddings) or not os.path.exists(args.labels):
        print(f"✗ Embeddings not found. Please run recognition evaluation first:")
        print(f"   python -m evaluation.evaluate_recognition --embeddings-path {args.embeddings} --labels-path {args.labels}")
        return
    
    print(f"✓ Found embeddings: {args.embeddings}")
    print(f"✓ Found labels: {args.labels}")
    
    # Step 3: Compute bias metrics
    print("\n[3/4] Computing bias metrics...")
    output_path = os.path.join(args.output_dir, "bias_report.json")
    
    # Try to find model
    model_path = args.model if args.model else "logs/recognition/recognition_model_best.pth"
    if not os.path.exists(model_path):
        model_path = None
        print(f"⚠️  Model not found at {model_path}, using nearest neighbor fallback")
    
    # Try to find test images
    test_images_dir = args.test_images if args.test_images else None
    
    bias_report = evaluate_facescrub_gender_bias(
        embeddings_path=args.embeddings,
        labels_path=args.labels,
        actor_indices_path=actor_indices_path,
        actress_indices_path=actress_indices_path,
        output_path=output_path,
        model_path=model_path,
        test_images_dir=test_images_dir
    )
    
    # Step 4: Log to database
    print("\n[4/4] Logging to database...")
    if not args.skip_database:
        log_bias_to_database(bias_report)
    else:
        print("⊗ Database logging skipped (--skip-database flag)")
    
    # Print summary
    print("\n" + "=" * 60)
    print("BIAS REPORT SUMMARY")
    print("=" * 60)
    print(f"Model: {bias_report['model_name']}")
    print(f"Total Samples: {bias_report['total_samples']}")
    print(f"Demographic Groups: {bias_report['num_groups']}")
    print(f"\nBias Score: {bias_report['overall_bias_score']:.4f}")
    print(f"Threshold: {bias_report['threshold']:.4f}")
    print(f"Flagged: {'⚠️  YES' if bias_report['is_flagged'] else '✓ NO'}")
    
    print("\nGroup Performance:")
    for group_name, metrics in bias_report['fairness_metrics']['accuracy_gap']['group_metrics'].items():
        print(f"  {group_name}:")
        print(f"    Accuracy: {metrics['accuracy']:.4f}")
        print(f"    F1 Score: {metrics['f1']:.4f}")
        print(f"    Samples: {metrics['sample_count']}")
    
    print(f"\n📊 Recommendation:")
    print(f"  {bias_report['recommendation']}")
    
    print(f"\n✓ Full report saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
