import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from pathlib import Path
import numpy as np

from .utils.dataset_loader import get_landmark_dataset
from .utils.training_utils import save_checkpoint, log_metrics
from .utils.evaluation_metrics import (
    MetricsTracker,
    evaluate_regression,
    visualize_landmarks
)
from .utils.feature_selection import (
    FeatureSelector,
    extract_features_from_model,
    save_feature_selector
)
from .utils.universal_pipeline import (
    WarmupCosineScheduler,
    ModelExporter,
    UniversalTransforms,
    DatasetSplitter,
    InferenceWrapper,
    calculate_nme,
    plot_ced_curve
)

# Placeholder for a landmark detection model
class LandmarkNet(nn.Module):
    def __init__(self, num_landmarks=4000, feature_dim=None):
        super(LandmarkNet, self).__init__()
        # In a real scenario, this would be a proper CNN, e.g., a ResNet variant
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((56, 56))  # Reduce from 224x224 to 56x56
        self.fc1 = nn.Linear(16 * 56 * 56, num_landmarks * 2)  # 50176 instead of 802816

        # Feature selection layer (optional)
        if feature_dim is not None and feature_dim < num_landmarks * 2:
            self.feature_selector_layer = nn.Linear(num_landmarks * 2, feature_dim)
            self.output_layer = nn.Linear(feature_dim, num_landmarks * 2)
        else:
            self.feature_selector_layer = None
            self.output_layer = None

    def forward(self, x):
        # This is a highly simplified forward pass
        x = torch.relu(self.conv1(x))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        # Apply feature selection if configured
        if self.feature_selector_layer is not None:
            features = self.feature_selector_layer(x)
            x = self.output_layer(features)

        return x.view(x.size(0), -1, 2)

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. DataLoader (AFLW2000 landmark dataset)
    # AFLW2000 does not have train/val/test splits, so create our own
    full_dataset = get_landmark_dataset(data_dir=args.data_path, split=None)
    
    # Split into train/val/test (70/15/15) using universal pipeline
    train_dataset, val_dataset, test_dataset = DatasetSplitter.split_dataset(
        full_dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42
    )
    
    # Apply standardized transforms
    train_transform = UniversalTransforms.get_train_transforms('landmarks')
    val_transform = UniversalTransforms.get_val_transforms()
    
    # Apply transforms to datasets
    if hasattr(train_dataset, 'dataset'):
        # For Subset datasets, apply transforms
        original_dataset = train_dataset.dataset
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = val_transform
        test_dataset.dataset.transform = val_transform
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

    # Infer number of landmarks from first sample
    sample_img, sample_landmarks = train_dataset[0]
    num_landmarks = sample_landmarks.shape[0]
    print(f"Detected {num_landmarks} landmarks per face.")

    # === FEATURE SELECTION SETUP ===
    feature_selector = None
    feature_dim = None

    if args.enable_feature_selection:
        print("\n" + "="*60)
        print("FEATURE SELECTION SETUP")
        print("="*60)
        print("Analyzing landmark data for feature selection...")

        # Take a subset of the training set to estimate landmark feature statistics
        import numpy as np
        subset_size = min(500, len(train_dataset))
        subset_indices = torch.randperm(len(train_dataset))[:subset_size]

        landmark_features_list = []
        for idx in subset_indices:
            _, landmarks = train_dataset[idx]  # (num_landmarks, 2)
            landmarks = landmarks.reshape(-1).cpu().numpy()  # flatten to 1D
            landmark_features_list.append(landmarks)

        landmark_features = np.stack(landmark_features_list, axis=0)
        orig_dim = landmark_features.shape[1]
        print(f"Detected {orig_dim // 2} landmarks per face "
              f"({orig_dim} scalar landmark features).")

        # Decide how many PCA features to actually use
        # - If user did not specify n_features → pick something reasonable
        # - If they did, clamp it to orig_dim to avoid PCA errors
        if args.n_features is None:
            n_features_effective = min(64, orig_dim)
        else:
            n_features_effective = min(args.n_features, orig_dim)

        print(f"Applying {args.feature_method} feature selection "
              f"(n_features: {n_features_effective})")

        feature_selector = FeatureSelector(
            method=args.feature_method,
            n_features=n_features_effective,
            task_type='regression'
        )

        # Fit selector and get transformed features (for logging / plots)
        selected_features = feature_selector.fit(landmark_features)
        feature_dim = selected_features.shape[1]

        # Save selector to disk
        selector_path = Path(args.log_dir) / "feature_selector.pkl"
        save_feature_selector(feature_selector, selector_path)

        # Optional: plots (if implemented in your FeatureSelector)
        if hasattr(feature_selector, "explained_variance") and \
           getattr(feature_selector, "explained_variance", None) is not None:
            feature_selector.plot_explained_variance(
                save_path=Path(args.log_dir) / "pca_explained_variance.png"
            )
        elif hasattr(feature_selector, "feature_importance") and \
             getattr(feature_selector, "feature_importance", None) is not None:
            feature_selector.plot_feature_importance(
                save_path=Path(args.log_dir) / "feature_importance.png"
            )

        print(f"✓ Landmark feature selection configured. "
              f"Original dim: {orig_dim}, Selected dim: {feature_dim}")

    model = LandmarkNet(num_landmarks=num_landmarks, feature_dim=feature_dim).to(device)

    # 3. Loss and Optimizer
    criterion = nn.MSELoss() # Mean Squared Error is a common choice for regression tasks
    
    # AdamW optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Warmup + Cosine LR scheduler
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
    
    # Initialize metrics tracker
    tracker = MetricsTracker(log_dir=args.log_dir, task_type='landmark')

    # === RESUME FROM CHECKPOINT IF EXISTS ===
    checkpoint_path = f"{args.log_dir}/landmark_model.pth"
    start_epoch = 0
    if Path(checkpoint_path).exists():
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resumed at epoch {start_epoch}")

    # 4. Training Loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for i, (inputs, landmarks) in enumerate(progress_bar):
            inputs, landmarks = inputs.to(device), landmarks.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, landmarks)
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate

            running_loss += loss.item()
            progress_bar.set_postfix({'loss': running_loss / (i + 1), 'lr': scheduler.get_last_lr()[0]})

        train_loss = running_loss / len(train_loader)
        
        # Evaluate on validation set
        print(f"\n[Epoch {epoch+1}] Evaluating on validation set...")
        val_metrics = evaluate_regression(model, val_loader, device)
        
        # Update metrics tracker
        epoch_metrics = {
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'val_mae': val_metrics['mae'],
            'val_rmse': val_metrics['rmse']
        }
        
        improved = tracker.update(epoch, epoch_metrics)
        
        # Log metrics
        log_str = f"[Epoch {epoch+1}/{args.epochs}] "
        log_str += f"Train Loss: {train_loss:.4f} | "
        log_str += f"Val Loss: {val_metrics['loss']:.4f} | "
        log_str += f"Val MAE: {val_metrics['mae']:.4f} | "
        log_str += f"Val RMSE: {val_metrics['rmse']:.4f}"
        print(log_str)
        
        log_metrics(epoch, epoch_metrics)
        
        # Save checkpoint
        checkpoint_path = f"{args.log_dir}/landmark_model.pth"
        save_checkpoint(model, optimizer, epoch, checkpoint_path)
        
        # Save best model
        if improved:
            best_path = f"{args.log_dir}/landmark_model_best.pth"
            save_checkpoint(model, optimizer, epoch, best_path)
            print(f"✓ New best model saved (val_loss: {val_metrics['loss']:.4f})")
        
        # Early stopping
        if tracker.should_stop(patience=args.patience):
            print(f"\n⚠ Early stopping triggered after {epoch+1} epochs (no improvement for {args.patience} epochs)")
            break
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    # Load best model
    best_path = f"{args.log_dir}/landmark_model_best.pth"
    if Path(best_path).exists():
        print(f"Loading best model from {best_path}")
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    final_metrics = evaluate_regression(model, val_loader, device)
    
    print(f"\nFinal Validation Metrics:")
    print(f"  Loss (MSE): {final_metrics['loss']:.4f}")
    print(f"  MAE:        {final_metrics['mae']:.4f}")
    print(f"  RMSE:       {final_metrics['rmse']:.4f}")
    
    # Calculate NME (Normalized Mean Error)
    if args.enable_landmark_analysis:
        print("\nCalculating NME and CED curve...")
        nme_errors = []
        model.eval()
        with torch.no_grad():
            for inputs, gt_landmarks in val_loader:
                inputs = inputs.to(device)
                pred_landmarks = model(inputs).cpu().numpy()
                gt_landmarks = gt_landmarks.cpu().numpy()
                
                for pred, gt in zip(pred_landmarks, gt_landmarks):
                    nme = calculate_nme(pred.reshape(-1, 2), gt.reshape(-1, 2))
                    nme_errors.append(nme)
        
        mean_nme = np.mean(nme_errors)
        std_nme = np.std(nme_errors)
        failure_rate = np.mean(np.array(nme_errors) > 0.08)
        
        print(f"  NME:        {mean_nme:.4f}")
        print(f"  NME Std:    {std_nme:.4f}")
        print(f"  Failure Rate (>0.08): {failure_rate:.4f}")
        
        # Plot CED curve
        plot_ced_curve(nme_errors, save_path=Path(args.log_dir) / 'ced_curve.png')
        
        # Qualitative landmark visualization
        print("\nGenerating landmark visualizations...")
        # Get a few samples from validation set
        val_iter = iter(val_loader)
        images, gt_landmarks_batch = next(val_iter)
        images = images.to(device)
        
        model.eval()
        with torch.no_grad():
            pred_landmarks_batch = model(images).cpu().numpy()
            gt_landmarks_batch = gt_landmarks_batch.cpu().numpy()
        
        # Reshape to (batch, num_landmarks, 2)
        pred_landmarks_batch = pred_landmarks_batch.reshape(pred_landmarks_batch.shape[0], -1, 2)
        gt_landmarks_batch = gt_landmarks_batch.reshape(gt_landmarks_batch.shape[0], -1, 2)
        
        visualize_landmarks(
            images.cpu(), pred_landmarks_batch, gt_landmarks_batch,
            save_path=Path(args.log_dir) / 'landmark_examples.png',
            title="Predicted vs Ground Truth Landmarks"
        )
    else:
        print("\nSkipping detailed landmark analysis (use --enable-landmark-analysis to enable)")
    
    # Plot training curves
    tracker.plot_metrics()
    tracker.save_summary()
    
    # =========================
    # TEST SET EVALUATION
    # =========================
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    
    test_metrics = evaluate_regression(model, test_loader, device)
    print(f"\nTest Set Metrics:")
    print(f"  Loss (MSE): {test_metrics['loss']:.4f}")
    print(f"  MAE:        {test_metrics['mae']:.4f}")
    print(f"  RMSE:       {test_metrics['rmse']:.4f}")
    
    # Calculate NME on test set
    test_nme_errors = []
    with torch.no_grad():
        for inputs, gt_landmarks in test_loader:
            inputs = inputs.to(device)
            pred_landmarks = model(inputs).cpu().numpy()
            gt_landmarks = gt_landmarks.cpu().numpy()
            
            for pred, gt in zip(pred_landmarks, gt_landmarks):
                nme = calculate_nme(pred.reshape(-1, 2), gt.reshape(-1, 2))
                test_nme_errors.append(nme)
    
    test_mean_nme = np.mean(test_nme_errors)
    print(f"  NME:        {test_mean_nme:.4f}")
    
    # Save training summary JSON
    summary_data = {
        'model_type': 'landmark',
        'hyperparameters': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'patience': args.patience,
            'enable_feature_selection': args.enable_feature_selection,
            'n_features': args.n_features,
            'feature_method': args.feature_method
        },
        'dataset_info': {
            'data_path': args.data_path,
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'test_samples': len(test_dataset)
        },
        'best_epoch': tracker.best_epoch,
        'final_metrics': {
            'loss': final_metrics['loss'],
            'mae': final_metrics['mae'],
            'rmse': final_metrics['rmse']
        },
        'nme_metrics': {
            'val_nme': mean_nme if 'mean_nme' in locals() else None,
            'val_nme_std': std_nme if 'std_nme' in locals() else None,
            'val_failure_rate': failure_rate if 'failure_rate' in locals() else None,
            'test_nme': test_mean_nme
        } if args.enable_landmark_analysis else None,
        'system_info': {
            'device': str(device),
            'cpu_count': torch.get_num_threads(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }
    }
    
    import json
    summary_path = Path(args.log_dir) / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    print(f"✓ Training summary saved to {summary_path}")
    
    # After training completes, register the model in the registry
    print("\nTraining completed. Registering model in registry...")
    try:
        from backend.database.dependencies import SessionLocal
        from backend.services.model_registry import register_model_version
        
        db = SessionLocal()
        try:
            # Determine next version number
            from backend.services.model_registry import list_model_versions
            existing_versions = list_model_versions(db, "landmarks")
            next_version = max([v.version for v in existing_versions], default=0) + 1
            
            # Register this training run
            model_version = register_model_version(
                db=db,
                name="landmarks",
                version=next_version,
                model_path=checkpoint_path,
                training_output_path=args.log_dir,
                set_active=True
            )
            print(f"✓ Model registered: landmarks v{next_version} (ID: {model_version.id})")
        finally:
            db.close()
    except Exception as e:
        print(f"⚠ Warning: Could not register model in database: {e}")
        print("  Model file saved but not registered in model registry.")

    # Export model to TorchScript and ONNX
    print("\n" + "="*60)
    print("EXPORTING MODEL")
    print("="*60)
    
    # Create sample input for export
    sample_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Export to TorchScript
    torchscript_path = Path(args.log_dir) / 'landmark_model_torchscript.pt'
    ModelExporter.export_torchscript(model, sample_input, torchscript_path)
    
    # Export to ONNX
    onnx_path = Path(args.log_dir) / 'landmark_model.onnx'
    ModelExporter.export_onnx(model, sample_input, onnx_path)
    
    # Create inference wrapper
    inference_wrapper = InferenceWrapper(model, device, 'landmarks')
    wrapper_info = inference_wrapper.get_model_info()
    print(f"✓ Inference wrapper created: {wrapper_info}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Landmark Detection Model (AFLW2000)")
    parser.add_argument("--data-path", type=str, default="Data/AFLW2000", help="Path to AFLW2000 data folder (contains .jpg/.mat pairs). Default: Data/AFLW2000")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for regularization.")
    parser.add_argument("--log-dir", type=str, default="./logs/landmarks", help="Directory for logs and checkpoints.")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience (epochs without improvement).")
    parser.add_argument("--enable-feature-selection", action="store_true", help="Enable feature selection for landmark coordinates")
    parser.add_argument("--n-features", type=int, default=None, help="Number of landmark features to select (default: auto)")
    parser.add_argument("--feature-method", type=str, default="pca", choices=["pca", "mutual_info"], help="Feature selection method for landmarks")
    parser.add_argument("--enable-landmark-analysis", action="store_true", help="Enable detailed landmark analysis (NME/CED/visualization)")
    args = parser.parse_args()
    print("\n[INFO] To use the AFLW2000 dataset, make sure you have .jpg and .mat pairs in the specified folder.\n")
    main(args)
