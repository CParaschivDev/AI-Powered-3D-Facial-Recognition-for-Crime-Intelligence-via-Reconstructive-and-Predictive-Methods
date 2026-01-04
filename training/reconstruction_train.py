import torch
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import numpy as np

from .utils.dataset_loader import get_reconstruction_dataset
from .utils.training_utils import save_checkpoint, log_metrics
from .utils.losses import IdentityPreservingLoss
from .utils.evaluation_metrics import MetricsTracker, evaluate_regression, vertex_error, landmark_3d_error
from .utils.feature_selection import (
    FeatureSelector,
    extract_features_from_model,
    save_feature_selector
)
from .utils.head_renderer import HeadTemplateRenderer
from .utils.universal_pipeline import (
    WarmupCosineScheduler,
    ModelExporter,
    UniversalTransforms,
    DatasetSplitter,
    InferenceWrapper
)

# Placeholder for a landmark detection model
class LandmarkNet(nn.Module):
    def __init__(self, num_landmarks, feature_dim=None):
        super(LandmarkNet, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((112, 112))
        self.landmark_predictor = nn.Linear(64 * 112 * 112, num_landmarks * 2)  # x,y coordinates for each landmark

        # Feature selection layer (optional)
        if feature_dim is not None and feature_dim < num_landmarks * 2:
            self.feature_selector_layer = nn.Linear(num_landmarks * 2, feature_dim)
            self.output_layer = nn.Linear(feature_dim, num_landmarks * 2)
        else:
            self.feature_selector_layer = None
            self.output_layer = None

    def forward(self, x):
        features = self.backbone(x)
        features = self.adaptive_pool(features)
        features = features.view(features.size(0), -1)
        landmarks = self.landmark_predictor(features)

        # Apply feature selection if configured
        if self.feature_selector_layer is not None:
            selected_landmarks = self.feature_selector_layer(landmarks)
            landmarks = self.output_layer(selected_landmarks)

        return landmarks

# Placeholder for a reconstruction model (e.g., a network that predicts 3DMM parameters)
class ReconstructionNet(nn.Module):
    def __init__(self, num_params, feature_dim=None):
        super(ReconstructionNet, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((112, 112))
        self.param_predictor = nn.Linear(64 * 112 * 112, num_params)

        # Feature selection layer (optional)
        if feature_dim is not None and feature_dim < num_params:
            self.feature_selector_layer = nn.Linear(num_params, feature_dim)
            self.output_layer = nn.Linear(feature_dim, num_params)
        else:
            self.feature_selector_layer = None
            self.output_layer = None

    def forward(self, x):
        features = self.backbone(x)
        features = self.adaptive_pool(features)  # Pool to 112x112
        features = features.view(features.size(0), -1)
        params = self.param_predictor(features)

        # Apply feature selection if configured
        if self.feature_selector_layer is not None:
            selected_params = self.feature_selector_layer(params)
            params = self.output_layer(selected_params)

        return params

def create_renderer(device: str, use_real_renderer: bool = True,
                   mesh_path: str = "Data/data/head_template_mesh.obj",
                   landmark_path: str = "Data/data/landmark_embedding.npy"):
    """
    Create a renderer for 3D face reconstruction.

    Args:
        device: Device to run renderer on
        use_real_renderer: Whether to use real 3D mesh renderer or dummy
        mesh_path: Path to template mesh
        landmark_path: Path to landmark embedding

    Returns:
        Renderer function or object
    """
    if use_real_renderer:
        try:
            renderer = HeadTemplateRenderer(
                mesh_path=mesh_path,
                landmark_path=landmark_path,
                device=device
            )
            print("✓ Using real 3D head mesh renderer for identity preservation")
            return renderer
        except Exception as e:
            print(f"⚠ Failed to load real renderer: {e}")
            print("Falling back to dummy renderer")
            return None
    else:
        print("Using dummy renderer (returns input image)")
        return None

def dummy_renderer(params, input_image):
    """
    Dummy renderer: For now, just return the input image.
    In practice, this would render a face from 3DMM parameters.
    """
    # Placeholder: return input image as rendered face
    return input_image

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. DataLoaders
    # This requires a dataset with images and corresponding ground truth 3DMM parameters
    full_dataset = get_reconstruction_dataset(data_dir=args.data_path, split=None)
    
    # Split into train/val/test (70/15/15) using universal pipeline
    train_dataset, val_dataset, test_dataset = DatasetSplitter.split_dataset(
        full_dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42
    )
    
    # Apply standardized transforms
    train_transform = UniversalTransforms.get_train_transforms('reconstruction')
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

    # 2. Model
    # Infer number of 3DMM parameters from first sample
    sample_img, sample_params = train_dataset[0]
    num_params = sample_params.shape[0]
    print(f"Detected {num_params} reconstruction parameters per face.")
    
    # Log parameter statistics for debugging
    print("Sample params stats:",
          "min", float(sample_params.min()),
          "max", float(sample_params.max()),
          "mean", float(sample_params.mean()),
          "std", float(sample_params.std()))

    # === FEATURE SELECTION SETUP ===
    feature_selector = None
    feature_dim = None
    if args.enable_feature_selection:
        print("\n" + "="*60)
        print("FEATURE SELECTION SETUP")
        print("="*60)

        # Create feature selection pipeline for 3DMM parameters
        feature_selector = FeatureSelector(
            method=args.feature_method,
            n_features=args.n_features,
            task_type='regression'
        )

        # Extract 3DMM parameter data for feature selection
        print("Analyzing 3DMM parameters for feature selection...")
        subset_size = min(500, len(train_dataset))
        subset_indices = torch.randperm(len(train_dataset))[:subset_size]
        subset_params = []

        for idx in subset_indices:
            _, params = train_dataset[idx]
            subset_params.append(params.numpy())

        param_features = np.array(subset_params)

        # Fit feature selector on 3DMM parameters
        selected_features = feature_selector.fit(param_features)

        # Save feature selector
        selector_path = f"{args.log_dir}/feature_selector.pkl"
        save_feature_selector(feature_selector, selector_path)

        # Plot feature analysis
        if hasattr(feature_selector, 'explained_variance') and feature_selector.explained_variance is not None:
            feature_selector.plot_explained_variance(
                save_path=Path(args.log_dir) / '3dmm_pca_variance.png'
            )
        elif hasattr(feature_selector, 'feature_importance') and feature_selector.feature_importance is not None:
            feature_selector.plot_feature_importance(
                save_path=Path(args.log_dir) / '3dmm_feature_importance.png'
            )

        feature_dim = selected_features.shape[1]
        print(f"✓ 3DMM feature selection configured. Original: {param_features.shape[1]}, Selected: {feature_dim}")

    model = ReconstructionNet(num_params=num_params, feature_dim=feature_dim).to(device)

    # 3. Loss and Optimizer
    # Use MSE loss for 3DMM parameter prediction
    param_loss_fn = nn.MSELoss()

    # Create renderer
    renderer = create_renderer(
        device=device,
        use_real_renderer=args.use_real_renderer,
        mesh_path=args.mesh_path,
        landmark_path=args.landmark_path
    )

    # Initialize 3D vertex buffers for later metrics / viz
    gt_vertices_list = []
    pred_vertices_list = []

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Warmup + Cosine LR scheduler
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
    
    # Initialize metrics tracker
    tracker = MetricsTracker(log_dir=args.log_dir, task_type='regression')

    # === RESUME FROM CHECKPOINT IF EXISTS ===
    checkpoint_path = f"{args.log_dir}/reconstruction_model.pth"
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

        for i, (inputs, gt_params) in enumerate(progress_bar):
            inputs, gt_params = inputs.to(device), gt_params.to(device)

            optimizer.zero_grad()
            pred_params = model(inputs)
            
            # 3DMM parameter loss
            loss = param_loss_fn(pred_params, gt_params)
            
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate

            running_loss += loss.item()
            progress_bar.set_postfix({'loss': running_loss / (i + 1)})

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
        checkpoint_path = f"{args.log_dir}/reconstruction_model.pth"
        save_checkpoint(model, optimizer, epoch, checkpoint_path)
        
        # Save best model
        if improved:
            best_path = f"{args.log_dir}/reconstruction_model_best.pth"
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
    best_path = f"{args.log_dir}/reconstruction_model_best.pth"
    if Path(best_path).exists():
        print(f"Loading best model from {best_path}")
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    final_metrics = evaluate_regression(model, val_loader, device)
    
    print(f"\nFinal Validation Metrics:")
    print(f"  Loss (MSE): {final_metrics['loss']:.4f}")
    print(f"  MAE:        {final_metrics['mae']:.4f}")
    print(f"  RMSE:       {final_metrics['rmse']:.4f}")
    
    # Compute reconstruction parameter errors
    print("\nComputing reconstruction parameter errors...")
    
    # Collect predictions and targets for parameter error calculation
    model.eval()
    all_pred_params = []
    all_gt_params = []
    
    with torch.no_grad():
        for inputs, gt_params in val_loader:
            inputs = inputs.to(device)
            pred_params = model(inputs)
            all_pred_params.extend(pred_params.cpu().numpy())
            all_gt_params.extend(gt_params.cpu().numpy())
    
    all_pred_params = np.array(all_pred_params)
    all_gt_params = np.array(all_gt_params)
    
    # Calculate parameter errors
    param_errors = np.mean((all_pred_params - all_gt_params) ** 2, axis=1)
    mean_param_error = np.mean(param_errors)
    std_param_error = np.std(param_errors)
    max_param_error = np.max(param_errors)
    
    print(f"  Mean Parameter Error: {mean_param_error:.4f}")
    print(f"  Std Parameter Error:  {std_param_error:.4f}")
    print(f"  Max Parameter Error:  {max_param_error:.4f}")
    
    # Plot training curves
    tracker.plot_metrics()
    tracker.save_summary()
    
    # Initialize vertex buffers for visualization
    gt_vertices_list = []
    pred_vertices_list = []
    
    # Reconstruction visualization and analysis
    if args.enable_reconstruction_viz:
        print("\nGenerating reconstruction visualizations...")
        try:
            import matplotlib.pyplot as plt
            
            # Get some test samples
            test_iter = iter(test_loader)
            images, gt_params = next(test_iter)
            images = images[:8].to(device)  # Visualize first 8 samples
            gt_params = gt_params[:8]
            
            model.eval()
            with torch.no_grad():
                pred_params = model(images)
            
            # Create visualization grid
            fig, axes = plt.subplots(4, 8, figsize=(24, 12))
            
            # Render ground truth and predictions if renderer is available
            gt_renders = []
            pred_renders = []
            
            if renderer is not None:
                try:
                    # Render ground truth faces
                    num_samples = min(8, len(images))
                    for j in range(num_samples):
                        # Split parameters back into shape, expression, pose
                        params = gt_params[j].cpu().numpy()
                        shape_params = params[:199]  # First 199 are shape parameters
                        exp_params = params[199:199+29]  # Next 29 are expression parameters
                        # Remaining 7 are pose parameters (not used for current rendering)
                        
                        gt_vertices = renderer.params_to_vertices(
                            torch.tensor(shape_params, device=device).unsqueeze(0),
                            torch.tensor(exp_params, device=device).unsqueeze(0)
                        ).squeeze(0)
                        gt_render = renderer.render(gt_vertices.unsqueeze(0)).squeeze(0)
                        
                        # Just convert to numpy and store; no shape policing
                        gt_renders.append(gt_render.detach().cpu().numpy())
                        gt_vertices_list.append(gt_vertices)
                    
                    # Render predicted faces
                    for j in range(num_samples):
                        # Split parameters back into shape, expression, pose
                        params = pred_params[j].cpu().numpy()
                        shape_params = params[:199]  # First 199 are shape parameters
                        exp_params = params[199:199+29]  # Next 29 are expression parameters
                        # Remaining 7 are pose parameters (not used for current rendering)
                        
                        pred_vertices = renderer.params_to_vertices(
                            torch.tensor(shape_params, device=device).unsqueeze(0),
                            torch.tensor(exp_params, device=device).unsqueeze(0)
                        ).squeeze(0)
                        pred_render = renderer.render(pred_vertices.unsqueeze(0)).squeeze(0)
                        
                        # Just convert to numpy and store; no shape policing
                        pred_renders.append(pred_render.detach().cpu().numpy())
                        pred_vertices_list.append(pred_vertices)
                    
                    print("✓ Rendered 3D faces for visualization")
                except Exception as e:
                    print(f"⚠ Rendering failed: {e}, using placeholder images")
                    renderer = None
            
            for i in range(8):
                # Original image
                if i < len(images):
                    img = images[i].cpu().permute(1, 2, 0).numpy()
                    # Normalize image to [0, 1] for matplotlib
                    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                    axes[0, i].imshow(img)
                else:
                    axes[0, i].imshow(np.zeros((224, 224, 3)))
                axes[0, i].set_title('Input')
                axes[0, i].axis('off')
                
                # Ground truth reconstruction
                if renderer is not None and i < len(gt_renders):
                    gt_img = gt_renders[i]
                    # Normalize rendered image
                    if gt_img.max() > 1.0 or gt_img.min() < 0.0:
                        gt_img = (gt_img - gt_img.min()) / (gt_img.max() - gt_img.min() + 1e-8)
                    axes[1, i].imshow(gt_img)
                else:
                    if i < len(images):
                        img = images[i].cpu().permute(1, 2, 0).numpy()
                        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                        axes[1, i].imshow(img)  # Placeholder
                    else:
                        axes[1, i].imshow(np.zeros((224, 224, 3)))
                axes[1, i].set_title('Ground Truth')
                axes[1, i].axis('off')
                
                # Predicted reconstruction
                if renderer is not None and i < len(pred_renders):
                    pred_img = pred_renders[i]
                    # Normalize rendered image
                    if pred_img.max() > 1.0 or pred_img.min() < 0.0:
                        pred_img = (pred_img - pred_img.min()) / (pred_img.max() - pred_img.min() + 1e-8)
                    axes[2, i].imshow(pred_img)
                else:
                    if i < len(images):
                        img = images[i].cpu().permute(1, 2, 0).numpy()
                        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                        axes[2, i].imshow(img)  # Placeholder
                    else:
                        axes[2, i].imshow(np.zeros((224, 224, 3)))
                axes[2, i].set_title('Prediction')
                axes[2, i].axis('off')
                
                # Error visualization (difference)
                if renderer is not None and i < len(gt_renders) and i < len(pred_renders):
                    # Compute difference between rendered images
                    gt_img = gt_renders[i]
                    pred_img = pred_renders[i]
                    error_img = np.abs(gt_img.astype(np.float32) - pred_img.astype(np.float32))
                    if error_img.ndim == 3:
                        error_img = np.mean(error_img, axis=2)  # Convert to grayscale error
                    elif error_img.ndim == 2 and error_img.shape[1] == 3:
                        # Vertex color error: compute L2 norm per vertex
                        error_img = np.linalg.norm(error_img, axis=1)
                    # Normalize error image
                    error_img = error_img / (error_img.max() + 1e-8)
                    axes[3, i].imshow(error_img.reshape(-1, 1), cmap='hot', vmin=0, vmax=1, aspect='auto')
                else:
                    error_img = np.zeros((224, 224))  # Placeholder
                    axes[3, i].imshow(error_img, cmap='hot')
                axes[3, i].set_title('Error Map')
                axes[3, i].axis('off')
            
            plt.tight_layout()
            plt.savefig(Path(args.log_dir) / 'reconstruction_visualization.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Reconstruction visualization saved to {Path(args.log_dir) / 'reconstruction_visualization.png'}")
            
            # Parameter distribution analysis
            plt.figure(figsize=(15, 10))
            
            # Plot parameter distributions
            n_params_to_plot = min(20, gt_params.shape[1])  # Use actual parameter count
            for i in range(n_params_to_plot):
                plt.subplot(4, 5, i+1)
                plt.hist(gt_params[:, i].cpu().numpy(), alpha=0.7, label='Ground Truth', bins=20)
                plt.hist(pred_params[:, i].cpu().numpy(), alpha=0.7, label='Prediction', bins=20)
                plt.title(f'Param {i+1}')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(Path(args.log_dir) / 'parameter_distributions.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Parameter distribution analysis saved to {Path(args.log_dir) / 'parameter_distributions.png'}")
            
            # 3D Error analysis plots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Parameter error distribution
            param_errors = np.mean((pred_params.cpu().numpy() - gt_params.cpu().numpy()) ** 2, axis=1)
            axes[0, 0].hist(param_errors, bins=30, alpha=0.7)
            axes[0, 0].set_xlabel('Parameter MSE')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Parameter Error Distribution')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Landmark error distribution
            # Note: For reconstruction, we analyze parameter errors instead of 2D landmark errors
            axes[0, 1].text(0.5, 0.5, 'Parameter-based analysis\n(no 2D landmarks)', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Parameter Error Analysis')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Cumulative error distribution
            axes[1, 0].hist(param_errors, bins=50, cumulative=True, density=True, alpha=0.7, label='Parameter MSE')
            axes[1, 0].set_xlabel('Parameter MSE')
            axes[1, 0].set_ylabel('Cumulative Probability')
            axes[1, 0].set_title('Cumulative Parameter Error Distribution')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Error statistics summary
            axes[1, 1].axis('off')
            mean_err = float(param_errors.mean())
            std_err = float(param_errors.std())
            max_err = float(param_errors.max())
            p90 = float(np.percentile(param_errors, 90))
            p95 = float(np.percentile(param_errors, 95))
            p99 = float(np.percentile(param_errors, 99))
            stats_text = f"Parameter Error Stats\n  mean = {mean_err:.6f}\n  std = {std_err:.6f}\n  max = {max_err:.6f}\n  p90 = {p90:.6f}\n  p95 = {p95:.6f}\n  p99 = {p99:.6f}"
            axes[1, 1].text(0.1, 0.8, stats_text, fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            
            plt.tight_layout()
            plt.savefig(Path(args.log_dir) / '3d_error_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Parameter error analysis saved to {Path(args.log_dir) / '3d_error_analysis.png'}")
        
        except Exception as e:
            print(f"⚠ Reconstruction visualization failed: {e}")
    
    elif args.enable_reconstruction_viz:
        print("⚠ Reconstruction visualization skipped: matplotlib not available")
    
    # Compute 3D reconstruction errors if renderer is available
    vertex_rmse = None
    lm3d_rmse = None
    if renderer is not None and 'gt_vertices_list' in locals() and len(gt_vertices_list) > 0:
        print("\nComputing 3D reconstruction errors...")
        try:
            # Calculate vertex errors
            vertex_errors = []
            landmark_3d_errors = []
            
            for i in range(len(gt_vertices_list)):
                # Vertex error (RMSE between predicted and GT vertices)
                v_error_dict = vertex_error(pred_vertices_list[i].cpu().numpy(), gt_vertices_list[i].cpu().numpy())
                vertex_errors.append(v_error_dict['mean_vertex_error'])
                
                # 3D landmark error
                lm3d_error_dict = landmark_3d_error(pred_vertices_list[i].cpu().numpy(), gt_vertices_list[i].cpu().numpy())
                landmark_3d_errors.append(lm3d_error_dict['mean_landmark_error'])
            
            vertex_rmse = np.mean(vertex_errors)
            lm3d_rmse = np.mean(landmark_3d_errors)
            
            print(f"  Mean Vertex RMSE: {vertex_rmse:.6f}")
            print(f"  Mean 3D Landmark RMSE: {lm3d_rmse:.6f}")
            
        except Exception as e:
            print(f"⚠ 3D error calculation failed: {e}")
    
    # Save training summary JSON
    summary_data = {
        'model_type': 'reconstruction',
        'hyperparameters': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'patience': args.patience,
            'enable_feature_selection': args.enable_feature_selection,
            'n_features': args.n_features,
            'feature_method': args.feature_method,
            'enable_reconstruction_viz': args.enable_reconstruction_viz
        },
        'dataset_info': {
            'data_path': args.data_path,
            'num_params': num_params,
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
        'param_metrics': {
            'mean_param_error': mean_param_error,
            'std_param_error': std_param_error,
            'max_param_error': max_param_error
        },
        '3d_metrics': {
            'vertex_rmse': vertex_rmse,
            'landmark_3d_rmse': lm3d_rmse
        } if vertex_rmse is not None else {},
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
    
    # Export model for deployment
    print("\n" + "="*60)
    print("MODEL EXPORT")
    print("="*60)
    
    # Export to TorchScript
    torchscript_path = f"{args.log_dir}/reconstruction_model_torchscript.pt"
    ModelExporter.export_torchscript(model, sample_input=torch.randn(1, 3, 224, 224).to(device), save_path=torchscript_path)
    
    # Export to ONNX
    onnx_path = f"{args.log_dir}/reconstruction_model.onnx"
    ModelExporter.export_onnx(model, sample_input=torch.randn(1, 3, 224, 224).to(device), save_path=onnx_path)
    
    # After training completes, register the model in the registry
    print("\nTraining completed. Registering model in registry...")
    try:
        from backend.database.dependencies import SessionLocal
        from backend.services.model_registry import register_model_version, list_model_versions

        db = SessionLocal()
        try:
            # Determine next version number
            existing_versions = list_model_versions(db, "reconstruction")
            next_version = max([v.version for v in existing_versions], default=0) + 1

            # Register this training run
            model_version = register_model_version(
                db=db,
                name="reconstruction",
                version=next_version,
                model_path=checkpoint_path,
                training_output_path=args.log_dir,
                set_active=True
            )
            print(f"✓ Model registered: reconstruction v{next_version} (ID: {model_version.id})")
        finally:
            db.close()
    except Exception as e:
        print(f"⚠ Model registration failed: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train 3D Face Reconstruction Model')
    parser.add_argument('--data-path', type=str, required=True, help='Path to AFLW2000 dataset directory')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--log-dir', type=str, default='logs/reconstruction', help='Directory to save logs and checkpoints')
    parser.add_argument('--enable-feature-selection', action='store_true', help='Enable feature selection for 3DMM parameters')
    parser.add_argument('--n-features', type=int, default=100, help='Number of features to select')
    parser.add_argument('--feature-method', type=str, default='pca', choices=['pca', 'mutual_info', 'f_regression'], help='Feature selection method')
    parser.add_argument('--enable-reconstruction-viz', action='store_true', help='Enable 3D reconstruction visualization')
    parser.add_argument('--use-real-renderer', action='store_true', help='Use real 3D mesh renderer instead of dummy')
    parser.add_argument('--mesh-path', type=str, default='Data/data/head_template_mesh.obj', help='Path to head template mesh')
    parser.add_argument('--landmark-path', type=str, default='Data/data/landmark_embedding.npy', help='Path to landmark embedding')
    
    args = parser.parse_args()
    main(args)
