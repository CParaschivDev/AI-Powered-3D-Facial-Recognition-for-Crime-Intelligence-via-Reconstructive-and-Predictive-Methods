import torch
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import argparse
from tqdm import tqdm
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import json
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

try:
    import umap
except ImportError:
    umap = None

from .utils.dataset_loader import get_recognition_dataset
from .utils.training_utils import save_checkpoint, log_metrics
from .utils.evaluation_metrics import (
    MetricsTracker,
    evaluate_classification,
    plot_confusion_matrix,
    plot_roc_curve,
    generate_classification_report,
    evaluate_face_verification,
    explain_with_shap,
    explain_with_lime
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
    GradCAM
)
from .utils.losses import TripletLoss, ArcFaceEmbedding
from backend.utils.augmentation import CCTVAugmentation
from backend.database.dependencies import SessionLocal
from backend.services.model_registry import register_model_version, list_model_versions

def create_triplets(embeddings, labels, margin=0.5):
    """
    Create triplets from embeddings and labels for triplet loss training.
    
    Args:
        embeddings: Face embeddings
        labels: Identity labels
        margin: Triplet margin
        
    Returns:
        Triplets as tensors
    """
    anchors, positives, negatives = [], [], []
    
    # Group by identity
    identity_groups = {}
    for i, label in enumerate(labels):
        if label not in identity_groups:
            identity_groups[label] = []
        identity_groups[label].append(i)
    
    for anchor_idx in range(len(embeddings)):
        anchor_label = labels[anchor_idx]
        
        # Sample positive (same identity, different image)
        if len(identity_groups[anchor_label]) > 1:
            positive_candidates = [i for i in identity_groups[anchor_label] if i != anchor_idx]
            positive_idx = np.random.choice(positive_candidates)
        else:
            continue  # Skip if no positive available
            
        # Sample negative (different identity)
        negative_labels = [l for l in identity_groups.keys() if l != anchor_label]
        negative_label = np.random.choice(negative_labels)
        negative_idx = np.random.choice(identity_groups[negative_label])
        
        anchors.append(embeddings[anchor_idx])
        positives.append(embeddings[positive_idx])
        negatives.append(embeddings[negative_idx])
    
    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)


class RecognitionNet(nn.Module):
    def __init__(self, num_classes, embedding_size=512, feature_dim=None, head_type='linear'):
        super(RecognitionNet, self).__init__()
        # Match inference architecture exactly
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Adaptive pooling to handle any input size (224x224 from transforms)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((112, 112))
        # Only embedding layer - matches inference
        self.embedding_layer = nn.Linear(64 * 112 * 112, embedding_size)
        # Feature selection layer (optional)
        if feature_dim is not None and feature_dim < embedding_size:
            self.feature_selector_layer = nn.Linear(embedding_size, feature_dim)
            classifier_input_dim = feature_dim
        else:
            self.feature_selector_layer = None
            classifier_input_dim = embedding_size
        
        self.head_type = head_type
        if head_type == 'linear':
            # Standard classifier
            self.classifier = nn.Linear(classifier_input_dim, num_classes)
        elif head_type in ['arcface', 'cosface']:
            # ArcFace/CosFace head
            self.classifier = ArcFaceEmbedding(embedding_dim=classifier_input_dim, pretrained=False)
            # Override the embedding layer since ArcFace has its own
            self.embedding_layer = nn.Identity()  # ArcFace handles embedding
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    def forward(self, x, labels=None, return_embedding=False):
        x = self.backbone(x)
        x = self.adaptive_pool(x)  # Pool to 112x112
        x = x.view(x.size(0), -1)
        embedding = self.embedding_layer(x)

        # Apply feature selection if configured
        if self.feature_selector_layer is not None:
            embedding = self.feature_selector_layer(embedding)

        if return_embedding:
            return embedding
        
        if self.head_type == 'linear':
            logits = self.classifier(embedding)
            return logits
        elif self.head_type in ['arcface', 'cosface']:
            # For ArcFace/CosFace, pass labels for margin computation
            logits = self.classifier(embedding, labels=labels, return_embedding=False)
            return logits

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. DataLoaders - Support multiple data paths
    data_paths = [p.strip() for p in args.data_path.split(',')]
    print(f"\nLoading data from {len(data_paths)} path(s):")
    for p in data_paths:
        print(f"  - {p}")
    
    # Load datasets from all paths
    datasets = []
    all_classes = set()
    fast_mode = args.fast_mode if hasattr(args, 'fast_mode') else False

    # Check if we have pre-split train/val/test folders
    has_pre_split = False
    train_datasets = []
    val_datasets = []
    test_datasets = []

    for data_path in data_paths:
        train_path = os.path.join(data_path, 'train')
        val_path = os.path.join(data_path, 'val')
        test_path = os.path.join(data_path, 'test')

        if os.path.exists(train_path) and os.path.exists(val_path):
            # Use pre-split datasets
            has_pre_split = True
            print(f"  Using pre-split data from {data_path}")
            print(f"    Train: {train_path}")
            print(f"    Val: {val_path}")

            train_datasets.append(get_recognition_dataset(data_dir=train_path, split=None, fast_mode=fast_mode))
            val_datasets.append(get_recognition_dataset(data_dir=val_path, split=None, fast_mode=fast_mode))

            if os.path.exists(test_path):
                test_datasets.append(get_recognition_dataset(data_dir=test_path, split=None, fast_mode=fast_mode))
                print(f"    Test: {test_path}")
        else:
            # Fall back to single directory or train subdirectory
            if os.path.exists(train_path):
                data_path = train_path
                print(f"  Using train subdirectory: {data_path}")
            dataset = get_recognition_dataset(data_dir=data_path, split=None, fast_mode=fast_mode)
            datasets.append(dataset)
            all_classes.update(dataset.classes)

    if has_pre_split:
        # Collect classes from all datasets before creating ConcatDataset
        for ds in train_datasets:
            all_classes.update(ds.classes)
        for ds in val_datasets:
            all_classes.update(ds.classes)
        for ds in test_datasets:
            all_classes.update(ds.classes)

        # Combine pre-split datasets
        if train_datasets:
            train_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
        if val_datasets:
            val_dataset = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]
        if test_datasets:
            test_dataset = ConcatDataset(test_datasets) if len(test_datasets) > 1 else test_datasets[0]
        else:
            # Create test split from validation data if no test folder
            # Get classes from val dataset before splitting
            if hasattr(val_dataset, 'classes'):
                all_classes.update(val_dataset.classes)

            val_dataset_temp, _, test_dataset = DatasetSplitter.split_dataset(
                val_dataset, train_ratio=0.7, val_ratio=0.0, test_ratio=0.3, seed=42
            )
            val_dataset = val_dataset_temp
            print("No test folder found, created test split from validation data")

        print(f"Using pre-split datasets: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    else:
        # Combine all datasets and split
        if len(datasets) > 1:
            combined_dataset = ConcatDataset(datasets)
            print(f"\nCombined {len(datasets)} datasets into {len(combined_dataset)} total samples")
        else:
            combined_dataset = datasets[0]

        # Split into train/val/test (70/15/15) using universal pipeline
        train_dataset, val_dataset, test_dataset = DatasetSplitter.split_dataset(
            combined_dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42
        )
    
    # Apply standardized transforms
    train_transform = UniversalTransforms.get_train_transforms('recognition')
    val_transform = UniversalTransforms.get_val_transforms()
    
    # Add CCTV augmentations if enabled
    if args.enable_cctv_aug:
        from torchvision import transforms
        cctv_aug = CCTVAugmentation(probability=0.3)  # 30% chance per image
        train_transform = transforms.Compose([
            cctv_aug,
            train_transform
        ])
        print("✓ CCTV augmentations enabled during training")
    
    # Apply transforms to datasets
    if hasattr(train_dataset, 'dataset'):
        # For Subset datasets, apply transforms
        original_dataset = train_dataset.dataset
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = val_transform
        test_dataset.dataset.transform = val_transform
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    num_classes = len(all_classes)
    print(f"\nTraining with {num_classes} unique classes.")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

    # === FEATURE SELECTION SETUP ===
    feature_selector = None
    if args.enable_feature_selection:
        print("\n" + "="*60)
        print("FEATURE SELECTION SETUP")
        print("="*60)

        # Create feature selection pipeline
        feature_selector = FeatureSelector(
            method=args.feature_method,
            n_features=args.n_features,
            task_type='classification'
        )

        # Extract features from a small subset for feature selection
        print("Extracting features for feature selection analysis...")
        subset_size = min(1000, len(train_dataset))  # Use subset for efficiency
        subset_indices = torch.randperm(len(train_dataset))[:subset_size]
        subset_dataset = torch.utils.data.Subset(train_dataset, subset_indices)
        subset_loader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=False)

        # Create a temporary model for feature extraction
        temp_model = RecognitionNet(num_classes=num_classes).to(device)

        # Extract raw features (before any selection)
        raw_features, labels = extract_features_from_model(
            temp_model, subset_loader, device, layer_name='embedding_layer'
        )

        # Fit feature selector
        selected_features = feature_selector.fit(raw_features, labels)

        # Save feature selector
        selector_path = f"{args.log_dir}/feature_selector.pkl"
        save_feature_selector(feature_selector, selector_path)

        # Plot feature analysis
        if hasattr(feature_selector, 'explained_variance') and feature_selector.explained_variance is not None:
            feature_selector.plot_explained_variance(
                save_path=Path(args.log_dir) / 'pca_explained_variance.png'
            )
        elif hasattr(feature_selector, 'feature_importance') and feature_selector.feature_importance is not None:
            feature_selector.plot_feature_importance(
                save_path=Path(args.log_dir) / 'feature_importance.png'
            )

        print(f"✓ Feature selection configured. Original features: {raw_features.shape[1]}, Selected: {selected_features.shape[1]}")

    # 2. Model
    feature_dim = feature_selector.n_features if feature_selector else None
    
    # Determine head type based on loss mode
    if args.loss_mode in ['arcface', 'cosface']:
        head_type = args.loss_mode
    else:
        head_type = 'linear'
    
    model = RecognitionNet(num_classes=num_classes, feature_dim=feature_dim, head_type=head_type).to(device)

    # 3. Loss and Optimizer
    # Support different loss modes for academic comparison
    if args.loss_mode == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif args.loss_mode == 'triplet':
        criterion = TripletLoss(margin=args.triplet_margin)
        # For triplet loss, we need to modify the training loop to sample triplets
        print("✓ Using Triplet Loss for training")
    elif args.loss_mode in ['arcface', 'cosface']:
        criterion = nn.CrossEntropyLoss()  # ArcFace/CosFace compute margin internally
        print(f"✓ Using {args.loss_mode.upper()} loss for training")
    else:
        raise ValueError(f"Unknown loss mode: {args.loss_mode}")
    
    # Adam optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Warmup + Cosine LR scheduler
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
    
    # Initialize metrics tracker
    tracker = MetricsTracker(log_dir=args.log_dir, task_type='classification')
    class_names = sorted(list(all_classes))

    # === RESUME FROM CHECKPOINT IF EXISTS ===
    checkpoint_path = f"{args.log_dir}/recognition_model.pth"
    start_epoch = 0
    if Path(checkpoint_path).exists():
        print(f"Found checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Check if checkpoint is compatible (same number of classes)
        checkpoint_classes = checkpoint['model_state_dict']['classifier.weight'].shape[0]
        if checkpoint_classes == num_classes:
            print(f"Resuming from epoch {checkpoint.get('epoch', 0) + 1} (compatible checkpoint)")
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
        else:
            print(f"⚠ Checkpoint has {checkpoint_classes} classes but current model has {num_classes} classes")
            print(f"⚠ Starting fresh training (checkpoint incompatible)")
            # Optionally backup old checkpoint
            backup_path = f"{args.log_dir}/recognition_model_backup_{checkpoint_classes}classes.pth"
            Path(checkpoint_path).rename(backup_path)
            print(f"✓ Old checkpoint backed up to: {backup_path}")

    # 4. Training Loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for i, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            
            if args.loss_mode == 'cross_entropy':
                logits = model(inputs)
                loss = criterion(logits, labels)
            elif args.loss_mode == 'triplet':
                # Extract embeddings for triplet loss
                embeddings = model(inputs, return_embedding=True)
                # Create triplets from batch
                batch_embeddings = embeddings.detach().cpu().numpy()
                batch_labels = labels.detach().cpu().numpy()
                
                try:
                    anchors, positives, negatives = create_triplets(
                        batch_embeddings, batch_labels, margin=args.triplet_margin
                    )
                    anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
                    loss = criterion(anchors, positives, negatives)
                except:
                    # Fallback to cross-entropy if triplet creation fails
                    logits = model(inputs)
                    loss = nn.CrossEntropyLoss()(logits, labels)
            elif args.loss_mode in ['arcface', 'cosface']:
                logits = model(inputs, labels=labels)
                loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate

            running_loss += loss.item()
            progress_bar.set_postfix({'loss': running_loss / (i + 1), 'lr': scheduler.get_last_lr()[0]})

        train_loss = running_loss / len(train_loader)
        
        # Evaluate on validation set
        print(f"\n[Epoch {epoch+1}] Evaluating on validation set...")
        val_metrics = evaluate_classification(model, val_loader, device, num_classes)
        
        # Update metrics tracker
        epoch_metrics = {
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
            'val_f1': val_metrics['f1'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall']
        }
        
        # Add AUC-ROC if available
        if val_metrics.get('auc_roc') is not None:
            epoch_metrics['val_auc_roc'] = val_metrics['auc_roc']
        
        improved = tracker.update(epoch, epoch_metrics)
        
        # Log metrics
        log_str = f"[Epoch {epoch+1}/{args.epochs}] "
        log_str += f"Train Loss: {train_loss:.4f} | "
        log_str += f"Val Loss: {val_metrics['loss']:.4f} | "
        log_str += f"Val Acc: {val_metrics['accuracy']:.4f} | "
        log_str += f"Val F1: {val_metrics['f1']:.4f}"
        if val_metrics.get('auc_roc') is not None:
            log_str += f" | AUC-ROC: {val_metrics['auc_roc']:.4f}"
        print(log_str)
        
        log_metrics(epoch, epoch_metrics)
        
        # Save checkpoint
        checkpoint_path = f"{args.log_dir}/recognition_model.pth"
        save_checkpoint(model, optimizer, epoch, checkpoint_path)
        
        # Save best model
        if improved:
            best_path = f"{args.log_dir}/recognition_model_best.pth"
            save_checkpoint(model, optimizer, epoch, best_path)
            print(f"✓ New best model saved (val_loss: {val_metrics['loss']:.4f})")
        
        # Early stopping
        if tracker.should_stop(patience=args.patience):
            print(f"\n⚠ Early stopping triggered after {epoch+1} epochs (no improvement for {args.patience} epochs)")
            break
    
    # Final evaluation and visualization
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    # Load best model
    best_path = f"{args.log_dir}/recognition_model_best.pth"
    if Path(best_path).exists():
        print(f"Loading best model from {best_path}")
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on validation set with best model
    final_metrics = evaluate_classification(model, val_loader, device, num_classes)
    
    print(f"\nFinal Validation Metrics:")
    print(f"  Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"  F1 Score:  {final_metrics['f1']:.4f}")
    print(f"  Precision: {final_metrics['precision']:.4f}")
    print(f"  Recall:    {final_metrics['recall']:.4f}")
    if final_metrics.get('auc_roc') is not None:
        print(f"  AUC-ROC:   {final_metrics['auc_roc']:.4f}")
    print(f"  Loss:      {final_metrics['loss']:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        final_metrics['confusion_matrix'],
        class_names,
        Path(args.log_dir) / 'confusion_matrix.png'
    )
    
    # Plot ROC curve
    if final_metrics.get('probabilities_sampled') is not None and len(final_metrics['probabilities_sampled']) > 0:
        plot_roc_curve(
            final_metrics['labels_sampled'],
            final_metrics['probabilities_sampled'],
            num_classes,
            Path(args.log_dir) / 'roc_curve.png'
        )
    
    # Generate classification report
    if final_metrics.get('labels_sampled') is not None and len(final_metrics['labels_sampled']) > 0:
        generate_classification_report(
            final_metrics['labels_sampled'],
            final_metrics['predictions_sampled'],
            class_names,
            Path(args.log_dir) / 'classification_report.txt'
        )
    
    # Plot training curves
    tracker.plot_metrics()
    tracker.save_summary()
    
    # =========================
    # EXPORT TO TORCHSCRIPT / ONNX
    # =========================
    print("\n" + "="*60)
    print("EXPORTING MODEL")
    print("="*60)

    sample_input = torch.randn(1, 3, 224, 224).to(device)

    torchscript_path = Path(args.log_dir) / 'recognition_model_torchscript.pt'
    ModelExporter.export_torchscript(model, sample_input, torchscript_path)

    onnx_path = Path(args.log_dir) / 'recognition_model.onnx'
    ModelExporter.export_onnx(model, sample_input, onnx_path)

    inference_wrapper = InferenceWrapper(model, device, 'recognition')
    wrapper_info = inference_wrapper.get_model_info()
    print(f"✓ Inference wrapper created: {wrapper_info}")
    
    # =========================
    # TEST SET EVALUATION
    # =========================
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)

    test_metrics = evaluate_classification(model, test_loader, device, num_classes)
    print(f"\nTest Set Metrics:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    if test_metrics.get('auc_roc') is not None:
        print(f"  AUC-ROC:   {test_metrics['auc_roc']:.4f}")

    # =========================
    # FACE VERIFICATION + EMBEDDINGS
    # =========================
    verification_metrics = None
    if hasattr(model, 'embedding_layer') and args.enable_verification_eval:
        print("\nEvaluating face verification performance...")
        test_embeddings = []
        test_labels = []

        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                embeddings = model(inputs, return_embedding=True)
                test_embeddings.extend(embeddings.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

        verification_metrics = evaluate_face_verification(
            test_embeddings, test_labels,
            save_path=Path(args.log_dir) / 'verification_roc.png'
        )

        print("Face Verification Metrics:")
        print(f"  AUC: {verification_metrics['auc']:.4f}")
        print(f"  EER: {verification_metrics['eer']:.4f}")
        print(f"  TAR@FAR=0.01%: {verification_metrics['tar_at_far_001']:.4f}")

        if args.enable_embedding_viz:
            print("\nGenerating embedding visualizations.")
            try:
                import numpy as np

                # Convert to numpy arrays
                embeddings_arr = np.asarray(test_embeddings)
                labels_arr = np.asarray(test_labels)

                # Optional: subsample for speed
                max_points = 5000
                if embeddings_arr.shape[0] > max_points:
                    idx = np.random.choice(embeddings_arr.shape[0], max_points, replace=False)
                    embeddings_arr = embeddings_arr[idx]
                    labels_arr = labels_arr[idx]

                # t-SNE
                tsne = TSNE(n_components=2, random_state=42)
                embeddings_2d = tsne.fit_transform(embeddings_arr)

                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(
                    embeddings_2d[:, 0],
                    embeddings_2d[:, 1],
                    c=labels_arr,
                    cmap='tab20',
                    alpha=0.7
                )
                plt.colorbar(scatter)
                plt.title('t-SNE Visualization of Face Embeddings')
                plt.xlabel('t-SNE 1')
                plt.ylabel('t-SNE 2')
                plt.savefig(Path(args.log_dir) / 'embeddings_tsne.png', dpi=150, bbox_inches='tight')
                plt.close()
                print(f"✓ t-SNE plot saved to {Path(args.log_dir) / 'embeddings_tsne.png'}")

                # UMAP (if available)
                if umap is not None:
                    try:
                        reducer = umap.UMAP(random_state=42)
                        embeddings_umap = reducer.fit_transform(embeddings_arr)

                        plt.figure(figsize=(10, 8))
                        scatter = plt.scatter(
                            embeddings_umap[:, 0],
                            embeddings_umap[:, 1],
                            c=labels_arr,
                            cmap='tab20',
                            alpha=0.7
                        )
                        plt.colorbar(scatter)
                        plt.title('UMAP Visualization of Face Embeddings')
                        plt.xlabel('UMAP 1')
                        plt.ylabel('UMAP 2')
                        plt.savefig(Path(args.log_dir) / 'embeddings_umap.png', dpi=150, bbox_inches='tight')
                        plt.close()
                        print(f"✓ UMAP plot saved to {Path(args.log_dir) / 'embeddings_umap.png'}")
                    except Exception as e:
                        print(f"⚠ UMAP visualization failed: {e}")
                else:
                    print("⚠ UMAP not installed, skipping UMAP visualization")
            except Exception as e:
                print(f"⚠ Embedding visualization failed: {e}")
    elif args.enable_verification_eval:
        print("⚠ Face verification evaluation skipped: model does not have embedding layer")

    # =========================
    # GRAD-CAM
    # =========================
    if args.enable_gradcam and hasattr(model, 'backbone'):
        print("\nGenerating Grad-CAM explanations...")
        try:
            target_layer = None
            for module in model.backbone.modules():
                if isinstance(module, nn.Conv2d):
                    target_layer = module

            if target_layer is not None:
                grad_cam = GradCAM(model, target_layer)

                test_iter = iter(test_loader)
                images, labels = next(test_iter)
                images = images[:4].to(device)

                model.eval()
                with torch.no_grad():
                    outputs = model(images)
                    preds = torch.argmax(outputs, dim=1)

                fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                for i in range(4):
                    img = images[i].cpu()
                    pred_class = preds[i].item()
                    true_class = labels[i].item()

                    cam = grad_cam.generate_cam(img.unsqueeze(0), target_class=pred_class)
                    overlay = GradCAM.overlay_heatmap(img, cam)

                    axes[0, i].imshow(img.permute(1, 2, 0).numpy())
                    axes[0, i].set_title(f'True: {true_class}, Pred: {pred_class}')
                    axes[0, i].axis('off')

                    axes[1, i].imshow(overlay)
                    axes[1, i].set_title('Grad-CAM')
                    axes[1, i].axis('off')

                plt.tight_layout()
                plt.savefig(Path(args.log_dir) / 'gradcam_analysis.png', dpi=150, bbox_inches='tight')
                plt.close()
                print(f"✓ Grad-CAM analysis saved to {Path(args.log_dir) / 'gradcam_analysis.png'}")
            else:
                print("⚠ Could not find convolutional layer for Grad-CAM")
        except Exception as e:
            print(f"⚠ Grad-CAM analysis failed: {e}")
    elif args.enable_gradcam:
        print("⚠ Grad-CAM analysis skipped: model does not have backbone attribute")

    # =========================
    # REGION OCCLUSION ANALYSIS
    # =========================
    if args.enable_region_analysis and hasattr(model, 'backbone'):
        print("\nPerforming region-based facial occlusion analysis...")
        try:
            regions = {
                'eyes': [(0.2, 0.2, 0.8, 0.4)],
                'nose': [(0.3, 0.4, 0.7, 0.6)],
                'mouth': [(0.25, 0.6, 0.75, 0.8)],
                'left_cheek': [(0.0, 0.3, 0.3, 0.7)],
                'right_cheek': [(0.7, 0.3, 1.0, 0.7)],
                'forehead': [(0.2, 0.0, 0.8, 0.25)],
                'chin': [(0.3, 0.75, 0.7, 1.0)]
            }

            test_iter = iter(test_loader)
            images, labels = next(test_iter)
            images = images[:10].to(device)

            model.eval()
            with torch.no_grad():
                original_outputs = model(images)
                original_preds = torch.argmax(original_outputs, dim=1)

            region_sensitivities = {region: [] for region in regions.keys()}

            for region_name, region_coords in regions.items():
                occluded_images = []
                for img in images:
                    occluded_img = img.clone()
                    for x1, y1, x2, y2 in region_coords:
                        h, w = occluded_img.shape[1], occluded_img.shape[2]
                        x1_pix, x2_pix = int(x1 * w), int(x2 * w)
                        y1_pix, y2_pix = int(y1 * h), int(y2 * h)
                        occluded_img[:, y1_pix:y2_pix, x1_pix:x2_pix] = 0
                    occluded_images.append(occluded_img)

                occluded_batch = torch.stack(occluded_images).to(device)
                occluded_outputs = model(occluded_batch)
                occluded_preds = torch.argmax(occluded_outputs, dim=1)

                for i in range(len(images)):
                    original_conf = torch.softmax(original_outputs[i], dim=0)[original_preds[i]].item()
                    occluded_conf = torch.softmax(occluded_outputs[i], dim=0)[occluded_preds[i]].item()
                    sensitivity = original_conf - occluded_conf
                    region_sensitivities[region_name].append(sensitivity)

            plt.figure(figsize=(12, 6))
            region_names = list(regions.keys())
            sensitivities = [np.mean(region_sensitivities[region]) for region in region_names]
            errors = [np.std(region_sensitivities[region]) for region in region_names]

            bars = plt.bar(region_names, sensitivities, yerr=errors, capsize=5)
            plt.xlabel('Facial Region')
            plt.ylabel('Average Confidence Drop')
            plt.title('Region-Based Facial Occlusion Sensitivity Analysis')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)

            for bar, sensitivity in zip(bars, sensitivities):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                         f'{sensitivity:.3f}', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(Path(args.log_dir) / 'region_occlusion_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Region occlusion analysis saved to {Path(args.log_dir) / 'region_occlusion_analysis.png'}")

            occlusion_results = {
                'regions': region_names,
                'mean_sensitivities': sensitivities,
                'std_sensitivities': errors,
                'sample_size': len(images)
            }
            with open(Path(args.log_dir) / 'region_occlusion_results.json', 'w') as f:
                json.dump(occlusion_results, f, indent=2)
            print(f"✓ Detailed occlusion results saved to {Path(args.log_dir) / 'region_occlusion_results.json'}")
        except Exception as e:
            print(f"⚠ Region occlusion analysis failed: {e}")
    elif args.enable_region_analysis:
        print("⚠ Region occlusion analysis skipped: model does not have backbone attribute")

    # =========================
    # ROBUSTNESS TESTING UNDER CCTV CONDITIONS
    # =========================
    if args.enable_robustness_test:
        print("\nPerforming robustness testing under CCTV conditions...")
        try:
            test_conditions = {
                'original': {'blur': 0, 'noise': 0, 'compression': 100, 'occlusion': 0},
                'blur_5px': {'blur': 5, 'noise': 0, 'compression': 100, 'occlusion': 0},
                'blur_10px': {'blur': 10, 'noise': 0, 'compression': 100, 'occlusion': 0},
                'noise_01': {'blur': 0, 'noise': 0.1, 'compression': 100, 'occlusion': 0},
                'noise_02': {'blur': 0, 'noise': 0.2, 'compression': 100, 'occlusion': 0},
                'compression_50': {'blur': 0, 'noise': 0, 'compression': 50, 'occlusion': 0},
                'compression_25': {'blur': 0, 'noise': 0, 'compression': 25, 'occlusion': 0},
                'occlusion_20': {'blur': 0, 'noise': 0, 'compression': 100, 'occlusion': 0.2},
                'occlusion_40': {'blur': 0, 'noise': 0, 'compression': 100, 'occlusion': 0.4},
                'combined': {'blur': 3, 'noise': 0.1, 'compression': 70, 'occlusion': 0.1}
            }

            robustness_results = {}

            test_iter = iter(test_loader)
            original_images, original_labels = next(test_iter)
            original_images = original_images[:50].to(device)
            original_labels = original_labels[:50]

            model.eval()
            with torch.no_grad():
                original_outputs = model(original_images)
                original_preds = torch.argmax(original_outputs, dim=1)
                original_acc = (original_preds == original_labels.to(device)).float().mean().item()

            robustness_results['original'] = {
                'accuracy': original_acc,
                'condition': test_conditions['original']
            }

            for condition_name, params in test_conditions.items():
                if condition_name == 'original':
                    continue

                print(f"  Testing condition: {condition_name}")

                cctv_aug = CCTVAugmentation(
                    blur_kernel=params['blur'],
                    noise_std=params['noise'],
                    compression_quality=params['compression'],
                    occlusion_prob=params['occlusion']
                )

                augmented_images = []
                for img in original_images:
                    # Convert from float32 (0-1) to uint8 (0-255) for OpenCV
                    img_np = (img.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    augmented_img = cctv_aug.apply_augmentation(img_np)
                    # Convert back to float32 (0-1) for PyTorch
                    augmented_img_float = augmented_img.astype(np.float32) / 255.0
                    augmented_images.append(torch.from_numpy(augmented_img_float).permute(2, 0, 1))

                augmented_batch = torch.stack(augmented_images).to(device)

                augmented_outputs = model(augmented_batch)
                augmented_preds = torch.argmax(augmented_outputs, dim=1)
                augmented_acc = (augmented_preds == original_labels.to(device)).float().mean().item()

                robustness_results[condition_name] = {
                    'accuracy': augmented_acc,
                    'condition': params,
                    'accuracy_drop': original_acc - augmented_acc
                }

            plt.figure(figsize=(14, 6))

            conditions = list(robustness_results.keys())
            accuracies = [robustness_results[c]['accuracy'] for c in conditions]
            accuracy_drops = [robustness_results[c].get('accuracy_drop', 0) for c in conditions]

            plt.subplot(1, 2, 1)
            bars = plt.bar(conditions, accuracies)
            plt.xlabel('Test Condition')
            plt.ylabel('Accuracy')
            plt.title('Model Robustness Under CCTV Conditions')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.axhline(y=original_acc, color='r', linestyle='--', alpha=0.7,
                        label=f'Original: {original_acc:.3f}')
            plt.legend()

            for bar, acc in zip(bars, accuracies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                         f'{acc:.3f}', ha='center', va='bottom', fontsize=8)

            plt.subplot(1, 2, 2)
            drop_conditions = conditions[1:]
            drops = accuracy_drops[1:]
            bars = plt.bar(drop_conditions, drops)
            plt.xlabel('Test Condition')
            plt.ylabel('Accuracy Drop')
            plt.title('Accuracy Degradation Under CCTV Conditions')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)

            for bar, drop in zip(bars, drops):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                         f'{drop:.3f}', ha='center', va='bottom', fontsize=8)

            plt.tight_layout()
            plt.savefig(Path(args.log_dir) / 'robustness_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Robustness analysis saved to {Path(args.log_dir) / 'robustness_analysis.png'}")

            with open(Path(args.log_dir) / 'robustness_results.json', 'w') as f:
                json.dump(robustness_results, f, indent=2)
            print(f"✓ Detailed robustness results saved to {Path(args.log_dir) / 'robustness_results.json'}")

            print("\nRobustness Summary:")
            print(f"  Original Accuracy: {original_acc:.4f}")
            for condition in conditions[1:]:
                acc = robustness_results[condition]['accuracy']
                drop = robustness_results[condition]['accuracy_drop']
                print(f"  {condition}: {acc:.4f} (drop: {drop:.4f})")
        except Exception as e:
            print(f"⚠ Robustness testing failed: {e}")

    # =========================
    # TRAINING SUMMARY JSON
    # =========================
    summary_data = {
        'model_type': 'recognition',
        'hyperparameters': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'loss_mode': args.loss_mode,
            'triplet_margin': args.triplet_margin,
            'patience': args.patience,
            'enable_feature_selection': args.enable_feature_selection,
            'n_features': args.n_features,
            'feature_method': args.feature_method,
            'enable_cctv_aug': args.enable_cctv_aug,
            'enable_verification_eval': args.enable_verification_eval,
            'enable_embedding_viz': args.enable_embedding_viz,
            'enable_gradcam': args.enable_gradcam,
            'enable_region_analysis': args.enable_region_analysis,
            'enable_robustness_test': args.enable_robustness_test
        },
        'dataset_info': {
            'data_paths': data_paths,
            'num_classes': num_classes,
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'test_samples': len(test_dataset)
        },
        'best_epoch': tracker.best_epoch,
        'final_metrics': {
            'accuracy': final_metrics['accuracy'],
            'f1': final_metrics['f1'],
            'precision': final_metrics['precision'],
            'recall': final_metrics['recall'],
            'auc_roc': final_metrics.get('auc_roc'),
            'loss': final_metrics['loss']
        },
        'test_metrics': {
            'accuracy': test_metrics['accuracy'],
            'f1': test_metrics['f1'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'auc_roc': test_metrics.get('auc_roc')
        },
        'verification_metrics': verification_metrics,
        'system_info': {
            'device': str(device),
            'cpu_count': torch.get_num_threads(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }
    }

    summary_path = Path(args.log_dir) / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    print(f"✓ Training summary saved to {summary_path}")

    # =========================
    # XAI (SHAP / LIME)
    # =========================
    if args.enable_xai:
        print("\n" + "="*60)
        print("GENERATING EXPLAINABILITY ANALYSIS (xAI)")
        print("="*60)

        shap_result = explain_with_shap(
            model, val_loader, device,
            num_samples=min(100 if not args.fast_mode else 10, len(val_dataset)),
            save_path=Path(args.log_dir) / 'shap_summary.png'
        )

        try:
            sample_input, _ = next(iter(val_loader))
            explain_with_lime(
                model,
                sample_input[0],
                device,
                num_samples=1000 if not args.fast_mode else 100,
                save_path=Path(args.log_dir) / 'lime_explanation.png'
            )
        except Exception as e:
            print(f"⚠ LIME analysis failed: {e}")

    # =========================
    # MODEL REGISTRY
    # =========================
    print("\nTraining completed. Registering model in registry...")
    try:
        db = SessionLocal()
        try:
            existing_versions = list_model_versions(db, "recognition")
            next_version = max([v.version for v in existing_versions], default=0) + 1

            model_version = register_model_version(
                db=db,
                name="recognition",
                version=next_version,
                model_path=checkpoint_path,
                training_output_path=args.log_dir,
                set_active=True
            )
            print(f"✓ Model registered: recognition v{next_version} (ID: {model_version.id})")
        finally:
            db.close()
    except Exception as e:
        print(f"⚠ Warning: Could not register model in database: {e}")
        print("  Model file saved but not registered in model registry.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Face Recognition Model")
    parser.add_argument(
        "--data-path", 
        type=str, 
        default="Data/recognition_faces",
        help="Path to unified recognition dataset (one root with identity subfolders)"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for regularization.")
    parser.add_argument("--loss-mode", type=str, default="cross_entropy", choices=["cross_entropy", "triplet", "arcface", "cosface"], help="Loss function mode")
    parser.add_argument("--triplet-margin", type=float, default=0.5, help="Margin for triplet loss")
    parser.add_argument("--log-dir", type=str, default="./logs/recognition", help="Directory for logs and checkpoints.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs without improvement).")
    parser.add_argument("--enable-xai", action="store_true", help="Enable explainability analysis (SHAP, LIME)")
    parser.add_argument("--fast-mode", action="store_true", help="Use 128x128 images with augmentation for faster training")
    parser.add_argument("--enable-cctv-aug", action="store_true", help="Enable CCTV-specific augmentations during training")
    parser.add_argument("--enable-verification-eval", action="store_true", help="Enable face verification evaluation after training")
    parser.add_argument("--enable-embedding-viz", action="store_true", help="Enable embedding visualization (t-SNE/UMAP)")
    parser.add_argument("--enable-gradcam", action="store_true", help="Enable Grad-CAM analysis for explainability")
    parser.add_argument("--enable-region-analysis", action="store_true", help="Enable region-based facial occlusion analysis")
    parser.add_argument("--enable-robustness-test", action="store_true", help="Enable robustness testing under CCTV conditions")
    parser.add_argument("--enable-feature-selection", action="store_true",
                        help="Enable feature selection / dimensionality reduction")
    parser.add_argument("--n-features", type=int, default=None,
                        help="Number of features to select (None = auto)")
    parser.add_argument("--feature-method", type=str, default="auto",
                        choices=["auto", "pca", "lda", "lasso", "rfe", "mutual_info"],
                        help="Feature selection method")
    args = parser.parse_args()
    # Print user guidance
    print("\n[INFO] Training face recognition model")
    print("[INFO] Each subfolder should be a separate identity/class. Example: Data/actor_faces/Brad_Pitt/*.jpg")
    print("[INFO] Multiple datasets will be combined for training.\n")
    main(args)
