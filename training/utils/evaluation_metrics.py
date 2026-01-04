"""
Comprehensive evaluation metrics for model training.
GEN-4 AI Copilot enhancement.
"""
import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class MetricsTracker:
    """Track and visualize training metrics across epochs."""
    
    def __init__(self, log_dir: str, task_type: str = 'classification'):
        """
        Args:
            log_dir: Directory to save metrics and plots
            task_type: 'classification', 'regression', or 'landmark'
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.task_type = task_type
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
        }
        
        if self.task_type == 'classification':
            self.history.update({
                'train_acc': [],
                'val_acc': [],
                'train_f1': [],
                'val_f1': [],
                'train_precision': [],
                'val_precision': [],
                'train_recall': [],
                'val_recall': [],
                'val_auc_roc': []  # AUC-ROC for classification
            })
        elif task_type in ['regression', 'landmark']:
            self.history.update({
                'train_mae': [],
                'val_mae': [],
                'train_rmse': [],
                'val_rmse': []
            })
            if task_type == 'landmark':
                self.history.update({
                    'val_nme': [],
                    'val_nme_std': [],
                    'val_failure_rate': []
                })
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
    
    def update(self, epoch: int, metrics: dict):
        """Update history with new metrics."""
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
        
        # Check for improvement
        val_loss = metrics.get('val_loss')
        if val_loss is not None:
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                return True  # Improved
            else:
                self.patience_counter += 1
                return False  # No improvement
        return False
    
    def should_stop(self, patience: int = 10) -> bool:
        """Check if training should stop based on early stopping."""
        return self.patience_counter >= patience
    
    def plot_metrics(self):
        """Generate and save metric plots."""
        if len(self.history['train_loss']) == 0:
            print("⚠ No metrics recorded, skipping metric plot.")
            return

        # Explicit epoch numbers: 1..N
        epochs = list(range(1, len(self.history['train_loss']) + 1))
        best_epoch_plot = self.best_epoch + 1  # convert from 0-based to 1-based

        try:
            # Clear any existing figures
            plt.close('all')

            fig = plt.figure(figsize=(12, 4))

            # ----- Loss curves -----
            plt.subplot(1, 2, 1)
            plt.plot(epochs, self.history['train_loss'], label='Train Loss', marker='o')
            plt.plot(epochs, self.history['val_loss'], label='Val Loss', marker='s')
            plt.axvline(
                x=best_epoch_plot,
                color='r',
                linestyle='--',
                label=f'Best Epoch ({best_epoch_plot})'
            )
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # ----- Right-hand plot -----
            if self.task_type == 'classification':
                plt.subplot(1, 2, 2)
                plt.plot(epochs, self.history['val_acc'], label='Accuracy', marker='o')
                plt.plot(epochs, self.history['val_f1'], label='F1 Score', marker='s')
                if self.history.get('val_auc_roc'):
                    plt.plot(epochs, self.history['val_auc_roc'], label='AUC-ROC', marker='d')
                plt.plot(epochs, self.history['val_precision'], label='Precision', marker='^')
                plt.plot(epochs, self.history['val_recall'], label='Recall', marker='v')
                plt.xlabel('Epoch')
                plt.ylabel('Score')
                plt.title('Validation Metrics')
                plt.legend()
                plt.grid(True, alpha=0.3)
            elif self.task_type in ['regression', 'landmark']:
                plt.subplot(1, 2, 2)
                plt.plot(epochs, self.history['val_mae'], label='MAE', marker='o')
                plt.plot(epochs, self.history['val_rmse'], label='RMSE', marker='s')
                if self.task_type == 'landmark' and self.history.get('val_nme'):
                    plt.plot(epochs, self.history['val_nme'], label='NME', marker='d')
                plt.xlabel('Epoch')
                plt.ylabel('Error')
                plt.title('Validation Errors')
                plt.legend()
                plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = self.log_dir / 'training_metrics.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)  # Explicitly close the figure
            print(f"✓ Metrics plot saved to {plot_path}")

            # Verify the file was created and has content
            if plot_path.exists():
                file_size = plot_path.stat().st_size
                if file_size < 1000:  # Less than 1KB is probably empty
                    print("⚠ Warning: Plot file seems too small, may be empty")
            else:
                print("⚠ Warning: Plot file was not created")

        except Exception as e:
            print(f"⚠ Error creating metrics plot: {e}")
            import traceback
            traceback.print_exc()
    
    def save_summary(self):
        """Save training summary to file."""
        summary_path = self.log_dir / 'training_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("=== TRAINING SUMMARY ===\n\n")
            f.write(f"Task Type: {self.task_type}\n")
            f.write(f"Total Epochs: {len(self.history['train_loss'])}\n")
            f.write(f"Best Epoch: {self.best_epoch}\n")
            f.write(f"Best Val Loss: {self.best_val_loss:.6f}\n\n")
            
            f.write("=== FINAL METRICS ===\n")
            for key, values in self.history.items():
                if values:
                    f.write(f"{key}: {values[-1]:.6f}\n")
            
            if self.task_type == 'landmark':
                f.write("\n=== LANDMARK METRICS ===\n")
                if self.history.get('val_nme'):
                    f.write(f"NME: {self.history['val_nme'][-1]:.6f}\n")
                if self.history.get('val_failure_rate'):
                    f.write(f"Failure Rate: {self.history['val_failure_rate'][-1]:.6f}\n")
            
            f.write("\n=== BEST METRICS (at epoch {}) ===\n".format(self.best_epoch))
            for key, values in self.history.items():
                if values and len(values) > self.best_epoch:
                    f.write(f"{key}: {values[self.best_epoch]:.6f}\n")
        
        print(f"✓ Training summary saved to {summary_path}")


def evaluate_classification(model, dataloader, device, num_classes: int, *, max_sample_collect: int = 100000):
    """
    Evaluate classification model and compute all metrics.
    
    Returns:
        dict with accuracy, f1, precision, recall, auc_roc, loss, confusion_matrix
    """
    model.eval()
    # To avoid unbounded memory growth we use a hybrid strategy:
    # - Maintain running counts for accuracy (exact)
    # - Keep a reservoir sample up to `max_sample_collect` for computing
    #   precision/recall/F1 and AUC approximately when datasets are large.
    import random

    sampled_preds = []
    sampled_labels = []
    sampled_probs = []
    sample_k = int(max_sample_collect)

    total_loss = 0.0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        global_index = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()
            probs_np = probs.cpu().numpy()

            # running accuracy
            correct += int((preds_np == labels_np).sum())
            total += labels_np.shape[0]

            # reservoir sampling into sampled_* lists
            for i in range(labels_np.shape[0]):
                if len(sampled_labels) < sample_k:
                    sampled_labels.append(int(labels_np[i]))
                    sampled_preds.append(int(preds_np[i]))
                    sampled_probs.append(probs_np[i])
                else:
                    r = random.randint(0, global_index)
                    if r < sample_k:
                        idx = r
                        sampled_labels[idx] = int(labels_np[i])
                        sampled_preds[idx] = int(preds_np[i])
                        sampled_probs[idx] = probs_np[i]
                global_index += 1
    
    # Finalize metrics
    accuracy = float(correct) / max(1, total)

    # Use sampled data for F1/precision/recall/AUC to limit memory usage
    sampled_labels = np.array(sampled_labels)
    sampled_preds = np.array(sampled_preds)
    sampled_probs = np.stack(sampled_probs, axis=0) if len(sampled_probs) > 0 else np.array([])

    average = 'binary' if num_classes == 2 else 'weighted'
    try:
        f1 = f1_score(sampled_labels, sampled_preds, average=average, zero_division=0) if sampled_labels.size else None
        precision = precision_score(sampled_labels, sampled_preds, average=average, zero_division=0) if sampled_labels.size else None
        recall = recall_score(sampled_labels, sampled_preds, average=average, zero_division=0) if sampled_labels.size else None
    except Exception:
        f1 = precision = recall = None
    
    # Compute AUC-ROC
    auc_roc = None
    try:
        if sampled_probs.size and num_classes == 2:
            auc_roc = roc_auc_score(sampled_labels, sampled_probs[:, 1])
        elif sampled_probs.size and num_classes > 2:
            unique_classes = len(np.unique(sampled_labels))
            if unique_classes == num_classes:
                auc_roc = roc_auc_score(sampled_labels, sampled_probs, multi_class='ovr', average='weighted')
            else:
                auc_roc = None  # Not enough classes in sample for reliable AUC
    except Exception as e:
        print(f"Warning: Could not compute AUC-ROC from sampled data: {e}")
        auc_roc = None
    
    conf_matrix = confusion_matrix(sampled_labels, sampled_preds) if sampled_labels.size else None

    return {
        'loss': total_loss / max(1, len(dataloader)),
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc_roc': auc_roc,
        'confusion_matrix': conf_matrix,
        'predictions_sampled': sampled_preds,
        'labels_sampled': sampled_labels,
        'probabilities_sampled': sampled_probs,
        # aliases to keep older code working
        'predictions': sampled_preds,
        'labels': sampled_labels,
        'probabilities': sampled_probs,
        'total_samples_evaluated': int(total)
    }


def evaluate_regression(model, dataloader, device):
    """
    Evaluate regression/landmark model and compute MAE, RMSE.
    
    Returns:
        dict with loss, mae, rmse
    """
    model.eval()
    # Online accumulators to avoid holding everything in memory
    count = 0
    sum_abs_err = 0.0
    sum_sq_err = 0.0
    total_loss = 0.0
    criterion = torch.nn.MSELoss()
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            out_np = outputs.cpu().numpy()
            tgt_np = targets.cpu().numpy()
            # accumulate errors
            err = tgt_np - out_np
            sum_abs_err += float(np.abs(err).sum())
            sum_sq_err += float((err ** 2).sum())
            count += out_np.shape[0]

    mae = float(sum_abs_err) / max(1, count)
    mse = float(sum_sq_err) / max(1, count)
    rmse = np.sqrt(mse)

    return {
        'loss': total_loss / max(1, len(dataloader)),
        'mae': mae,
        'rmse': rmse,
        'total_samples_evaluated': int(count)
    }


def plot_confusion_matrix(conf_matrix, class_names, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names if len(class_names) <= 20 else False,
                yticklabels=class_names if len(class_names) <= 20 else False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved to {save_path}")


def generate_classification_report(labels, predictions, class_names, save_path):
    """Generate and save detailed classification report."""
    report = classification_report(labels, predictions, 
                                   target_names=class_names if len(class_names) <= 100 else None,
                                   zero_division=0)
    
    with open(save_path, 'w') as f:
        f.write("=== CLASSIFICATION REPORT ===\n\n")
        f.write(report)
    
    print(f"✓ Classification report saved to {save_path}")
    return report


def plot_roc_curve(labels, probabilities, num_classes, save_path):
    """Plot and save ROC curve(s)."""
    plt.figure(figsize=(10, 8))
    
    if num_classes == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(labels, probabilities[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
    else:
        # Multi-class (plot macro-average and per-class)
        from sklearn.preprocessing import label_binarize
        
        # Binarize labels
        labels_bin = label_binarize(labels, classes=np.arange(num_classes))
        
        # Compute ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(min(num_classes, 10)):  # Plot max 10 classes
            try:
                fpr[i], tpr[i], _ = roc_curve(labels_bin[:, i], probabilities[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                plt.plot(fpr[i], tpr[i], lw=2, alpha=0.7,
                        label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
            except:
                pass
        
        # Compute macro-average in a memory-frugal way
        # Collect only existing fpr arrays (up to 10 classes)
        present = [i for i in range(min(num_classes, 10)) if i in fpr]
        if present:
            # Build a sorted unique array without concatenating large arrays all at once
            union_set = set()
            for i in present:
                # flatten and convert to native floats to avoid large temporary numpy concatenations
                union_set.update(map(float, np.ravel(fpr[i])))
            all_fpr = np.array(sorted(union_set))
            mean_tpr = np.zeros_like(all_fpr, dtype=float)
            for i in present:
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= max(1, len(present))
        else:
            # Fallback minimal curve
            all_fpr = np.array([0.0, 1.0])
            mean_tpr = np.array([0.0, 1.0])
        
        macro_auc = auc(all_fpr, mean_tpr)
        plt.plot(all_fpr, mean_tpr, color='navy', lw=3,
                label=f'Macro-average (AUC = {macro_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class ROC Curves')
        plt.legend(loc="lower right", fontsize=8)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ ROC curve saved to {save_path}")


def vertex_error(pred_mesh, gt_mesh):
    """
    Compute vertex-wise error for 3D mesh reconstruction.
    
    Args:
        pred_mesh: Predicted mesh vertices (N, 3) or (B, N, 3)
        gt_mesh: Ground truth mesh vertices (N, 3) or (B, N, 3)
    
    Returns:
        dict with mean_vertex_error, std_vertex_error, max_vertex_error
    """
    pred_mesh = np.array(pred_mesh)
    gt_mesh = np.array(gt_mesh)
    
    if pred_mesh.ndim == 2 and gt_mesh.ndim == 2:
        # Single mesh
        errors = np.linalg.norm(pred_mesh - gt_mesh, axis=1)
    elif pred_mesh.ndim == 3 and gt_mesh.ndim == 3:
        # Batch of meshes
        errors = np.linalg.norm(pred_mesh - gt_mesh, axis=2)
        errors = errors.flatten()
    else:
        raise ValueError("Mesh dimensions must be (N, 3) or (B, N, 3)")
    
    return {
        'mean_vertex_error': float(np.mean(errors)),
        'std_vertex_error': float(np.std(errors)),
        'max_vertex_error': float(np.max(errors))
    }


def landmark_3d_error(pred_landmarks, gt_landmarks):
    """
    Compute 3D landmark error.
    
    Args:
        pred_landmarks: Predicted 3D landmarks (N, 3) or (B, N, 3)
        gt_landmarks: Ground truth 3D landmarks (N, 3) or (B, N, 3)
    
    Returns:
        dict with mean_landmark_error, std_landmark_error, max_landmark_error
    """
    pred_landmarks = np.array(pred_landmarks)
    gt_landmarks = np.array(gt_landmarks)
    
    if pred_landmarks.ndim == 2 and gt_landmarks.ndim == 2:
        # Single set of landmarks
        errors = np.linalg.norm(pred_landmarks - gt_landmarks, axis=1)
    elif pred_landmarks.ndim == 3 and gt_landmarks.ndim == 3:
        # Batch of landmark sets
        errors = np.linalg.norm(pred_landmarks - gt_landmarks, axis=2)
        errors = errors.flatten()
    else:
        raise ValueError("Landmark dimensions must be (N, 3) or (B, N, 3)")
    
    return {
        'mean_landmark_error': float(np.mean(errors)),
        'std_landmark_error': float(np.std(errors)),
        'max_landmark_error': float(np.max(errors))
    }


def visualize_landmarks(images, pred_landmarks, gt_landmarks, save_path, title="Landmark Visualization"):
    """
    Visualize predicted vs ground truth landmarks overlaid on images.
    
    Args:
        images: Batch of images (B, C, H, W)
        pred_landmarks: Predicted landmarks (B, N, 2) 
        gt_landmarks: Ground truth landmarks (B, N, 2)
        save_path: Path to save visualization
        title: Plot title
    """
    import matplotlib.pyplot as plt
    
    batch_size = min(images.shape[0], 4)  # Show max 4 images
    fig, axes = plt.subplots(1, batch_size, figsize=(4*batch_size, 4))
    if batch_size == 1:
        axes = [axes]
    
    for i in range(batch_size):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0,1]
        
        axes[i].imshow(img)
        
        # Plot ground truth landmarks (green)
        gt_lm = gt_landmarks[i]
        axes[i].scatter(gt_lm[:, 0], gt_lm[:, 1], c='green', s=10, alpha=0.7, label='GT')
        
        # Plot predicted landmarks (red)
        pred_lm = pred_landmarks[i]
        axes[i].scatter(pred_lm[:, 0], pred_lm[:, 1], c='red', s=10, alpha=0.7, label='Pred')
        
        axes[i].set_title(f'Sample {i+1}')
        axes[i].legend()
        axes[i].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Landmark visualization saved to {save_path}")


def evaluate_face_verification(embeddings, labels, save_path=None):
    """
    Evaluate face verification performance using embeddings.
    
    Args:
        embeddings: Face embeddings (N, D)
        labels: Identity labels (N,)
        save_path: Path to save ROC curve plot
    
    Returns:
        dict with auc, eer, tar_at_far_001, etc.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.metrics import roc_curve, auc
    import numpy as np
    
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    # Generate positive and negative pairs
    positive_pairs = []
    negative_pairs = []
    
    unique_labels = np.unique(labels)
    
    for i in range(len(unique_labels)):
        for j in range(i+1, len(unique_labels)):
            label1, label2 = unique_labels[i], unique_labels[j]
            
            # Get embeddings for each identity
            emb1 = embeddings[labels == label1]
            emb2 = embeddings[labels == label2]
            
            # Positive pairs (same identity)
            for e1 in emb1:
                for e2 in emb1:
                    if not np.array_equal(e1, e2):  # Different images
                        positive_pairs.append((e1, e2))
            
            # Negative pairs (different identities)
            for e1 in emb1:
                for e2 in emb2:
                    negative_pairs.append((e1, e2))
    
    # Limit pairs for efficiency
    max_pairs = 10000
    if len(positive_pairs) > max_pairs:
        positive_pairs = positive_pairs[:max_pairs]
    if len(negative_pairs) > max_pairs:
        negative_pairs = negative_pairs[:max_pairs]
    
    # Compute similarities
    positive_similarities = []
    negative_similarities = []
    
    for emb1, emb2 in positive_pairs:
        sim = cosine_similarity([emb1], [emb2])[0][0]
        positive_similarities.append(sim)
    
    for emb1, emb2 in negative_pairs:
        sim = cosine_similarity([emb1], [emb2])[0][0]
        negative_similarities.append(sim)
    
    # Create labels and scores for ROC
    y_true = [1] * len(positive_similarities) + [0] * len(negative_similarities)
    y_scores = positive_similarities + negative_similarities
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Compute EER (Equal Error Rate)
    fnr = 1 - tpr  # False Negative Rate
    eer_threshold = thresholds[np.nanargmin(np.absolute(fpr - fnr))]
    eer = fpr[np.nanargmin(np.absolute(fpr - fnr))]
    
    # TAR@FAR=0.01% (True Acceptance Rate at False Acceptance Rate = 0.01%)
    far_001_threshold = thresholds[np.searchsorted(fpr, 0.0001, side='left')]
    tar_at_far_001 = tpr[np.searchsorted(fpr, 0.0001, side='left')]
    
    # Plot ROC curve if save_path provided
    if save_path:
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.scatter([eer], [1-eer], color='red', s=50, 
                   label=f'EER = {eer:.4f} (threshold = {eer_threshold:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FAR)')
        plt.ylabel('True Positive Rate (TAR)')
        plt.title('Face Verification ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Face verification ROC curve saved to {save_path}")
    
    return {
        'auc': float(roc_auc),
        'eer': float(eer),
        'eer_threshold': float(eer_threshold),
        'tar_at_far_001': float(tar_at_far_001),
        'num_positive_pairs': len(positive_similarities),
        'num_negative_pairs': len(negative_similarities)
    }


def explain_with_shap(model, dataloader, device, num_samples=100, save_path=None):
    """
    Generate SHAP explanations for the classification model.

    Args:
        model: Trained PyTorch model
        dataloader: DataLoader for validation or test set
        device: torch.device
        num_samples: Number of samples to use for SHAP
        save_path: Optional path to save summary plot
    """
    try:
        import shap
    except ImportError:
        print("⚠ SHAP is not installed. Install with 'pip install shap' to enable SHAP explanations.")
        return

    model.eval()
    model.to(device)

    # Collect a subset of data for SHAP
    samples = []
    labels = []
    for i, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        samples.append(inputs)
        labels.append(targets)
        if len(samples) * inputs.size(0) >= num_samples:
            break

    if not samples:
        print("⚠ No samples available for SHAP analysis.")
        return

    inputs = torch.cat(samples, dim=0)
    labels = torch.cat(labels, dim=0)
    inputs = inputs[:num_samples]
    labels = labels[:num_samples]

    print(f"Running SHAP analysis on {inputs.size(0)} samples...")

    # Wrap model prediction for SHAP (fallback only)
    def model_predict(x):
        x = torch.from_numpy(x).to(device)
        with torch.no_grad():
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)
        return probs.cpu().numpy()

    # Prepare data as tensors for GradientExplainer
    background = inputs[:min(50, len(inputs))]
    test_data = inputs

    # Use GradientExplainer for PyTorch models
    try:
        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(test_data)
        
        # Convert to numpy for plotting
        if isinstance(shap_values, list):
            shap_values = [sv.cpu().numpy() for sv in shap_values]
        else:
            shap_values = shap_values.cpu().numpy()
        test_data_plot = test_data.cpu().numpy()
        
    except Exception as e:
        print(f"⚠ GradientExplainer failed ({e}), falling back to KernelExplainer (may be slower).")
        background_numpy = background.cpu().numpy()
        test_data_plot = test_data.cpu().numpy()
        
        # For KernelExplainer, flatten images to 2D
        original_shape = background_numpy.shape[1:]  # (C, H, W)
        background_flat = background_numpy.reshape(background_numpy.shape[0], -1)
        test_data_flat = test_data_plot.reshape(test_data_plot.shape[0], -1)
        
        def model_predict_flat(x):
            # Reshape back to image shape
            x_reshaped = x.reshape(x.shape[0], *original_shape)
            return model_predict(x_reshaped)
        
        explainer = shap.KernelExplainer(model_predict_flat, background_flat)
        shap_values = explainer.shap_values(test_data_flat, nsamples=100)

    if save_path:
        # Summary plot for the top class (if shap_values is a list)
        plt.figure(figsize=(10, 6))
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[0], test_data_plot, show=False)
        else:
            shap.summary_plot(shap_values, test_data_plot, show=False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ SHAP summary plot saved to {save_path}")

    return {
        "shap_values": shap_values,
        "inputs": test_data,
        "labels": labels.cpu().numpy()
    }


def explain_with_lime(model, image_tensor, device, num_samples=1000, save_path=None):
    """
    Generate a LIME explanation for a single image.

    Args:
        model: Trained PyTorch model
        image_tensor: Single image tensor (C, H, W), normalized like training data
        device: torch.device
        num_samples: Number of perturbation samples for LIME
        save_path: Optional path to save visualization
    """
    try:
        from lime import lime_image
        from skimage.segmentation import mark_boundaries
    except ImportError:
        print("⚠ LIME is not installed. Install with 'pip install lime' to enable LIME explanations.")
        return {"status": "not_available", "message": "lime not installed"}

    model.eval()
    model.to(device)

    # Denormalize to [0,1] for LIME
    img = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img * std + mean).clip(0, 1)

    def batch_predict(images):
        images = np.array(images)
        # [N, H, W, C] -> [N, C, H, W]
        images = images.transpose(0, 3, 1, 2)
        # Normalize back to model's expected input
        images = (images - mean.reshape(1, 3, 1, 1)) / std.reshape(1, 3, 1, 1)
        images_t = torch.from_numpy(images).float().to(device)
        with torch.no_grad():
            logits = model(images_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img,
        batch_predict,
        top_labels=1,
        num_samples=num_samples
    )

    if save_path:
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=5,
            hide_rest=False
        )
        plt.figure(figsize=(5, 5))
        plt.imshow(mark_boundaries(temp, mask))
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ LIME explanation saved to {save_path}")

    return {"status": "ok", "explanation": explanation}

