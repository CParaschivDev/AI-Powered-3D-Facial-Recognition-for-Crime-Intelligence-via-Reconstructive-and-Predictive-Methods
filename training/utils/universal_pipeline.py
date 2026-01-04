"""
Universal Pipeline Components
GEN-4 AI Copilot enhancement for comprehensive deep learning pipelines.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import warnings
import io
from contextlib import redirect_stdout, redirect_stderr

warnings.filterwarnings('ignore')


class WarmupCosineScheduler:
    """
    Learning rate scheduler with warmup and cosine annealing.
    """
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.current_step = 0

        # Calculate initial LR
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self):
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            # Linear warmup
            lr_scale = self.current_step / self.warmup_steps
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr_scale = 0.5 * (1 + np.cos(np.pi * progress))

        # Apply minimum LR
        lr_scale = max(lr_scale, self.min_lr / min(self.base_lrs))

        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = self.base_lrs[i] * lr_scale

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


class ModelExporter:
    """
    Export models to TorchScript and ONNX formats.
    """

    @staticmethod
    def export_torchscript(model, sample_input, save_path):
        """Export model to TorchScript format."""
        try:
            model.eval()
            with torch.no_grad():
                # Trace the model
                traced_model = torch.jit.trace(model, sample_input)

                # Save TorchScript model
                torch.jit.save(traced_model, save_path)
                print(f"✓ TorchScript model exported to {save_path}")
                return True
        except Exception as e:
            print(f"⚠ TorchScript export failed: {e}")
            return False

    @staticmethod
    def export_onnx(model, sample_input, save_path, input_names=None, output_names=None):
        """Export model to ONNX format."""
        try:
            import onnxruntime as ort

            model.eval()
            with torch.no_grad():
                # Suppress torch.onnx.export diagnostic banner
                buf = io.StringIO()
                with redirect_stdout(buf), redirect_stderr(buf):
                    torch.onnx.export(
                        model,
                        sample_input,
                        save_path,
                        export_params=True,
                        opset_version=11,
                        do_constant_folding=True,
                        input_names=input_names or ['input'],
                        output_names=output_names or ['output'],
                        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                    )

                print(f"✓ ONNX model exported to {save_path}")

                # Verify ONNX model
                ort_session = ort.InferenceSession(save_path)
                print("✓ ONNX model verification passed")
                return True
        except ImportError:
            print("⚠ ONNX export skipped: onnxruntime not installed")
            return False
        except Exception as e:
            print(f"⚠ ONNX export failed: {e}")
            return False


class UniversalTransforms:
    """
    Standardized data transformations and augmentations for all models.
    """

    @staticmethod
    def get_train_transforms(model_type='universal', img_size=224):
        """Get standardized training transforms."""
        import torchvision.transforms as transforms

        if model_type == 'recognition':
            return transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif model_type in ['landmarks', 'reconstruction']:
            return transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:  # universal
            return transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    @staticmethod
    def get_val_transforms(img_size=224):
        """Get standardized validation/test transforms."""
        import torchvision.transforms as transforms

        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


class DatasetSplitter:
    """
    Standardized train/val/test splitting for all datasets.
    """

    @staticmethod
    def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
        """
        Split dataset into train/val/test sets.

        Args:
            dataset: PyTorch dataset
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            seed: Random seed for reproducibility

        Returns:
            train_dataset, val_dataset, test_dataset
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

        dataset_size = len(dataset)
        train_size = int(train_ratio * dataset_size)
        val_size = int(val_ratio * dataset_size)
        test_size = dataset_size - train_size - val_size

        # Split the dataset
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.manual_seed(seed)
        )

        print(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
        return train_dataset, val_dataset, test_dataset


class InferenceWrapper:
    """
    Standardized inference wrapper for all models.
    """

    def __init__(self, model, device='cuda', model_type='universal'):
        self.model = model.to(device)
        self.device = device
        self.model_type = model_type
        self.model.eval()

    def predict(self, input_data, return_embeddings=False):
        """
        Run inference on input data.

        Args:
            input_data: Input tensor or batch
            return_embeddings: Whether to return embeddings (for recognition models)

        Returns:
            Model predictions
        """
        with torch.no_grad():
            if not isinstance(input_data, torch.Tensor):
                input_data = torch.tensor(input_data)

            input_data = input_data.to(self.device)

            if self.model_type == 'recognition':
                if return_embeddings:
                    return self.model(input_data, return_embedding=True)
                else:
                    return self.model(input_data)
            else:
                return self.model(input_data)

    def get_model_info(self):
        """Get model information and capabilities."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'model_type': self.model_type,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': self.device
        }


class GradCAM:
    """
    Grad-CAM implementation for model explainability.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Move model to device to avoid device mismatches
        device = next(self.model.parameters()).device
        self.model = self.model.to(device)

        # Hook to capture gradients and activations
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: Input image tensor
            target_class: Target class for classification (None for regression)

        Returns:
            CAM heatmap
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)

        # Forward pass
        output = self.model(input_tensor)

        if target_class is not None:
            # For classification
            one_hot = torch.zeros_like(output)
            one_hot[0, target_class] = 1
            self.model.zero_grad()
            output.backward(gradient=one_hot, retain_graph=True)
        else:
            # For regression (use output gradient)
            self.model.zero_grad()
            output.backward(retain_graph=True)

        # Generate CAM
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)

        # Normalize
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-8)

        return cam.squeeze().cpu().numpy()

    @staticmethod
    def overlay_heatmap(img, cam, alpha=0.5):
        """Overlay CAM heatmap on original image."""
        import cv2

        # Resize CAM to image size
        cam_resized = cv2.resize(cam, (img.shape[2], img.shape[1]))

        # Convert to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)

        # Convert image to BGR if needed
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()
            if img.dim() == 3 and img.shape[0] == 3:
                # [C,H,W] -> [H,W,C]
                img_bgr = img.permute(1, 2, 0).numpy()
            else:
                img_bgr = img.numpy()
        else:
            img_bgr = img

        img_bgr = np.uint8(255 * img_bgr)

        # Overlay
        overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap, alpha, 0)
        return overlay


def plot_ced_curve(errors, save_path=None, title="Cumulative Error Distribution"):
    """
    Plot Cumulative Error Distribution (CED) curve for landmark evaluation.

    Args:
        errors: Array of normalized mean errors
        save_path: Path to save the plot
        title: Plot title
    """
    errors = np.array(errors)
    errors_sorted = np.sort(errors)

    # Calculate cumulative distribution
    y = np.arange(1, len(errors_sorted) + 1) / len(errors_sorted)
    x = errors_sorted

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2, label='CED Curve')
    plt.xlabel('Normalized Mean Error')
    plt.ylabel('Cumulative Percentage')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add key percentiles
    for percentile in [0.5, 0.7, 0.9]:
        error_val = np.percentile(errors_sorted, percentile * 100)
        plt.axvline(x=error_val, color='r', linestyle='--', alpha=0.7,
                   label='.0f')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ CED curve saved to {save_path}")
    else:
        plt.show()


def calculate_nme(landmarks_pred, landmarks_gt, normalize_by='interocular'):
    """
    Calculate Normalized Mean Error (NME) for landmark detection.

    Args:
        landmarks_pred: Predicted landmarks (N, 2) or (N, 136) for 68 points
        landmarks_gt: Ground truth landmarks
        normalize_by: Normalization method ('interocular', 'pupil', 'bbox')

    Returns:
        NME value
    """
    landmarks_pred = np.array(landmarks_pred).reshape(-1, 2)
    landmarks_gt = np.array(landmarks_gt).reshape(-1, 2)

    # Calculate normalization factor
    if normalize_by == 'interocular':
        # Distance between eye corners (points 36 and 45 in 68-point model)
        if landmarks_gt.shape[0] >= 68:
            left_eye = landmarks_gt[36]   # left eye outer corner
            right_eye = landmarks_gt[45]  # right eye outer corner
            norm_factor = np.linalg.norm(left_eye - right_eye)
        else:
            # Fallback: use first and last points
            norm_factor = np.linalg.norm(landmarks_gt[0] - landmarks_gt[-1])
    elif normalize_by == 'pupil':
        # Distance between pupil centers
        if landmarks_gt.shape[0] >= 68:
            left_pupil = landmarks_gt[37:42].mean(axis=0)   # left eye landmarks
            right_pupil = landmarks_gt[43:48].mean(axis=0)  # right eye landmarks
            norm_factor = np.linalg.norm(left_pupil - right_pupil)
        else:
            norm_factor = 100  # fallback
    else:  # bbox
        bbox_size = max(
            landmarks_gt[:, 0].max() - landmarks_gt[:, 0].min(),
            landmarks_gt[:, 1].max() - landmarks_gt[:, 1].min()
        )
        norm_factor = bbox_size

    # Calculate mean Euclidean distance
    distances = np.linalg.norm(landmarks_pred - landmarks_gt, axis=1)
    nme = np.mean(distances) / norm_factor

    return nme


def evaluate_face_verification(embeddings, labels, save_path=None):
    """
    Evaluate face verification performance.

    Args:
        embeddings: Face embeddings (N, D)
        labels: Identity labels (N,)
        save_path: Path to save ROC curve

    Returns:
        dict with verification metrics
    """
    from sklearn.metrics.pairwise import cosine_similarity

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    # Generate all possible pairs
    n_samples = len(embeddings)
    similarities = []
    same_identity = []

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0, 0]
            similarities.append(sim)
            same_identity.append(labels[i] == labels[j])

    similarities = np.array(similarities)
    same_identity = np.array(same_identity)

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(same_identity, similarities)
    roc_auc = auc(fpr, tpr)

    # Calculate EER (Equal Error Rate)
    fnr = 1 - tpr  # False Negative Rate
    eer_threshold = thresholds[np.argmin(np.abs(fpr - fnr))]
    eer = fpr[np.argmin(np.abs(fpr - fnr))]

    # Calculate TAR@FAR=0.01 (True Accept Rate at False Accept Rate = 1%)
    far_1_percent_idx = np.where(fpr <= 0.01)[0]
    if len(far_1_percent_idx) > 0:
        tar_at_far_001 = tpr[far_1_percent_idx[-1]]
    else:
        tar_at_far_001 = 0.0

    # Plot ROC curve
    if save_path:
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='.3f')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Face Verification ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Verification ROC curve saved to {save_path}")

    return {
        'auc': roc_auc,
        'eer': eer,
        'eer_threshold': eer_threshold,
        'tar_at_far_001': tar_at_far_001,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds
    }