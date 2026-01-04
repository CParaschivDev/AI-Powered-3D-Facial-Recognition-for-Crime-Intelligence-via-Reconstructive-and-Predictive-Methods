from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import torch
import scipy.io

class GenericImageDataset(Dataset):
    """
    A generic dataset for image-related tasks (classification, regression).
    Assumes data is in a directory where each subdirectory is a class.
    """
    def __init__(self, data_dir, split='train', transform=None):
        # Try data_dir/split, fallback to data_dir if not found
        split_dir = os.path.join(data_dir, split) if split else data_dir
        if os.path.isdir(split_dir):
            self.data_dir = split_dir
        elif os.path.isdir(data_dir):
            self.data_dir = data_dir
        else:
            raise FileNotFoundError(f"Data directory not found: {split_dir} or {data_dir}")
        self.transform = transform
        self.image_files = []
        self.labels = []

        self.classes = sorted([d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for cls_name in self.classes:
            class_dir = os.path.join(self.data_dir, cls_name)
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_files.append(os.path.join(class_dir, filename))
                    self.labels.append(self.class_to_idx[cls_name])
        print(f"Found {len(self.image_files)} images in {len(self.classes)} classes in {self.data_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

# --- LandmarkDataset for AFLW2000 ---
class LandmarkDataset(Dataset):
    """
    Dataset for facial landmark detection using AFLW2000.
    Each image has a corresponding .mat file with landmark coordinates.
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = []
        self.landmark_files = []

        # Find all .jpg files and their .mat pairs
        for fname in os.listdir(self.data_dir):
            if fname.lower().endswith('.jpg'):
# --- LandmarkDataset for AFLW2000 ---
                img_path = os.path.join(self.data_dir, fname)
                mat_path = os.path.splitext(img_path)[0] + '.mat'
                if os.path.exists(mat_path):
                    self.image_files.append(img_path)
                    self.landmark_files.append(mat_path)
        print(f"Found {len(self.image_files)} image/.mat pairs in {self.data_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mat_path = self.landmark_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # Load landmarks from .mat file
        import scipy.io
        mat = scipy.io.loadmat(mat_path)
        # AFLW2000: landmarks are usually in 'pt3d_68' or similar
        # Try common keys
        for key in ['pt3d_68', 'pt2d', 'landmarks', 'pts']:
            if key in mat:
                landmarks = mat[key]
                break
        else:
            raise KeyError(f"No known landmark key found in {mat_path}")
        # Squeeze and flatten to (N, 2)
        landmarks = np.array(landmarks)
        if landmarks.shape[-1] == 2:
            pass
        elif landmarks.shape[0] == 2:
            landmarks = landmarks.T
        else:
            landmarks = landmarks.reshape(-1, 2)
        # Normalize landmarks to [0,1] relative to image size (assume 224x224 after transform)
        landmarks = landmarks / 224.0
        return image, torch.tensor(landmarks, dtype=torch.float32)

def get_default_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_fast_recognition_transform():
    """Faster transform for recognition training with data augmentation."""
    return transforms.Compose([
        transforms.Resize((128, 128)),  # Smaller size = faster
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_landmark_dataset(data_dir, split=None):
    # Loads images and their corresponding landmark files for AFLW2000
    # Ignore split, use all data in the folder
    return LandmarkDataset(data_dir, transform=get_default_transform())

class ReconstructionDataset(Dataset):
    """
    Loads images and corresponding .mat 3DMM parameter files for 3D face reconstruction.
    Each sample: (image, params) where params is a 1D tensor of 3DMM parameters.
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = []
        self.param_files = []

        # Find all .jpg files and check for corresponding .mat
        for fname in os.listdir(self.data_dir):
            if fname.lower().endswith('.jpg'):
                base = fname[:-4]
                mat_path = os.path.join(self.data_dir, base + '.mat')
                img_path = os.path.join(self.data_dir, fname)
                if os.path.isfile(mat_path):
                    self.image_files.append(img_path)
                    self.param_files.append(mat_path)
        print(f"Found {len(self.image_files)} image/.mat pairs for 3DMM in {self.data_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mat_path = self.param_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        import scipy.io
        mat = scipy.io.loadmat(mat_path)
        
        # Handle AFLW2000 format with separate parameter types
        if 'Shape_Para' in mat and 'Exp_Para' in mat and 'Pose_Para' in mat:
            # Concatenate Shape (199), Expression (29), and Pose (7) parameters
            shape_params = mat['Shape_Para'].flatten()
            exp_params = mat['Exp_Para'].flatten()
            pose_params = mat['Pose_Para'].flatten()
            params = np.concatenate([shape_params, exp_params, pose_params])
        else:
            # Try common keys for 3DMM params
            for key in ['params', '3dmm_params', 'coeffs', 'param', 'alpha']:
                if key in mat:
                    params = mat[key]
                    break
            else:
                # Fallback: use first 1D or 2D ndarray
                params = next((v for v in mat.values() if isinstance(v, np.ndarray) and (v.ndim == 1 or (v.ndim == 2 and min(v.shape) == 1))), None)
                if params is None:
                    raise KeyError(f"No known 3DMM param key found in {mat_path}")
            params = np.array(params).flatten()
        
        # Normalize parameters to prevent huge MSE values
        params = params.astype(np.float32)
        params = (params - np.mean(params)) / (np.std(params) + 1e-8)
        
        return image, torch.tensor(params, dtype=torch.float32)

def get_reconstruction_dataset(data_dir, split=None):
    # Loads images and their corresponding 3DMM parameter files for 3D face reconstruction
    # Ignore split, use all data in the folder
    return ReconstructionDataset(data_dir, transform=get_default_transform())
def get_landmark_dataset(data_dir, split):
    # AFLW2000 does not have train/val/test splits by default, so ignore split
    return LandmarkDataset(data_dir, transform=get_default_transform())
def get_recognition_dataset(data_dir, split, fast_mode=False):
    # This would load images and their identity labels
    transform = get_fast_recognition_transform() if fast_mode else get_default_transform()
    return GenericImageDataset(data_dir, split, transform=transform)
