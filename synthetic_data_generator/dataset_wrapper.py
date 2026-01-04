import os
from torch.utils.data import Dataset
from PIL import Image

class SyntheticCCTVDataset(Dataset):
    """
    A PyTorch Dataset for loading synthetically generated CCTV-style images.
    Assumes a directory structure where each subdirectory is a class (identity).
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        self.class_to_idx = {cls: i for i, cls in enumerate(sorted(os.listdir(root_dir)))}
        
        for cls_name, cls_idx in self.class_to_idx.items():
            cls_dir = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                self.image_paths.append(os.path.join(cls_dir, img_name))
                self.labels.append(cls_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
