import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class SimpleFaceDataset(Dataset):
    """Simple dataset for face recognition training."""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = []
        self.labels = []
        
        # Get all subdirectories (each is a class)
        classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        
        for cls_name in classes:
            class_dir = os.path.join(data_dir, cls_name)
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_files.append(os.path.join(class_dir, filename))
                    self.labels.append(self.class_to_idx[cls_name])
        
        print(f"Found {len(self.image_files)} images in {len(classes)} classes in {data_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

class SimpleRecognitionNet(nn.Module):
    """Improved CNN for face recognition."""
    
    def __init__(self, num_classes):
        super(SimpleRecognitionNet, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def main(args):
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set up transforms
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets from the unified structure
    data_path = args.data_path
    if not os.path.exists(data_path):
        print(f"Error: Data path {data_path} does not exist")
        return
    
    # Load train and validation datasets
    train_dataset = SimpleFaceDataset(os.path.join(data_path, 'train'), transform=train_transform)
    val_dataset = SimpleFaceDataset(os.path.join(data_path, 'val'), transform=val_transform)
    
    print(f"Loaded training dataset: {len(train_dataset)} images, {len(train_dataset.class_to_idx)} classes")
    print(f"Loaded validation dataset: {len(val_dataset)} images, {len(val_dataset.class_to_idx)} classes")
    
    # Verify classes are consistent
    if set(train_dataset.class_to_idx.keys()) != set(val_dataset.class_to_idx.keys()):
        print("Warning: Train and validation datasets have different classes!")
        print(f"Train classes: {len(train_dataset.class_to_idx)}")
        print(f"Val classes: {len(val_dataset.class_to_idx)}")
    
    total_classes = len(train_dataset.class_to_idx)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Create model
    model = SimpleRecognitionNet(total_classes)
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    model.train()
    
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0
    
    for epoch in range(args.epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # Step the scheduler
        scheduler.step()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Loss = {epoch_loss:.4f}, Train Acc = {epoch_acc:.2f}%, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_path = os.path.join(args.log_dir, 'recognition_model_best.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"✓ New best model saved with validation accuracy: {val_acc:.2f}%")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠ Early stopping triggered after {epoch+1} epochs (no improvement for {patience} epochs)")
            break
        
        model.train()
    
    # Save final model (load best model first)
    best_model_path = os.path.join(args.log_dir, 'recognition_model_best.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model for final save (validation accuracy: {best_val_acc:.2f}%)")
    
    model_path = os.path.join(args.log_dir, 'recognition_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Final model saved to {model_path}")
    
    print("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Face Recognition Training")
    parser.add_argument("--data-path", type=str, default="Data/recognition_faces", 
                       help="Path to training data with train/val subdirectories")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--log-dir", type=str, default="./logs/recognition", help="Directory for logs and checkpoints")
    
    args = parser.parse_args()
    main(args)