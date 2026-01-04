import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from training.landmark_train import LandmarkNet
from training.utils.dataset_loader import get_landmark_dataset

MODEL_PATH = "logs/landmarks/landmark_model.pth"
DATA_PATH = "Data/AFLW2000"
OUTPUT_DIR = "logs/landmarks"

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load dataset
    dataset = get_landmark_dataset(data_dir=DATA_PATH, split=None)
    print(f"Loaded {len(dataset)} samples from {DATA_PATH}")

    # Infer number of landmarks
    _, sample_landmarks = dataset[0]
    num_landmarks = sample_landmarks.shape[0]
    print(f"Detected {num_landmarks} landmarks per face.")

    # Load model
    model = LandmarkNet(num_landmarks=num_landmarks).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    preds = []
    gts = []

    for img, gt in tqdm(dataset, desc="Predicting landmarks"):
        with torch.no_grad():
            img_tensor = img.unsqueeze(0).to(device)
            pred = model(img_tensor).cpu().numpy()[0]
        preds.append(pred)
        gts.append(gt.numpy())

    preds = np.stack(preds)
    gts = np.stack(gts)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    np.save(f"{OUTPUT_DIR}/test_preds.npy", preds)
    np.save(f"{OUTPUT_DIR}/test_gt.npy", gts)
    print(f"Saved predictions to {OUTPUT_DIR}/test_preds.npy and ground truth to {OUTPUT_DIR}/test_gt.npy")
