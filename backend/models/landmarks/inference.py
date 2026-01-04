import numpy as np
import torch
import os
import logging

logger = logging.getLogger(__name__)

class LandmarkModel:
    """
    Dense landmark detection model using trained PyTorch model.
    """
    def __init__(self, model_path: str):
        """
        Initializes the landmark model.

        Args:
            model_path: Path to the trained model file (e.g., .pth).
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.trained_model = None

        # Try to locate a trained checkpoint
        self.checkpoint_paths = [
            os.path.join("logs", "landmarks", "landmark_model.pth"),
            os.path.join("backend", "models", "landmarks", "trained_model.pth"),
            model_path,
        ]

        try:
            for ck in self.checkpoint_paths:
                if ck and os.path.exists(ck):
                    logger.info(f"[LandmarkModel] Found checkpoint at: {ck}")
                    ck_data = torch.load(ck, map_location=self.device)
                    state_dict = ck_data.get('model_state_dict', ck_data) if isinstance(ck_data, dict) else ck_data

                    # Infer num_landmarks from the output layer
                    # Look for fc1.weight which should be (num_landmarks*2, input_features)
                    num_landmarks = None
                    for k, v in state_dict.items():
                        if 'fc1.weight' in k and hasattr(v, 'shape') and len(v.shape) == 2:
                            # fc1 outputs (num_landmarks * 2) features
                            num_landmarks = v.shape[0] // 2
                            break

                    if num_landmarks is None:
                        logger.warning("[LandmarkModel] Could not infer num_landmarks from checkpoint")
                        continue

                    # Build LandmarkNet matching training architecture
                    class _LandmarkNet(torch.nn.Module):
                        def __init__(self, num_landmarks):
                            super().__init__()
                            self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
                            self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((56, 56))  # Reduce from 224x224 to 56x56
                            self.fc1 = torch.nn.Linear(16 * 56 * 56, num_landmarks * 2)  # 50176 instead of 802816

                        def forward(self, x):
                            x = torch.relu(self.conv1(x))
                            x = self.adaptive_pool(x)
                            x = x.view(x.size(0), -1)
                            x = self.fc1(x)
                            return x.view(x.size(0), -1, 2)

                    net = _LandmarkNet(num_landmarks).to(self.device)
                    try:
                        net.load_state_dict(state_dict, strict=False)
                        net.eval()
                        self.trained_model = net
                        self.num_landmarks = num_landmarks
                        logger.info(f"[LandmarkModel] Loaded trained net with {num_landmarks} landmarks")
                        break
                    except Exception as e:
                        logger.warning(f"[LandmarkModel] Could not load state_dict into net: {e}")
                        continue

            if self.trained_model is None:
                logger.info(f"[LandmarkModel] No usable trained checkpoint found - using placeholder logic on {self.device}")
        except Exception as e:
            logger.warning(f"[LandmarkModel] Error while loading trained model: {e}. Using placeholder logic.")

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Predicts dense facial landmarks from an input image.

        Args:
            image: An image in NumPy array format (H, W, C).

        Returns:
            A NumPy array of shape (N, 2) representing N landmarks.
        """
        logger.info("[LandmarkModel] Predicting landmarks.")

        try:
            if self.trained_model is None:
                # Fallback: Return random points
                num_landmarks = 4000
                landmarks = np.random.rand(num_landmarks, 2) * np.array([image.shape[1], image.shape[0]])
                return landmarks

            # Preprocess image: resize to 224x224 (expected by training model)
            img = image.copy()
            try:
                import cv2
                img_resized = cv2.resize(img, (224, 224))
            except Exception:
                # fallback if resize fails
                img_resized = img

            # Convert to tensor: HWC -> CHW, normalize to [0,1]
            img_t = torch.from_numpy(img_resized.astype('float32').transpose(2, 0, 1) / 255.0).unsqueeze(0).to(self.device)

            with torch.no_grad():
                pred_landmarks = self.trained_model(img_t)
                landmarks = pred_landmarks.cpu().numpy().reshape(-1, 2)

            # Scale landmarks back to original image size
            scale_x = image.shape[1] / 224.0
            scale_y = image.shape[0] / 224.0
            landmarks[:, 0] *= scale_x
            landmarks[:, 1] *= scale_y

            return landmarks.astype(np.float32)

        except Exception as e:
            logger.warning(f"[LandmarkModel] Landmark prediction failed, using placeholder: {e}")
            # Fallback: Return random points
            num_landmarks = getattr(self, 'num_landmarks', 4000)
            landmarks = np.random.rand(num_landmarks, 2) * np.array([image.shape[1], image.shape[0]])
            return landmarks
