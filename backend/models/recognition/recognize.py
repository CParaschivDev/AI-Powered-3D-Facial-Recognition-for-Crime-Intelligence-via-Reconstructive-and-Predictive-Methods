import numpy as np
import os
import logging
from insightface.app import FaceAnalysis
from insightface import model_zoo
import torch
import cv2

try:
    # local fallback checkpoint path
    LOCAL_RECOG_CK = os.path.join('logs', 'recognition', 'recognition_model.pth')
    PACKAGE_RECOG_CK = os.path.join('backend', 'models', 'recognition', 'recognition_model.pth')
except Exception:
    LOCAL_RECOG_CK = 'logs/recognition/recognition_model.pth'
    PACKAGE_RECOG_CK = 'backend/models/recognition/recognition_model.pth'

logger = logging.getLogger(__name__)

class RecognitionModel:
    def __init__(self, arcface_path: str | None = None, gcn_path: str | None = None, det_size=(640,640)):
        # Initialize attributes
        self.torch_model = None
        self.fa = None
        self.arc = None
        
        # First, try to load a local PyTorch recognition checkpoint (training artifact)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device for recognition model: {device}")
        
        # Check paths in order: passed arcface_path first, then fallbacks
        checkpoint_paths = []
        if arcface_path:
            checkpoint_paths.append(arcface_path)
        checkpoint_paths.extend([LOCAL_RECOG_CK, PACKAGE_RECOG_CK])
        
        for ck in checkpoint_paths:
            try:
                if os.path.exists(ck):
                    logger.info(f"Loading local recognition checkpoint: {ck}")
                    # Use GPU if available, otherwise CPU
                    map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
                    ck_data = torch.load(ck, map_location=map_location)
                    state = ck_data.get('model_state_dict', ck_data) if isinstance(ck_data, dict) else ck_data

                    # Build RecognitionNet matching the training architecture
                    import torch.nn as nn
                    class _RecNet(nn.Module):
                        def __init__(self, embedding_size=512):
                            super().__init__()
                            self.backbone = nn.Sequential(
                                nn.Conv2d(3, 64, 7, stride=1, padding=3, bias=False),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True)
                            )
                            # Match training: 64 * 112 * 112 = 802816
                            self.embedding_layer = nn.Linear(64 * 112 * 112, embedding_size)

                        def forward(self, x, return_embedding=False):
                            x = self.backbone(x)
                            x = x.view(x.size(0), -1)
                            embedding = self.embedding_layer(x)
                            if return_embedding:
                                return embedding
                            return embedding

                    net = _RecNet(embedding_size=512)
                    try:
                        net.load_state_dict(state, strict=False)
                        net.eval()
                        # Move to appropriate device
                        if torch.cuda.is_available():
                            net = net.cuda()
                        self.torch_model = net
                        logger.info("Loaded PyTorch recognition model.")
                        break
                    except Exception as e:
                        logger.warning(f"Could not load recognition checkpoint into net: {e}")
                        self.torch_model = None
            except Exception as e:
                logger.warning(f"Error while attempting to load local recognition checkpoint {ck}: {e}")
                # Try with CPU if GPU failed
                if "cuda" in str(e).lower() and torch.cuda.is_available():
                    try:
                        logger.info("Retrying with CPU...")
                        ck_data = torch.load(ck, map_location='cpu')
                        state = ck_data.get('model_state_dict', ck_data) if isinstance(ck_data, dict) else ck_data
                        net = _RecNet(embedding_size=512)
                        net.load_state_dict(state, strict=False)
                        net.eval()
                        self.torch_model = net
                        logger.info("Loaded PyTorch recognition model on CPU.")
                        break
                    except Exception as cpu_e:
                        logger.warning(f"CPU loading also failed: {cpu_e}")

        # Only use custom PyTorch model - no InsightFace fallback for main operation
        if self.torch_model is None:
            logger.error("Custom PyTorch recognition model failed to load. Recognition will not work.")
            logger.error("Please ensure the model was trained and saved correctly.")
            # Don't set fallback models - keep them None to indicate failure
            self.fa = None
            self.arc = None

    def embed(self, bgr_img: np.ndarray) -> np.ndarray | None:
        # Only use custom PyTorch model - no InsightFace or dummy fallback
        if getattr(self, 'torch_model', None) is None:
            logger.error("No recognition model available - custom PyTorch model failed to load")
            return None
            
        try:
            img = bgr_img[..., ::-1]  # BGR -> RGB
            img_resized = cv2.resize(img, (112,112))
            img_t = torch.from_numpy(img_resized.astype('float32').transpose(2,0,1) / 255.0).unsqueeze(0)
            
            # Move to same device as model
            if torch.cuda.is_available() and next(self.torch_model.parameters()).is_cuda:
                img_t = img_t.cuda()
            
            with torch.no_grad():
                emb = self.torch_model(img_t, return_embedding=True)
            emb = emb.cpu().numpy().reshape(-1)
            emb = emb / (np.linalg.norm(emb) + 1e-12)
            return emb.astype('float32')
        except Exception as e:
            logger.error(f"Custom recognition model failed to embed image: {e}")
            return None
        
    def extract_fused_embedding(self, bgr_img: np.ndarray) -> np.ndarray | None:
        # For now fused = embed; future: combine arcface + 3D features
        return self.embed(bgr_img)
