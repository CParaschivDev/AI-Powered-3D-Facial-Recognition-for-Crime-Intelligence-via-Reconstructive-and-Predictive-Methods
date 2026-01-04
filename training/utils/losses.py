import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class ArcFaceEmbedding(nn.Module):
    """
    ArcFace/ResNet50 embedding extractor for identity preservation.
    """
    def __init__(self, embedding_dim=512, pretrained=True):
        super(ArcFaceEmbedding, self).__init__()
        # Load ResNet50 backbone
        if pretrained:
            self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.backbone = models.resnet50(weights=None)

        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Add embedding layer
        self.embedding = nn.Linear(2048, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)

        # ArcFace margin parameters
        self.margin = 0.5
        self.scale = 64.0

    def forward(self, x, labels=None, return_embedding=True):
        """
        Forward pass.

        Args:
            x: Input images
            labels: Class labels for ArcFace training (optional)
            return_embedding: Whether to return embeddings or logits

        Returns:
            Embeddings or classification logits
        """
        # Extract features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)

        # Generate embeddings
        embeddings = self.embedding(features)
        embeddings = self.bn(embeddings)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        if return_embedding or labels is None:
            return embeddings

        # ArcFace loss computation
        if labels is not None:
            # CosFace/ArcFace margin
            cos_theta = F.linear(embeddings, self.embedding.weight.t())
            cos_theta = cos_theta.clamp(-1, 1)

            # Apply margin
            theta = torch.acos(cos_theta)
            cos_theta_m = torch.cos(theta + self.margin)

            # Scale
            logits = self.scale * cos_theta_m
            return logits

        return embeddings

    def extract_embeddings(self, images):
        """Extract normalized embeddings from images."""
        self.eval()
        with torch.no_grad():
            return self.forward(images, return_embedding=True)

class IdentityPreservingLoss(nn.Module):
    """
    A loss function that encourages a reconstructed face to have the same
    identity features as the input face. It uses a pre-trained face
    recognition model as a feature extractor.
    """
    def __init__(self, feature_extractor='arcface', embedding_dim=512):
        super(IdentityPreservingLoss, self).__init__()
        if feature_extractor == 'arcface':
            self.feature_extractor = ArcFaceEmbedding(embedding_dim=embedding_dim, pretrained=True)
            print("✓ [IdentityPreservingLoss] Initialized with real ArcFace/ResNet50 embedding extractor.")
        else:
            raise NotImplementedError("Only 'arcface' feature extractor is supported.")

        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, rendered_face, ground_truth_face):
        """
        Args:
            rendered_face: A batch of rendered faces from the reconstructed mesh.
            ground_truth_face: A batch of the original input faces.
        """
        # Extract embeddings
        emb_rendered = self.feature_extractor.extract_embeddings(rendered_face)
        emb_gt = self.feature_extractor.extract_embeddings(ground_truth_face)

        # Cosine similarity loss (higher similarity = lower loss)
        cosine_sim = F.cosine_similarity(emb_rendered, emb_gt, dim=1)
        loss = 1.0 - cosine_sim.mean()

        return loss


class TripletLoss(nn.Module):
    """
    Triplet loss takes an anchor, a positive (same class as anchor) and a
    negative (different class from anchor) sample. It aims to push the
    anchor-positive distance to be smaller than the anchor-negative distance
    by a certain margin.
    """
    def __init__(self, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive, 2)
        distance_negative = F.pairwise_distance(anchor, negative, 2)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


class TextureLoss(nn.Module):
    """Placeholder for a texture loss, e.g., perceptual loss using VGG."""
    def __init__(self):
        super(TextureLoss, self).__init__()
        print("INFO: [TextureLoss] Initialized (placeholder).")

    def forward(self, pred_texture, gt_texture):
        return F.l1_loss(pred_texture, gt_texture)
