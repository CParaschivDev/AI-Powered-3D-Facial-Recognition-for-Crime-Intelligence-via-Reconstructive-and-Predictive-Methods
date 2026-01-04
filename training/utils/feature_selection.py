"""
Feature Selection and Dimensionality Reduction Utilities
GEN-4 AI Copilot enhancement for facial recognition training.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel
)
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class FeatureSelector:
    """
    Comprehensive feature selection and dimensionality reduction toolkit
    for facial recognition and landmark detection models.
    """

    def __init__(self, method='auto', n_features=None, task_type='classification'):
        """
        Args:
            method: Feature selection method ('pca', 'lda', 'lasso', 'rfe', 'mutual_info', 'auto')
            n_features: Number of features to select (None = auto)
            task_type: 'classification' or 'regression'
        """
        self.method = method
        self.n_features = n_features
        self.task_type = task_type
        self.selector = None
        self.feature_importance = None
        self.explained_variance = None

    def fit(self, X, y=None):
        """
        Fit feature selector on training data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (for supervised methods)
        """
        X = np.array(X)
        n_samples, n_features = X.shape

        if self.n_features is None:
            self.n_features = min(n_features // 2, 512)  # Auto-select reasonable number

        print(f"Applying {self.method} feature selection (n_features: {self.n_features})")

        if self.method == 'pca':
            self.selector = PCA(n_components=self.n_features, random_state=42)
            X_transformed = self.selector.fit_transform(X)
            self.explained_variance = self.selector.explained_variance_ratio_
            print(f"PCA explained variance: {self.explained_variance.sum():.3f}")

        elif self.method == 'lda':
            if y is None:
                raise ValueError("LDA requires target labels")
            self.selector = LinearDiscriminantAnalysis(n_components=min(self.n_features, len(np.unique(y))-1))
            X_transformed = self.selector.fit_transform(X, y)

        elif self.method == 'lasso':
            if y is None:
                raise ValueError("LASSO requires target labels")
            if self.task_type == 'classification':
                self.selector = SelectFromModel(LassoCV(cv=5, random_state=42))
            else:
                self.selector = SelectFromModel(RidgeCV())
            self.selector.fit(X, y)
            X_transformed = self.selector.transform(X)

        elif self.method == 'rfe':
            if y is None:
                raise ValueError("RFE requires target labels")
            estimator = RandomForestClassifier(n_estimators=100, random_state=42) if self.task_type == 'classification' else None
            if estimator is None:
                raise ValueError("RFE for regression not implemented")
            self.selector = RFE(estimator=estimator, n_features_to_select=self.n_features, step=10)
            self.selector.fit(X, y)
            X_transformed = self.selector.transform(X)

        elif self.method == 'mutual_info':
            if y is None:
                raise ValueError("Mutual information requires target labels")
            score_func = mutual_info_classif if self.task_type == 'classification' else None
            if score_func is None:
                raise ValueError("Mutual information for regression not implemented")
            self.selector = SelectKBest(score_func=score_func, k=self.n_features)
            X_transformed = self.selector.fit_transform(X, y)
            self.feature_importance = self.selector.scores_

        elif self.method == 'auto':
            # Auto-select best method based on data characteristics
            if y is not None and len(np.unique(y)) > 2:
                # Multi-class: try LDA first, fallback to PCA
                try:
                    self.selector = LinearDiscriminantAnalysis(n_components=min(self.n_features, len(np.unique(y))-1))
                    X_transformed = self.selector.fit_transform(X, y)
                    print("Auto-selected: LDA")
                except:
                    self.selector = PCA(n_components=self.n_features, random_state=42)
                    X_transformed = self.selector.fit_transform(X)
                    self.explained_variance = self.selector.explained_variance_ratio_
                    print("Auto-selected: PCA (LDA failed)")
            else:
                # Default to PCA for unsupervised or binary classification
                self.selector = PCA(n_components=self.n_features, random_state=42)
                X_transformed = self.selector.fit_transform(X)
                self.explained_variance = self.selector.explained_variance_ratio_
                print("Auto-selected: PCA")

        else:
            raise ValueError(f"Unknown method: {self.method}")

        print(f"Feature selection complete. Original: {n_features}, Selected: {X_transformed.shape[1]}")
        return X_transformed

    def transform(self, X):
        """Transform new data using fitted selector."""
        if self.selector is None:
            raise ValueError("Feature selector not fitted")
        X = np.array(X)
        return self.selector.transform(X)

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y)

    def get_feature_importance(self):
        """Get feature importance scores if available."""
        return self.feature_importance

    def plot_feature_importance(self, save_path=None):
        """Plot feature importance if available."""
        if self.feature_importance is None:
            print("No feature importance available for this method")
            return

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(self.feature_importance)), self.feature_importance)
        plt.xlabel('Feature Index')
        plt.ylabel('Importance Score')
        plt.title(f'Feature Importance ({self.method})')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Feature importance plot saved to {save_path}")
        else:
            plt.show()

    def plot_explained_variance(self, save_path=None):
        """Plot explained variance for PCA."""
        if self.explained_variance is None:
            print("No explained variance available (not PCA)")
            return

        plt.figure(figsize=(12, 6))

        # Cumulative explained variance
        plt.subplot(1, 2, 1)
        plt.plot(np.cumsum(self.explained_variance), 'bo-')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Cumulative Explained Variance')
        plt.grid(True, alpha=0.3)

        # Individual explained variance
        plt.subplot(1, 2, 2)
        plt.bar(range(len(self.explained_variance)), self.explained_variance)
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('PCA Individual Explained Variance')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Explained variance plot saved to {save_path}")
        else:
            plt.show()


class FeatureExtractor(nn.Module):
    """
    Neural feature extractor that can be integrated into training pipelines
    with built-in dimensionality reduction.
    """

    def __init__(self, input_dim, hidden_dims=[1024, 512, 256], output_dim=128,
                 dropout_rate=0.3, use_batch_norm=True):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dims: Hidden layer dimensions
            output_dim: Final output dimension
            dropout_rate: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super(FeatureExtractor, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        # Final output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.feature_extractor = nn.Sequential(*layers)

    def forward(self, x):
        return self.feature_extractor(x)

    def get_feature_dim(self):
        """Get the output feature dimension."""
        return self.feature_extractor[-1].out_features


def extract_features_from_model(model, dataloader, device, layer_name='embedding_layer'):
    """
    Extract features from a specific layer of a trained model.

    Args:
        model: Trained PyTorch model
        dataloader: DataLoader with input data
        device: torch device
        layer_name: Name of the layer to extract features from

    Returns:
        features: Extracted features (n_samples, feature_dim)
        labels: Corresponding labels if available
    """
    model.eval()

    features = []
    labels = []

    # Hook to capture intermediate features
    feature_hook = None
    def hook_fn(module, input, output):
        feature_hook.append(output.detach().cpu())

    # Register hook on specified layer
    for name, module in model.named_modules():
        if name == layer_name:
            hook_handle = module.register_forward_hook(hook_fn)
            break
    else:
        raise ValueError(f"Layer '{layer_name}' not found in model")

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            feature_hook = []  # Reset for each batch

            _ = model(inputs)

            batch_features = torch.cat(feature_hook, dim=0)
            features.append(batch_features)

            if targets is not None:
                labels.extend(targets.cpu().numpy())

    hook_handle.remove()

    features = torch.cat(features, dim=0).numpy()
    labels = np.array(labels) if labels else None

    print(f"Extracted features: {features.shape}")
    return features, labels


def apply_feature_selection_to_dataset(dataset, feature_selector, batch_size=128):
    """
    Apply feature selection to a dataset by extracting and transforming features.

    Args:
        dataset: PyTorch dataset
        feature_selector: Fitted FeatureSelector instance
        batch_size: Batch size for processing

    Returns:
        transformed_dataset: Dataset with selected features
    """
    from torch.utils.data import TensorDataset

    # Extract all features and labels
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_features = []
    all_labels = []

    for features, labels in dataloader:
        all_features.append(features)
        all_labels.append(labels)

    X = torch.cat(all_features, dim=0).numpy()
    y = torch.cat(all_labels, dim=0).numpy()

    # Apply feature selection
    X_transformed = feature_selector.transform(X)

    # Create new dataset with transformed features
    X_tensor = torch.tensor(X_transformed, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    transformed_dataset = TensorDataset(X_tensor, y_tensor)
    print(f"Dataset transformed: {X.shape} -> {X_transformed.shape}")

    return transformed_dataset


def create_feature_selection_pipeline(model_type='recognition', n_features=None):
    """
    Create a feature selection pipeline optimized for different model types.

    Args:
        model_type: 'recognition', 'landmarks', or 'reconstruction'
        n_features: Number of features to select

    Returns:
        FeatureSelector: Configured feature selector
    """
    if model_type == 'recognition':
        # For recognition, use PCA or LDA depending on available labels
        return FeatureSelector(method='auto', n_features=n_features, task_type='classification')

    elif model_type == 'landmarks':
        # For landmarks, use mutual information or RFE
        return FeatureSelector(method='mutual_info', n_features=n_features, task_type='regression')

    elif model_type == 'reconstruction':
        # For reconstruction, use PCA for dimensionality reduction
        return FeatureSelector(method='pca', n_features=n_features, task_type='regression')

    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Utility functions for integration with training scripts
def save_feature_selector(selector, filepath):
    """Save fitted feature selector to disk."""
    import pickle
    with open(filepath, 'wb') as f:
        pickle.dump(selector, f)
    print(f"Feature selector saved to {filepath}")

def load_feature_selector(filepath):
    """Load fitted feature selector from disk."""
    import pickle
    with open(filepath, 'rb') as f:
        selector = pickle.load(f)
    print(f"Feature selector loaded from {filepath}")
    return selector