import cv2
import numpy as np

class CCTVAugmentation:
    """CCTV-specific augmentations for robustness testing."""
    
    def __init__(self, probability=1.0, blur_kernel=0, noise_std=0.0, compression_quality=100, occlusion_prob=0.0):
        self.probability = probability
        self.blur_kernel = blur_kernel
        self.noise_std = noise_std
        self.compression_quality = compression_quality
        self.occlusion_prob = occlusion_prob
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply CCTV augmentations to an image with given probability."""
        if np.random.random() < self.probability:
            return self.apply_augmentation(image)
        return image
    
    def apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Apply CCTV augmentations to an image."""
        augmented = image.copy()
        
        # Apply blur
        if self.blur_kernel > 0:
            kernel_size = (self.blur_kernel * 2 + 1, self.blur_kernel * 2 + 1)
            augmented = cv2.GaussianBlur(augmented, kernel_size, 0)
        
        # Apply noise
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std * 255, augmented.shape).astype(np.uint8)
            augmented = cv2.add(augmented, noise)
        
        # Apply compression
        if self.compression_quality < 100:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.compression_quality]
            _, encoded_img = cv2.imencode('.jpg', augmented, encode_param)
            augmented = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        
        # Apply random occlusion
        if self.occlusion_prob > 0 and np.random.random() < self.occlusion_prob:
            h, w = augmented.shape[:2]
            # Random rectangle occlusion
            x1 = np.random.randint(0, w // 2)
            y1 = np.random.randint(0, h // 2)
            x2 = np.random.randint(x1 + w // 4, w)
            y2 = np.random.randint(y1 + h // 4, h)
            cv2.rectangle(augmented, (x1, y1), (x2, y2), (0, 0, 0), -1)
        
        return augmented

def apply_gaussian_blur(image: np.ndarray, kernel_size: tuple = (5, 5)) -> np.ndarray:
    """Apply Gaussian blur to an image."""
    return cv2.GaussianBlur(image, kernel_size, 0)

def add_gaussian_noise(image: np.ndarray, sigma: float = 25) -> np.ndarray:
    """Add Gaussian noise to an image."""
    noise = np.random.normal(0, sigma, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

def assess_image_quality(image: np.ndarray) -> float:
    """
    Assess image quality based on blur, occlusion, and illumination.
    Returns a score between 0 (poor) and 1 (excellent).
    """
    blur_score = assess_blur(image)
    occlusion_score = assess_occlusion(image)
    illumination_score = assess_illumination(image)

    # Combine the scores (you can adjust the weights as needed)
    quality_score = (0.4 * blur_score + 0.3 * occlusion_score + 0.3 * illumination_score)
    return max(0.0, min(1.0, quality_score))  # Ensure the score is within [0, 1]

def assess_blur(image: np.ndarray) -> float:
    """Assess blurriness using variance of Laplacian."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalize the variance to a 0-1 scale (adjust max_variance based on your data)
    max_variance = 150  # Adjust this based on your dataset
    return min(1.0, laplacian_variance / max_variance)

def assess_occlusion(image: np.ndarray) -> float:
    """Assess occlusion by detecting black regions."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    black_pixels = np.sum(gray < 20)  # Threshold for "black"
    total_pixels = gray.size
    occlusion_ratio = black_pixels / total_pixels
    return 1.0 - min(1.0, occlusion_ratio * 3)  # Higher ratio = lower score

def assess_illumination(image: np.ndarray) -> float:
    """Assess illumination by checking the average intensity in the HSV color space."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Ideal intensity is not too dark or too bright. We score based on distance from middle gray.
    mean_intensity = np.mean(hsv[:, :, 2])
    score = 1.0 - abs(mean_intensity - 128) / 128.0
    return max(0.0, score)