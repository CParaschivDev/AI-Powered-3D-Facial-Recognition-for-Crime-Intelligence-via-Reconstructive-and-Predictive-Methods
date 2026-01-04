import cv2
import numpy as np

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    """Applies Gaussian blur to an image."""
    return cv2.GaussianBlur(image, kernel_size, 0)

def apply_motion_blur(image, kernel_size=15):
    """Applies motion blur to an image."""
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    return cv2.filter2D(image, -1, kernel)

def add_gaussian_noise(image, mean=0, sigma=25):
    """Adds Gaussian noise to an image."""
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def change_brightness(image, value=30):
    """Changes the brightness of an image. Use negative value to darken."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    v[v < 0] = 0
    
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def apply_jpeg_compression(image, quality=50):
    """Simulates JPEG compression artifacts."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', image, encode_param)
    return cv2.imdecode(encimg, 1)

def add_occlusion(image, size_fraction=0.2):
    """Adds a black rectangle to occlude part of the image."""
    h, w, _ = image.shape
    occ_h, occ_w = int(h * size_fraction), int(w * size_fraction)
    
    x1 = np.random.randint(0, w - occ_w)
    y1 = np.random.randint(0, h - occ_h)
    
    occluded_image = image.copy()
    cv2.rectangle(occluded_image, (x1, y1), (x1 + occ_w, y1 + occ_h), (0, 0, 0), -1)
    return occluded_image


CCTV_AUGMENTATIONS = {
    "gaussian_blur": apply_gaussian_blur,
    "motion_blur": apply_motion_blur,
    "gaussian_noise": add_gaussian_noise,
    "low_light": lambda img: change_brightness(img, value=-50),
    "high_exposure": lambda img: change_brightness(img, value=50),
    "jpeg_compression": apply_jpeg_compression,
    "occlusion": add_occlusion,
}
