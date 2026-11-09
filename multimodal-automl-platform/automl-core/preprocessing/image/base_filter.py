import cv2
import numpy as np

class BaseImageFilter:
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, blur_threshold: float = 100.0):
        self.blur_threshold = blur_threshold

    def compute_laplacian_variance(self, image_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        mu = np.mean(laplacian)
        return float(np.mean((laplacian - mu) ** 2))

    def is_sharp(self, image_bgr: np.ndarray) -> bool:
        return self.compute_laplacian_variance(image_bgr) >= self.blur_threshold

    def global_zscore_normalize(self, image_bgr: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return (image_rgb - self.IMAGENET_MEAN) / self.IMAGENET_STD
