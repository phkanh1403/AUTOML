import cv2
import numpy as np
from typing import Tuple

class NaturalDomainPreprocessor:
    def __init__(self, color_space: str = "LAB", clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.color_space = color_space.upper()
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def apply_clahe_illumination(self, image_bgr: np.ndarray) -> np.ndarray:
        if self.color_space == "HSV":
            hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            v_eq = self.clahe.apply(v)
            return cv2.cvtColor(cv2.merge((h, s, v_eq)), cv2.COLOR_HSV2BGR)
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self.clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)

    @staticmethod
    def mixup(img1: np.ndarray, label1: np.ndarray, img2: np.ndarray, label2: np.ndarray, alpha: float = 0.4):
        lam = np.random.beta(alpha, alpha)
        mixed_img = lam * img1.astype(np.float32) + (1.0 - lam) * img2.astype(np.float32)
        mixed_label = lam * label1 + (1.0 - lam) * label2
        return mixed_img.astype(np.uint8), mixed_label

    @staticmethod
    def cutmix(img1: np.ndarray, label1: np.ndarray, img2: np.ndarray, label2: np.ndarray, alpha: float = 1.0):
        lam = np.random.beta(alpha, alpha)
        h, w, _ = img1.shape
        cut_rat = np.sqrt(1.0 - lam)
        cut_w, cut_h = int(w * cut_rat), int(h * cut_rat)
        cx, cy = np.random.randint(w), np.random.randint(h)
        x1, y1 = np.clip(cx - cut_w // 2, 0, w), np.clip(cy - cut_h // 2, 0, h)
        x2, y2 = np.clip(cx + cut_w // 2, 0, w), np.clip(cy + cut_h // 2, 0, h)
        mixed_img = img1.copy()
        mixed_img[y1:y2, x1:x2] = img2[y1:y2, x1:x2]
        lam_adj = 1.0 - ((x2 - x1) * (y2 - y1) / (w * h))
        return mixed_img, lam_adj * label1 + (1.0 - lam_adj) * label2
