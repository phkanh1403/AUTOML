import cv2
import numpy as np
from typing import Tuple

class MotionDomainPreprocessor:
    PADDING_COLOR_RGB = (114, 114, 114)

    @staticmethod
    def unsharp_masking(image_bgr: np.ndarray, kernel_size: Tuple[int, int] = (5, 5), sigma: float = 1.0) -> np.ndarray:
        orig_float = image_bgr.astype(np.float32)
        blurred = cv2.GaussianBlur(orig_float, kernel_size, sigma)
        sharpened = orig_float + (orig_float - blurred)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def letterbox_resize(self, image_bgr: np.ndarray, target_size: int = 224) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        scale = target_size / max(h, w)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_w, pad_h = target_size - new_w, target_size - new_h
        top, bottom = pad_h // 2, pad_h - (pad_h // 2)
        left, right = pad_w // 2, pad_w - (pad_w // 2)
        return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.PADDING_COLOR_RGB)
