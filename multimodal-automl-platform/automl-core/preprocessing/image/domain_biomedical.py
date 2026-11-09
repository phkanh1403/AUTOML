import cv2
import numpy as np

class BiomedicalSignalPreprocessor:
    @staticmethod
    def otsu_binarization(image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        _, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary_inv

    @staticmethod
    def morphological_closing(binary_mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        dilated = cv2.dilate(binary_mask, kernel, iterations=1)
        return cv2.erode(dilated, kernel, iterations=1)

    def process_ecg_graph(self, image_bgr: np.ndarray) -> np.ndarray:
        binary = self.otsu_binarization(image_bgr)
        cleaned = self.morphological_closing(binary, kernel_size=3)
        return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
