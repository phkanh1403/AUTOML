import cv2
import numpy as np
import os

class GradCAMVisualizer:
    @staticmethod
    def generate_heatmap(image_bgr: np.ndarray, output_path: str) -> str:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image_bgr, 0.55, heatmap, 0.45, 0)
        cv2.imwrite(output_path, overlay)
        return output_path
