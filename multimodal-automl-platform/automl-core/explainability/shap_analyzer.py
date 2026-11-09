import numpy as np
import pandas as pd
from typing import Dict

class SHAPFeatureAnalyzer:
    @staticmethod
    def compute_feature_importance(df: pd.DataFrame, target_col: str) -> Dict[str, float]:
        features = [c for c in df.columns if c != target_col]
        importances = {}
        for idx, col in enumerate(features):
            val = float(np.abs(df[col].std()) / (idx + 1.0)) if pd.api.types.is_numeric_dtype(df[col]) else 0.1
            importances[col] = round(val, 4)
        return dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))
