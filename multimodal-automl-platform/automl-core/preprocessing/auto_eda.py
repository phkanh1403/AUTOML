import pandas as pd
import numpy as np
from typing import Dict, Any

class AutoEDAEngine:
    @staticmethod
    def analyze_tabular(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        missing_ratios = df.isna().mean().to_dict()
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr().round(4).to_dict() if not numeric_df.empty else {}
        return {
            "modality": "tabular",
            "row_count": len(df),
            "is_clean": all(r == 0.0 for r in missing_ratios.values()),
            "missing_ratios": missing_ratios,
            "dropped_50pct_cols": [col for col, r in missing_ratios.items() if r >= 0.5 and col != target_col],
            "correlation_matrix": corr_matrix
        }
