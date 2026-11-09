import pandas as pd
from typing import Tuple, List
from .imputer import TabularImputer
from .outlier_filter import TabularOutlierFilter
from .scaler import TabularFeatureScaler

class TabularColumnRouter:
    def __init__(self, target_col: str, nan_threshold: float = 0.5, outlier_method: str = "winsorize", scale_method: str = "robust"):
        self.target_col = target_col
        self.nan_threshold = nan_threshold
        self.outlier_method = outlier_method
        self.scale_method = scale_method
        self.dropped_columns: List[str] = []

    def process(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        df = TabularImputer.validate_and_drop_rows(df, self.target_col)
        feature_cols = [c for c in df.columns if c != self.target_col]
        self.dropped_columns = [c for c in feature_cols if df[c].isna().mean() >= self.nan_threshold]
        df_clean = df.drop(columns=self.dropped_columns)

        numeric_cols = [c for c in df_clean.select_dtypes(include=['number']).columns if c != self.target_col]
        df_clean = TabularImputer.distribution_aware_impute(df_clean, numeric_cols)

        if self.outlier_method == "iqr":
            df_clean = TabularOutlierFilter.filter_iqr(df_clean, numeric_cols)
        elif self.outlier_method == "zscore":
            df_clean = TabularOutlierFilter.filter_zscore(df_clean, numeric_cols)
        else:
            df_clean = TabularOutlierFilter.winsorize_clip(df_clean, numeric_cols, upper_pct=0.95)

        scaler = TabularFeatureScaler(method=self.scale_method)
        df_clean = scaler.fit_transform(df_clean, numeric_cols)
        return df_clean, self.dropped_columns
