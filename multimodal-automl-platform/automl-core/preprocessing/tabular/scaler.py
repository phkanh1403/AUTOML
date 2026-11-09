import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler

class TabularFeatureScaler:
    def __init__(self, method: str = "standard"):
        self.scaler = RobustScaler() if method.lower() == "robust" else StandardScaler()

    def fit_transform(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        df_out = df.copy()
        if feature_cols:
            df_out[feature_cols] = self.scaler.fit_transform(df_out[feature_cols])
        return df_out
