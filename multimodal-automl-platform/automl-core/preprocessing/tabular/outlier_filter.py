import pandas as pd

class TabularOutlierFilter:
    @staticmethod
    def filter_zscore(df: pd.DataFrame, cols: list, threshold: float = 3.0) -> pd.DataFrame:
        df_out = df.copy()
        for col in cols:
            std = df_out[col].std()
            if std > 0:
                z = (df_out[col] - df_out[col].mean()) / std
                df_out = df_out[z.abs() <= threshold]
        return df_out

    @staticmethod
    def filter_iqr(df: pd.DataFrame, cols: list) -> pd.DataFrame:
        df_out = df.copy()
        for col in cols:
            q1, q3 = df_out[col].quantile(0.25), df_out[col].quantile(0.75)
            iqr = q3 - q1
            df_out = df_out[(df_out[col] >= q1 - 1.5 * iqr) & (df_out[col] <= q3 + 1.5 * iqr)]
        return df_out

    @staticmethod
    def winsorize_clip(df: pd.DataFrame, cols: list, upper_pct: float = 0.99) -> pd.DataFrame:
        df_out = df.copy()
        lower_pct = 1.0 - upper_pct
        for col in cols:
            df_out[col] = df_out[col].clip(lower=df_out[col].quantile(lower_pct), upper=df_out[col].quantile(upper_pct))
        return df_out
