import pandas as pd

class TabularImputer:
    @staticmethod
    def validate_and_drop_rows(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        if df[target_col].isna().any():
            raise ValueError(f"Tu choi du lieu: Phat hien ban ghi khuyet nhan muc tieu '{target_col}'!")
        return df.copy()

    @staticmethod
    def distribution_aware_impute(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
        df_out = df.copy()
        for col in numeric_cols:
            if df_out[col].isna().sum() == 0:
                continue
            skewness = df_out[col].skew()
            fill_val = df_out[col].mean() if abs(skewness) < 0.5 else df_out[col].median()
            df_out[col] = df_out[col].fillna(fill_val)
        return df_out

    @staticmethod
    def group_based_impute(df: pd.DataFrame, group_col: str, target_col: str, agg_func: str = "mean") -> pd.DataFrame:
        df_out = df.copy()
        if group_col in df_out.columns and target_col in df_out.columns:
            df_out[target_col] = df_out.groupby(group_col)[target_col].transform(
                lambda x: x.fillna(getattr(x, agg_func)())
            )
        return df_out
