from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from .common import classification_metrics, make_response_base, new_job_id, persist_model_artifact


MIN_ROWS = 12
MAX_ROWS = 250_000
MAX_COLUMNS = 2_000
HIGH_MISSING_THRESHOLD = 0.70
RANDOM_STATE = 42


def _read_csv(raw: bytes) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            frame = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding=encoding)
            if frame.empty and len(frame.columns) == 0:
                raise ValueError("The CSV contains no data columns.")
            if len(frame) > MAX_ROWS or len(frame.columns) > MAX_COLUMNS:
                raise ValueError(f"Dataset exceeds the current processing limits ({MAX_ROWS:,} rows, {MAX_COLUMNS:,} columns).")
            return frame
        except Exception as exc:  # try the next likely encoding/delimiter
            last_error = exc
    raise ValueError(f"Unable to read the CSV. Check its encoding and delimiter. Details: {last_error}")


def _prepare_frame(raw: bytes, target_column: str) -> tuple[pd.DataFrame, list[str], list[str], list[str], list[str]]:
    df = _read_csv(raw)
    df.columns = [str(c).strip() for c in df.columns]
    if any(not c for c in df.columns):
        df.columns = [c if c else f"column_{i+1}" for i, c in enumerate(df.columns)]
    if len(set(df.columns)) != len(df.columns):
        raise ValueError("CSV column names are duplicated after trimming whitespace. Rename columns so each name is unique.")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found. Available columns: {', '.join(df.columns[:30])}")

    initial_rows = len(df)
    df = df.replace(r"^\s*$", np.nan, regex=True)
    df = df.replace(["?", "NA", "N/A", "null", "NULL", "None", np.inf, -np.inf], np.nan)
    df = df.drop_duplicates().reset_index(drop=True)
    duplicated_rows = initial_rows - len(df)

    # The target cannot be imputed: rows without a ground-truth label are excluded.
    before_target = len(df)
    df = df.loc[df[target_column].notna()].copy().reset_index(drop=True)
    missing_target_rows = before_target - len(df)
    if len(df) < MIN_ROWS:
        raise ValueError(f"Only {len(df)} labeled rows remain after cleaning; at least {MIN_ROWS} rows are required for a more reliable evaluation.")

    # Convert object columns to numeric only when the overwhelming majority parses.
    # Infinite values are treated as missing before the missing-rate filter.
    for col in df.columns:
        if col != target_column and df[col].dtype == object:
            parsed = pd.to_numeric(df[col], errors="coerce")
            non_missing = int(df[col].notna().sum())
            if non_missing and parsed.notna().sum() / non_missing >= 0.95:
                df[col] = parsed
        if col != target_column and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    dropped_columns: list[str] = []
    # Remove all-empty / excessively-missing and constant predictors only.
    for col in list(df.columns):
        if col == target_column:
            continue
        missing_rate = float(df[col].isna().mean())
        if missing_rate >= HIGH_MISSING_THRESHOLD or df[col].nunique(dropna=True) <= 1:
            dropped_columns.append(col)

    feature_columns = [c for c in df.columns if c != target_column and c not in dropped_columns]
    if not feature_columns:
        raise ValueError("No valid features remain after preprocessing. Provide at least one column with varying values.")

    numeric_features = [c for c in feature_columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    categorical_features = [c for c in feature_columns if c not in numeric_features]
    # Keep categorical columns homogeneous so the imputer/encoder never sees mixed Python types.
    for col in categorical_features:
        df[col] = df[col].map(lambda value: str(value).strip() if pd.notna(value) else np.nan)

    # Remove duplicate predictor columns by value, retaining the first occurrence.
    seen: list[str] = []
    for col in feature_columns:
        duplicate_of = next((prior for prior in seen if df[col].equals(df[prior])), None)
        if duplicate_of is not None:
            dropped_columns.append(col)
        else:
            seen.append(col)
    feature_columns = [c for c in feature_columns if c not in dropped_columns]
    numeric_features = [c for c in numeric_features if c in feature_columns]
    categorical_features = [c for c in categorical_features if c in feature_columns]

    if not feature_columns:
        raise ValueError("All feature columns are empty, constant, or duplicates.")

    cleaning_info = [
        f"Removed {duplicated_rows} duplicate rows",
        f"Removed {missing_target_rows} rows with missing target labels",
        f"Removed {len(dropped_columns)} empty, highly missing (≥{int(HIGH_MISSING_THRESHOLD*100)}%), constant, or duplicate columns",
    ]
    if numeric_features:
        cleaning_info.append("Numeric: median imputation + missing indicator + RobustScaler trong Pipeline")
    if categorical_features:
        cleaning_info.append("Categorical: most-frequent imputation + OneHotEncoder trong Pipeline")
    return df, feature_columns, numeric_features, categorical_features, cleaning_info


def _infer_task(y: pd.Series, task_type: str) -> tuple[str, pd.Series]:
    if task_type not in {"auto", "classification", "regression"}:
        raise ValueError("task_type must be auto, classification, or regression.")
    numeric = pd.to_numeric(y, errors="coerce").replace([np.inf, -np.inf], np.nan)
    numeric_ratio = float(numeric.notna().mean())
    if task_type == "regression":
        if numeric_ratio < 1.0:
            raise ValueError("A regression target must contain only finite numeric values. Clean text or infinite values from the target column.")
        return "regression", numeric.astype(float)
    if task_type == "classification":
        return "classification", y.astype(str).str.strip()

    if numeric_ratio >= 0.98:
        if numeric_ratio < 1.0:
            raise ValueError("The target is mostly numeric but contains non-numeric or infinite values. Clean the target or choose Classification if it represents labels.")
        numeric = numeric.astype(float)
        unique_count = int(numeric.nunique())
        integer_like = bool(np.all(np.isclose(numeric.dropna().to_numpy(), np.round(numeric.dropna().to_numpy()))))
        class_limit = min(30, max(10, int(np.sqrt(len(numeric)) * 1.5)))
        if integer_like and 2 <= unique_count <= class_limit:
            return "classification", numeric.astype(str)
        return "regression", numeric
    return "classification", y.astype(str).str.strip()


def _build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric_features:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", RobustScaler()),
        ])
        transformers.append(("numeric", numeric_pipe, numeric_features))
    if categorical_features:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=True)),
        ])
        transformers.append(("categorical", categorical_pipe, categorical_features))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _candidate_models(task: str):
    if task == "classification":
        return [
            ("Logistic Regression", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
            ("Random Forest Classifier", RandomForestClassifier(n_estimators=250, min_samples_leaf=1, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=1)),
            ("Extra Trees Classifier", ExtraTreesClassifier(n_estimators=300, min_samples_leaf=1, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1)),
        ]
    return [
        ("Ridge Regression", Ridge(alpha=1.0)),
        ("Random Forest Regressor", RandomForestRegressor(n_estimators=250, min_samples_leaf=1, random_state=RANDOM_STATE, n_jobs=1)),
        ("Extra Trees Regressor", ExtraTreesRegressor(n_estimators=300, min_samples_leaf=1, random_state=RANDOM_STATE, n_jobs=1)),
    ]


def _normalize_importance(values: dict[str, float]) -> dict[str, float]:
    weights = {k: max(float(v), 0.0) for k, v in values.items() if np.isfinite(v)}
    total = sum(weights.values())
    if total <= 0:
        return {k: 0.0 for k in values}
    return {k: round(v / total, 6) for k, v in weights.items()}


def train_tabular(raw: bytes, filename: str, target_column: str, task_type: str, storage_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    df, features, numeric_features, categorical_features, steps = _prepare_frame(raw, target_column)
    task, y = _infer_task(df[target_column], task_type)
    X = df[features].copy()

    if task == "classification":
        counts = y.value_counts()
        too_small = counts[counts < 3]
        if len(counts) < 2:
            raise ValueError("A classification target must contain at least two classes.")
        if not too_small.empty:
            detail = ", ".join(f"{label}: {count}" for label, count in too_small.items())
            raise ValueError(f"Each class needs at least 3 samples for the train/validation split and cross-validation. Classes below the minimum: {detail}.")
        n_classes = len(counts)
        test_count = max(int(np.ceil(0.20 * len(df))), n_classes)
        if test_count >= len(df):
            raise ValueError("There is not enough data to create an independent holdout set.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_count, random_state=RANDOM_STATE, stratify=y
        )
        min_class_train = int(y_train.value_counts().min())
        folds = min(3, min_class_train)
        if folds < 2:
            raise ValueError("There are not enough samples per class in the training split for cross-validation. Add more samples to each class.")
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
        selection_scoring = "f1_macro"
    else:
        if len(df) < 12:
            raise ValueError("Regression requires at least 12 labeled rows.")
        test_count = max(2, int(np.ceil(0.20 * len(df))))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_count, random_state=RANDOM_STATE)
        folds = min(3, max(2, len(X_train) // 5))
        if len(X_train) < folds * 2:
            folds = 2
        cv = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
        selection_scoring = "neg_mean_absolute_error"

    scored: list[tuple[float, str, Any, float]] = []
    for name, estimator in _candidate_models(task):
        pipeline = Pipeline([
            ("preprocess", _build_preprocessor(numeric_features, categorical_features)),
            ("model", estimator),
        ])
        try:
            scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=selection_scoring, n_jobs=1, error_score="raise")
            scored.append((float(np.mean(scores)), name, estimator, float(np.std(scores))))
        except Exception as exc:
            # Do not silently select a broken candidate; keep reason visible in metadata.
            continue
    if not scored:
        raise ValueError("All AutoML candidates failed during cross-validation. Check the data types and values in the columns.")
    scored.sort(key=lambda x: x[0], reverse=True)
    cv_score, algorithm, selected_estimator, cv_std = scored[0]

    holdout_model = Pipeline([
        ("preprocess", _build_preprocessor(numeric_features, categorical_features)),
        ("model", selected_estimator),
    ])
    holdout_model.fit(X_train, y_train)
    y_pred = holdout_model.predict(X_test)
    if task == "classification":
        metrics = classification_metrics(y_test, y_pred)
        baseline_estimator = DummyClassifier(strategy="prior")
        baseline_name = "DummyClassifier (prior)"
        xai_scoring = "f1_macro"
        primary_metric = "F1-macro"
        primary_value = metrics["f1_macro"]
        target_labels = sorted(str(v) for v in y.unique())
    else:
        from .common import regression_metrics
        metrics = regression_metrics(y_test, y_pred)
        baseline_estimator = DummyRegressor(strategy="median")
        baseline_name = "DummyRegressor (median)"
        xai_scoring = "neg_mean_absolute_error"
        primary_metric = "MAE"
        primary_value = metrics["mae"]
        target_labels = []
    baseline_pipeline = Pipeline([
        ("preprocess", _build_preprocessor(numeric_features, categorical_features)),
        ("model", baseline_estimator),
    ])
    baseline_pipeline.fit(X_train, y_train)
    baseline_prediction = baseline_pipeline.predict(X_test)
    if task == "classification":
        baseline_metrics = classification_metrics(y_test, baseline_prediction)
        baseline_primary_value = baseline_metrics["f1_macro"]
        improvement_vs_baseline = float(primary_value - baseline_primary_value)
    else:
        baseline_metrics = regression_metrics(y_test, baseline_prediction)
        baseline_primary_value = baseline_metrics["mae"]
        improvement_vs_baseline = float(baseline_primary_value - primary_value)

    # Permutation importance uses the independent holdout and preserves the full preprocessing pipeline.
    try:
        perm = permutation_importance(holdout_model, X_test, y_test, n_repeats=3, random_state=RANDOM_STATE, scoring=xai_scoring, n_jobs=1)
        raw_importance = {feature: float(value) for feature, value in zip(features, perm.importances_mean)}
        importance_map = _normalize_importance(raw_importance)
        importance_map = dict(sorted(importance_map.items(), key=lambda item: item[1], reverse=True)[:12])
    except Exception:
        raw_importance, importance_map = {}, {}

    numeric_for_corr = [c for c in numeric_features if df[c].nunique(dropna=True) > 1][:6]
    corr_headers: list[str] = []
    corr_matrix: list[list[float]] = []
    if numeric_for_corr:
        corr = df[numeric_for_corr].corr(method="spearman").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        corr_headers = numeric_for_corr
        corr_matrix = [[round(float(v), 3) for v in row] for row in corr.to_numpy()]

    # Final serving model is refit on all labeled rows after holdout evaluation.
    final_pipeline = Pipeline([
        ("preprocess", _build_preprocessor(numeric_features, categorical_features)),
        ("model", selected_estimator),
    ])
    final_pipeline.fit(X, y)

    job_id = new_job_id()
    job_dir = storage_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    try:
        # Preserve missing cells in the auditable source-level export. The model pipeline
        # performs inference-time imputation using the same fitted preprocessing steps.
        cleaned_path = job_dir / "processed_dataset.csv"
        temporary_dataset = job_dir / ".processed_dataset.csv.tmp"
        df[features + [target_column]].to_csv(temporary_dataset, index=False, encoding="utf-8-sig")
        temporary_dataset.replace(cleaned_path)
        elapsed = time.perf_counter() - started

        result = make_response_base(job_id, "tabular", algorithm, metrics, elapsed)
        result.update({
            "name": f"Tabular - {algorithm}",
            "sourceFileName": Path(filename).name,
            "targetColumn": target_column,
            "taskType": task,
            "tabularTargetType": "categorical" if task == "classification" else "numerical",
            "tabularTargetLabels": target_labels,
            "features": features,
            "featureTypes": {c: ("numeric" if c in numeric_features else "categorical") for c in features},
            "droppedColumns": sorted(set(df.columns.tolist()) - set(features + [target_column])),
            "sampleCount": int(len(df)),
            "trainingSampleCount": int(len(X_train)),
            "testSampleCount": int(len(X_test)),
            "warnings": (["Small dataset: validation metrics may vary substantially."] if len(df) < 100 else []) + (["Class imbalance detected; also review balanced accuracy and macro F1."] if task == "classification" and y.value_counts(normalize=True).max() > 0.80 else []) + (["The AutoML model did not outperform the simple holdout baseline. Consider adding data or features, or reviewing the labels."] if improvement_vs_baseline <= 0 else []),
            "download": {"dataset": f"/api/jobs/{job_id}/dataset", "model": f"/api/jobs/{job_id}/model"},
            "outputs": {
                "output_1_dataset": {
                    "status_msg": f"Cleaned {len(df)} labeled rows and retained {len(features)} features. The exported CSV preserves blank cells for auditability; imputers are applied inside the model pipeline.",
                    "preprocessing_applied": "; ".join(steps),
                    "artifact_path": cleaned_path.name,
                },
                "output_2_model": {"algorithm": algorithm, "format": "joblib", "artifact_path": "model.joblib"},
                "output_3_history": {
                    "metrics": metrics,
                    "primary_metric": primary_metric,
                    "primary_value": float(primary_value),
                    "cv_score": float(-cv_score if task == "regression" else cv_score),
                    "cv_metric": "MAE" if task == "regression" else "F1-macro",
                    "cv_std": float(cv_std),
                    "cv_folds": int(folds),
                    "baseline_algorithm": baseline_name,
                    "baseline_metric": primary_metric,
                    "baseline_value": float(baseline_primary_value),
                    "improvement_vs_baseline": float(improvement_vs_baseline),
                    "baseline_metrics": baseline_metrics,
                    "candidate_scores": [{"algorithm": n, "cv_score": float(-score if task == "regression" else score), "cv_std": float(std)} for score, n, _, std in scored],
                    "execution_time_sec": round(float(elapsed), 3),
                },
                "output_4_xai": {
                    "method": "Permutation importance on the holdout set + Spearman correlation (descriptive only)",
                    "top_contributors": "Features with the largest impact on holdout metrics; correlation is descriptive and is not used to remove features.",
                    "importanceMap": importance_map,
                    "importanceRaw": raw_importance,
                    "correlation_matrix": corr_matrix,
                    "correlation_headers": corr_headers,
                    "final_features": features,
                },
            },
        })
        bundle = {
            "modality": "tabular",
            "pipeline": final_pipeline,
            "feature_names": features,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "task": task,
        }
        persist_model_artifact(job_dir, result, bundle)
        return result
    except Exception:
        import shutil
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
