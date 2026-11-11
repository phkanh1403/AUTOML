from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np


def new_job_id() -> str:
    return f"job_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"


def safe_job_id(value: str) -> str:
    if not re.fullmatch(r"job_[A-Za-z0-9_-]{8,64}", value or ""):
        raise ValueError("Invalid model ID.")
    return value


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        val = float(value)
        return val if np.isfinite(val) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, data: Any) -> None:
    """Atomically write JSON so incomplete metadata is never published."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = json.dumps(json_safe(data), ensure_ascii=False, indent=2)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def persist_model_artifact(job_dir: Path, result: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    """Persist a trained model as a durable, downloadable joblib artifact.

    metadata.json is written last and acts as the registry commit marker. A job is
    listed only after both its model artifact and metadata have been fully written.
    """
    job_dir.mkdir(parents=True, exist_ok=True)
    model_path = job_dir / "model.joblib"
    temporary = job_dir / f".model.{uuid.uuid4().hex}.tmp"
    try:
        joblib.dump(bundle, temporary, compress=3)
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, model_path)

        digest = hashlib.sha256()
        with model_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        artifact = {
            "file_name": f"{result['jobId']}_model.joblib",
            "format": "joblib",
            "relative_path": f"data/models/{result['jobId']}/model.joblib",
            "size_bytes": model_path.stat().st_size,
            "sha256": digest.hexdigest(),
            "download_url": f"/api/jobs/{result['jobId']}/model",
        }
        result["modelArtifact"] = artifact
        result.setdefault("outputs", {}).setdefault("output_2_model", {}).update({
            "format": "joblib",
            "artifact_path": "model.joblib",
            "artifact_size_bytes": artifact["size_bytes"],
            "artifact_sha256": artifact["sha256"],
        })
        result.setdefault("download", {})["model"] = artifact["download_url"]
        write_json(job_dir / "metadata.json", result)
        return artifact
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def classification_metrics(y_true, y_pred) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

    labels = sorted(str(value) for value in np.unique(np.asarray(y_true, dtype=str)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        },
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


def regression_metrics(y_true, y_pred) -> dict[str, Any]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else None,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def make_response_base(job_id: str, modality: str, algorithm: str, metrics: dict[str, Any], elapsed: float) -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "jobId": job_id,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "modality": modality,
        "outputs": {
            "output_1_dataset": {},
            "output_2_model": {"algorithm": algorithm, "format": "joblib"},
            "output_3_history": {"metrics": metrics, "execution_time_sec": round(float(elapsed), 3)},
            "output_4_xai": {},
        },
    }
