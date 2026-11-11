from __future__ import annotations

import hashlib
import json
import os
import re
import time
import unicodedata
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


def _slug_model_label(value: Any, fallback: str) -> str:
    """Convert a class/target label into a safe lowercase filename token."""
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
    token = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
    return token or fallback


def _model_name_prefix(result: dict[str, Any]) -> str:
    """Build the requested <label>-<modality> part of the model name."""
    modality = str(result.get("modality") or "model").strip().lower()

    if modality == "image":
        labels = result.get("labels") or result.get("classes") or []
        label_tokens = [
            _slug_model_label(label, f"class-{index + 1}")
            for index, label in enumerate(labels)
        ]
        label_part = "&".join(label_tokens) if label_tokens else "image"
    elif modality == "tabular":
        label_part = _slug_model_label(result.get("targetColumn"), "target")
    else:
        modality = "model"
        label_part = _slug_model_label(result.get("name"), "model")

    # Keep generated filenames comfortably below common filesystem limits.
    max_label_length = 180 - len(modality) - 12
    if len(label_part) > max_label_length:
        label_part = label_part[:max_label_length].rstrip("-&-") or modality

    return f"{label_part}-{modality}"


def _allocate_model_name(storage_dir: Path, prefix: str) -> tuple[str, int]:
    """Allocate a globally increasing sequence shared by image and tabular models.

    The sequence file intentionally lives beside job folders, not inside one job,
    so deleting a model or using Delete All does not reset/reuse sequence numbers.
    """
    storage_dir.mkdir(parents=True, exist_ok=True)
    sequence_file = storage_dir / ".model_sequence"
    lock_file = storage_dir / ".model_sequence.lock"
    deadline = time.monotonic() + 20.0

    while True:
        try:
            descriptor = os.open(
                str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
            )
            os.close(descriptor)
            break
        except FileExistsError:
            try:
                # Recover only from a clearly abandoned lock; normal allocation
                # holds this lock for a few milliseconds.
                if time.time() - lock_file.stat().st_mtime > 600:
                    lock_file.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() >= deadline:
                raise RuntimeError("Timed out while allocating a unique model sequence number.")
            time.sleep(0.05)

    try:
        try:
            last_sequence = int(sequence_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            last_sequence = 0

        # Recover from committed metadata if the counter file was removed/corrupted.
        for metadata_path in storage_dir.glob("*/metadata.json"):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(metadata, dict):
                continue
            stored_sequence = metadata.get("modelSequence")
            if isinstance(stored_sequence, int):
                last_sequence = max(last_sequence, stored_sequence)
            existing_name = str(metadata.get("name") or "")
            match = re.search(r"-(?:image|tabular|model)-(\d+)$", existing_name)
            if match:
                last_sequence = max(last_sequence, int(match.group(1)))

        next_sequence = last_sequence + 1
        temporary = storage_dir / f".model_sequence.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(str(next_sequence))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, sequence_file)
        finally:
            temporary.unlink(missing_ok=True)

        return f"{prefix}-{next_sequence}", next_sequence
    finally:
        lock_file.unlink(missing_ok=True)


def persist_model_artifact(
    job_dir: Path,
    result: dict[str, Any],
    bundle: dict[str, Any],
) -> dict[str, Any]:
    """Persist a named trained model artifact and commit its registry metadata."""
    job_dir.mkdir(parents=True, exist_ok=True)
    temporary = job_dir / f".model.{uuid.uuid4().hex}.tmp"

    try:
        joblib.dump(bundle, temporary, compress=3)
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())

        storage_dir = job_dir.parent
        model_name, sequence = _allocate_model_name(
            storage_dir, _model_name_prefix(result)
        )
        model_filename = f"{model_name}.joblib"
        model_path = job_dir / model_filename
        os.replace(temporary, model_path)

        digest = hashlib.sha256()
        with model_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)

        artifact = {
            "file_name": model_filename,
            "artifact_path": model_filename,
            "format": "joblib",
            "relative_path": f"data/models/{result['jobId']}/{model_filename}",
            "size_bytes": model_path.stat().st_size,
            "sha256": digest.hexdigest(),
            "download_url": f"/api/jobs/{result['jobId']}/model",
        }

        result["name"] = model_name
        result["modelSequence"] = sequence
        result["modelArtifact"] = artifact
        result.setdefault("outputs", {}).setdefault("output_2_model", {}).update({
            "name": model_name,
            "format": "joblib",
            "artifact_path": model_filename,
            "artifact_size_bytes": artifact["size_bytes"],
            "artifact_sha256": artifact["sha256"],
        })
        result.setdefault("download", {})["model"] = artifact["download_url"]

        # metadata.json is the registry commit marker and is written last.
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
