from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .services.common import read_json, safe_job_id
from .services.image import predict_image, train_images
from .services.tabular import train_tabular

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = PROJECT_ROOT / "web-ui"
STORAGE_DIR = PROJECT_ROOT / "data" / "models"
MAX_UPLOAD_BYTES = 512 * 1024 * 1024

STORAGE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Multimodal AutoML Platform",
    version="2.1.0",
    description="Local AutoML backend for tabular data and image classification.",
)


def _job_directory(job_id: str) -> Path:
    try:
        safe_id = safe_job_id(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    directory = (STORAGE_DIR / safe_id).resolve()
    if directory.parent != STORAGE_DIR.resolve() or not directory.is_dir():
        raise HTTPException(status_code=404, detail="Không tìm thấy mô hình hoặc job đã bị xóa.")
    return directory


def _load_model(job_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    directory = _job_directory(job_id)
    metadata = read_json(directory / "metadata.json")
    model_file = directory / "model.joblib"
    if not metadata or not model_file.is_file():
        raise HTTPException(status_code=404, detail="Artifact mô hình không đầy đủ.")
    try:
        bundle = joblib.load(model_file)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Không thể nạp mô hình: {exc}") from exc
    return bundle, metadata


def _predict_tabular(bundle: dict[str, Any], values: Any) -> dict[str, Any]:
    features = list(bundle["feature_names"])
    numeric = set(bundle.get("numeric_features", []))
    categorical = set(bundle.get("categorical_features", []))
    if isinstance(values, dict):
        missing = [name for name in features if name not in values]
        extra = [name for name in values if name not in features]
        if missing or extra:
            raise HTTPException(status_code=422, detail=f"Vector feature không khớp. Thiếu: {missing}; dư: {extra}.")
        row = {name: values[name] for name in features}
    elif isinstance(values, list):
        if len(values) != len(features):
            raise HTTPException(status_code=422, detail=f"Cần đúng {len(features)} giá trị theo thứ tự: {', '.join(features)}.")
        row = dict(zip(features, values))
    else:
        raise HTTPException(status_code=422, detail="inputs.features phải là array hoặc object.")

    parsed: dict[str, Any] = {}
    for name, value in row.items():
        if isinstance(value, (list, dict)):
            raise HTTPException(status_code=422, detail=f"Đặc trưng '{name}' phải là giá trị đơn, không phải array/object.")
        if value is None or (isinstance(value, str) and not value.strip()):
            parsed[name] = np.nan
        elif name in numeric:
            try:
                numeric_value = float(value)
                parsed[name] = numeric_value if np.isfinite(numeric_value) else np.nan
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=f"Đặc trưng '{name}' phải là số hữu hạn hoặc để trống.") from exc
        elif name in categorical:
            parsed[name] = str(value).strip()
        else:
            parsed[name] = value

    frame = pd.DataFrame([parsed], columns=features)
    pipeline = bundle["pipeline"]
    prediction = pipeline.predict(frame)[0]
    if bundle["task"] == "classification":
        predicted_class = str(prediction)
        probability_map: dict[str, float] = {}
        confidence = None
        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(frame)[0]
            class_names = [str(value) for value in pipeline.classes_]
            probability_map = {name: float(value) for name, value in zip(class_names, probabilities)}
            confidence = float(np.max(probabilities))
        return {"class": predicted_class, "confidence": confidence, "probabilities": probability_map}
    return {"target_value": float(prediction)}


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "multimodal-automl", "version": app.version}


@app.post("/api/gateway/submit")
async def submit_pipeline(
    modality: str = Form(...),
    file: UploadFile = File(...),
    target_column: str = Form("target_label"),
    task_type: str = Form("auto"),
):
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File vượt quá giới hạn 512 MB.")
    if not content:
        raise HTTPException(status_code=400, detail="File tải lên rỗng.")
    original_name = Path(file.filename or "upload").name
    try:
        if modality == "tabular":
            if Path(original_name).suffix.lower() != ".csv":
                raise HTTPException(status_code=415, detail="Tabular chỉ nhận file .CSV.")
            result = await run_in_threadpool(
                train_tabular, content, original_name, target_column.strip(), task_type, STORAGE_DIR
            )
        elif modality == "image":
            if Path(original_name).suffix.lower() != ".zip":
                raise HTTPException(status_code=415, detail="Image classification chỉ nhận file .ZIP.")
            result = await run_in_threadpool(train_images, content, original_name, STORAGE_DIR)
        else:
            raise HTTPException(status_code=422, detail="modality phải là tabular hoặc image.")
        return result
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        # Keep the server alive; log full details to console while returning a concise error.
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Training thất bại: {type(exc).__name__}: {exc}") from exc


@app.post("/api/gateway/predict")
async def predict_tabular(payload: dict[str, Any]):
    job_id = str(payload.get("model_id") or payload.get("jobId") or "")
    bundle, _ = _load_model(job_id)
    if bundle.get("modality") != "tabular":
        raise HTTPException(status_code=409, detail="Mô hình được chọn không phải mô hình tabular.")
    inputs = payload.get("inputs") or {}
    try:
        prediction = _predict_tabular(bundle, inputs.get("features"))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Không thể suy luận với vector đầu vào: {exc}") from exc
    return {"status": "SUCCESS", "model_id": job_id, "prediction": prediction}


@app.post("/api/gateway/predict/image")
async def predict_image_endpoint(
    model_id: str = Form(...),
    file: UploadFile = File(...),
):
    bundle, _ = _load_model(model_id)
    if bundle.get("modality") != "image":
        raise HTTPException(status_code=409, detail="Mô hình được chọn không phải mô hình ảnh.")
    content = await file.read(25 * 1024 * 1024 + 1)
    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Ảnh suy luận vượt quá giới hạn 25 MB.")
    try:
        prediction = await run_in_threadpool(predict_image, bundle, content)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"status": "SUCCESS", "model_id": model_id, "prediction": prediction}


@app.get("/api/models")
def list_models() -> dict[str, Any]:
    items = []
    for directory in STORAGE_DIR.iterdir():
        if not directory.is_dir():
            continue
        metadata = read_json(directory / "metadata.json")
        if metadata and (directory / "model.joblib").is_file():
            items.append(metadata)
    items.sort(key=lambda item: item.get("createdAt", ""), reverse=True)
    return {"items": items}


@app.delete("/api/models/{job_id}")
def delete_model(job_id: str) -> dict[str, Any]:
    directory = _job_directory(job_id)
    shutil.rmtree(directory)
    return {"status": "SUCCESS", "deleted": job_id}


@app.get("/api/jobs/{job_id}/dataset")
def download_dataset(job_id: str):
    directory = _job_directory(job_id)
    candidates = [directory / "processed_dataset.csv", directory / "processed_dataset.zip"]
    artifact = next((path for path in candidates if path.is_file()), None)
    if artifact is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy dataset đã xử lý.")
    media = "text/csv" if artifact.suffix == ".csv" else "application/zip"
    return FileResponse(artifact, media_type=media, filename=artifact.name)


@app.get("/api/jobs/{job_id}/model")
def download_model(job_id: str):
    directory = _job_directory(job_id)
    artifact = directory / "model.joblib"
    if not artifact.is_file():
        raise HTTPException(status_code=404, detail="Không tìm thấy model artifact.")
    return FileResponse(artifact, media_type="application/octet-stream", filename=f"{job_id}_model.joblib")


if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
