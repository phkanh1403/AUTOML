from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from backend.app.services.tabular import train_tabular
from backend.app.services.image import train_images, predict_image
from backend.app.main import _predict_tabular
import joblib


def test_tabular_classification_train_and_predict(tmp_path):
    rng = np.random.default_rng(7)
    n = 180
    signal = rng.normal(size=n)
    region = rng.choice(["north", "south", "west"], size=n)
    label = np.where(signal + (region == "north") * 0.8 > 0, "positive", "negative")
    frame = pd.DataFrame({
        "signal": signal,
        "region": region,
        "mostly_missing": [None if i % 10 else i for i in range(n)],
        "constant": 1,
        "target": label,
    })
    raw = frame.to_csv(index=False).encode("utf-8")
    result = train_tabular(raw, "sample.csv", "target", "classification", tmp_path)
    assert result["status"] == "SUCCESS"
    assert result["taskType"] == "classification"
    assert "mostly_missing" not in result["features"]
    bundle = joblib.load(tmp_path / result["jobId"] / "model.joblib")
    row = {"signal": 1.4, "region": "north"}
    prediction = _predict_tabular(bundle, row)
    assert prediction["class"] in {"positive", "negative"}
    assert 0 <= prediction["confidence"] <= 1
    assert (tmp_path / result["jobId"] / "processed_dataset.csv").exists()


def test_tabular_regression_train(tmp_path):
    rng = np.random.default_rng(12)
    n = 100
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    target = 3.2 * x - 0.7 * z + rng.normal(scale=0.2, size=n)
    frame = pd.DataFrame({"x": x, "z": z, "target": target})
    result = train_tabular(frame.to_csv(index=False).encode(), "reg.csv", "target", "regression", tmp_path)
    assert result["taskType"] == "regression"
    assert "mae" in result["outputs"]["output_3_history"]["metrics"]
    bundle = joblib.load(tmp_path / result["jobId"] / "model.joblib")
    prediction = _predict_tabular(bundle, [1.0, 0.0])
    assert isinstance(prediction["target_value"], float)


def _make_image(color: tuple[int, int, int], shape: str, seed: int) -> bytes:
    image = Image.new("RGB", (96, 96), color=(245, 245, 245))
    draw = ImageDraw.Draw(image)
    if shape == "circle":
        draw.ellipse((14 + seed % 8, 14, 78, 78), fill=color, outline=(20, 20, 20), width=3)
    else:
        draw.rectangle((16, 16 + seed % 8, 78, 78), fill=color, outline=(20, 20, 20), width=3)
    rng = np.random.default_rng(seed)
    for _ in range(12):
        x, y = int(rng.integers(0, 96)), int(rng.integers(0, 96))
        draw.point((x, y), fill=tuple(int(v) for v in rng.integers(0, 255, size=3)))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _make_image_zip(class_specs: dict[str, tuple[str, tuple[int, int, int], int]], wrapper: str | None = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for class_name, (shape, color, count) in class_specs.items():
            for i in range(count):
                prefix = f"{wrapper}/" if wrapper else ""
                archive.writestr(
                    f"{prefix}{class_name}/{class_name}_{i:04d}.png",
                    _make_image(color, shape, i + sum(map(ord, class_name))),
                )
    return buffer.getvalue()


def test_image_train_and_predict(tmp_path):
    raw_zip = _make_image_zip({
        "Apple": ("circle", (25, 170, 60), 100),
        "Banana": ("square", (180, 35, 35), 100),
    }, wrapper="dataset")
    result = train_images(raw_zip, "images.zip", tmp_path)
    assert result["status"] == "SUCCESS"
    assert set(result["labels"]) == {"Apple", "Banana"}
    assert result["classCounts"] == {"Apple": 100, "Banana": 100}
    assert result["confidenceThreshold"] == 0.64
    bundle = joblib.load(tmp_path / result["jobId"] / "model.joblib")
    prediction = predict_image(bundle, _make_image((25, 170, 70), "circle", 999))
    assert prediction["class"] in {"Apple", "Banana", "Other"}
    assert prediction["confidence"] is not None
    assert prediction["confidence_threshold"] == 0.64
    if prediction["class"] == "Other":
        assert prediction["confidence"] <= 0.64
    else:
        assert prediction["confidence"] > 0.64
    assert (tmp_path / result["jobId"] / "processed_dataset.zip").exists()


def test_image_single_folder_uses_novelty_detection(tmp_path):
    raw_zip = _make_image_zip({"Apple": ("circle", (25, 170, 60), 100)})
    result = train_images(raw_zip, "single_class.zip", tmp_path)
    assert result["status"] == "SUCCESS"
    assert result["labels"] == ["Apple"]
    assert result["imageMode"] == "one_class_novelty"
    bundle = joblib.load(tmp_path / result["jobId"] / "model.joblib")
    prediction = predict_image(bundle, _make_image((25, 170, 60), "circle", 999))
    unknown_prediction = predict_image(bundle, _make_image((30, 30, 220), "square", 1777))
    assert prediction["class"] == "Apple"
    assert unknown_prediction["class"] == "Other"
    assert prediction["confidence_type"] == "novelty_score"


def test_api_training_registry_prediction_and_delete(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from backend.app import main

    monkeypatch.setattr(main, "STORAGE_DIR", tmp_path)
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({
        "temperature": rng.normal(25, 3, 90),
        "humidity": rng.normal(60, 8, 90),
    })
    frame["target"] = np.where(frame["temperature"] > 25, "warm", "cool")
    client = TestClient(main.app)
    response = client.post(
        "/api/gateway/submit",
        data={"modality": "tabular", "target_column": "target", "task_type": "classification"},
        files={"file": ("api_test.csv", frame.to_csv(index=False).encode(), "text/csv")},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    job_id = result["jobId"]
    prediction = client.post("/api/gateway/predict", json={
        "model_id": job_id,
        "inputs": {"features": {"temperature": 29.0, "humidity": 40.0}},
    })
    assert prediction.status_code == 200, prediction.text
    assert prediction.json()["prediction"]["class"] in {"warm", "cool"}
    assert client.get("/api/models").json()["items"]
    assert client.get(f"/api/jobs/{job_id}/dataset").status_code == 200
    assert client.get(f"/api/jobs/{job_id}/model").status_code == 200
    assert client.delete(f"/api/models/{job_id}").status_code == 200
    assert client.get("/api/models").json()["items"] == []


def test_image_dataset_rejects_flat_unlabeled_zip():
    import pytest
    from backend.app.services.image import _extract_archive

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for i in range(8):
            archive.writestr(f"image_{i}.png", _make_image((40, 150, 70), "circle", i))
    with pytest.raises(ValueError, match="folder có tên là nhãn"):
        _extract_archive(buffer.getvalue())


def test_image_dataset_rejects_insufficient_class_count():
    import pytest
    from backend.app.services.image import _extract_archive

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for class_name, shape in (("healthy", "circle"), ("diseased", "square")):
            for i in range(99):
                archive.writestr(f"{class_name}/{i}.png", _make_image((30, 160, 70), shape, i + (0 if class_name == "healthy" else 1000)))
    with pytest.raises(ValueError, match="ít nhất 100 ảnh"):
        _extract_archive(buffer.getvalue())


def test_tabular_infinite_values_are_treated_as_missing(tmp_path):
    frame = pd.DataFrame({
        "numeric": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, float("inf"), 14, 15, 16, 17, 18, 19, 20],
        "target": [0, 1] * 10,
    })
    result = train_tabular(frame.to_csv(index=False).encode(), "inf.csv", "target", "classification", tmp_path)
    assert result["status"] == "SUCCESS"


def test_api_image_train_and_real_image_inference(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from backend.app import main

    monkeypatch.setattr(main, "STORAGE_DIR", tmp_path)
    archive_buffer = _make_image_zip({
        "Apple": ("circle", (30, 170, 60), 100),
        "Banana": ("square", (175, 40, 40), 100),
    }, wrapper="train")
    client = TestClient(main.app)
    assert client.get("/").status_code == 200
    response = client.post(
        "/api/gateway/submit",
        data={"modality": "image"},
        files={"file": ("leaves.zip", archive_buffer, "application/zip")},
    )
    assert response.status_code == 200, response.text
    model_id = response.json()["jobId"]
    image_response = client.post(
        "/api/gateway/predict/image",
        data={"model_id": model_id},
        files={"file": ("arbitrary_filename.png", _make_image((30, 170, 70), "circle", 29), "image/png")},
    )
    assert image_response.status_code == 200, image_response.text
    prediction = image_response.json()["prediction"]
    assert prediction["class"] in {"Apple", "Banana", "Other"}
    assert prediction["confidence_threshold"] == 0.64


def test_image_prediction_applies_strict_64_percent_threshold():
    class FixedProbabilityModel:
        classes_ = np.asarray(["Apple", "Banana"])

        def __init__(self, probabilities):
            self.probabilities = np.asarray([probabilities], dtype=float)

        def predict_proba(self, _features):
            return self.probabilities

        def predict(self, _features):
            return self.classes_[int(np.argmax(self.probabilities[0])):int(np.argmax(self.probabilities[0])) + 1]

    image_bytes = _make_image((40, 150, 60), "circle", 42)
    for probabilities, expected in [((0.64, 0.36), "Other"), ((0.60, 0.40), "Other"), ((0.65, 0.35), "Apple")]:
        result = predict_image({"pipeline": FixedProbabilityModel(probabilities), "mode": "multiclass"}, image_bytes)
        assert result["class"] == expected
        assert result["confidence_threshold"] == 0.64
        assert (result["confidence"] > 0.64) is (expected == "Apple")
