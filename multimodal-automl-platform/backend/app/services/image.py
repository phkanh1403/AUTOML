from __future__ import annotations

import hashlib
import io
import math
import re
import time
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import joblib
import numpy as np
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError
from skimage.feature import hog
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, IsolationForest, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .common import make_response_base, new_job_id, write_json

Image.MAX_IMAGE_PIXELS = 25_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = False
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_IMAGES = 5000
MIN_IMAGES_PER_CLASS = 100
MAX_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
CONFIDENCE_THRESHOLD = 0.64
RANDOM_STATE = 42


class DuplicateAcrossClassesError(ValueError):
    """Raised when the exact same image content is assigned to different labels."""


def _normalise_archive_path(name: str) -> tuple[str, ...] | None:
    clean = name.replace("\\", "/")
    pure = PurePosixPath(clean)
    if pure.is_absolute() or any(part in {"..", ""} for part in pure.parts):
        return None
    parts = tuple(part for part in pure.parts if part not in {".", "__MACOSX"})
    if not parts or any(part.startswith(".") for part in parts) or re.match(r"^[A-Za-z]:", parts[0]):
        return None
    return parts


def _decode_image(image_bytes: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise ValueError("Chỉ chấp nhận nội dung ảnh JPEG/PNG, không chỉ dựa vào phần mở rộng file.")
            image.verify()
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise ValueError("Chỉ chấp nhận nội dung ảnh JPEG/PNG.")
            image = ImageOps.exif_transpose(image)
            if image.width < 16 or image.height < 16:
                raise ValueError("Kích thước ảnh quá nhỏ (tối thiểu 16×16 pixel).")
            if image.width * image.height > Image.MAX_IMAGE_PIXELS:
                raise ValueError("Ảnh vượt giới hạn số pixel an toàn.")
            if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
                rgba = image.convert("RGBA")
                rgb = Image.new("RGB", rgba.size, (245, 245, 245))
                rgb.paste(rgba, mask=rgba.getchannel("A"))
                return rgb
            return image.convert("RGB")
    except ValueError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"File không phải ảnh JPEG/PNG hợp lệ: {exc}") from exc


def _image_feature(image: Image.Image) -> np.ndarray:
    # Preserve aspect ratio and letterbox rather than stretching object geometry.
    image = image.convert("RGB").copy()
    image.thumbnail((64, 64), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (64, 64), (245, 245, 245))
    canvas.paste(image, ((64 - image.width) // 2, (64 - image.height) // 2))
    rgb = np.asarray(canvas, dtype=np.float32) / 255.0
    gray = np.dot(rgb[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))
    hog_vector = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)

    histograms = []
    for channel in range(3):
        hist, _ = np.histogram(rgb[:, :, channel], bins=16, range=(0.0, 1.0), density=True)
        histograms.append(hist.astype(np.float32))
    color_vector = np.concatenate(histograms)
    return np.concatenate([hog_vector, color_vector]).astype(np.float32)


def _extract_archive(raw_zip: bytes) -> tuple[list[np.ndarray], list[str], list[tuple[str, bytes]], list[str]]:
    """Read class-folder ZIPs. The class name is the folder immediately below the dataset root."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw_zip))
    except (zipfile.BadZipFile, OSError) as exc:
        raise ValueError("File tải lên không phải ZIP hợp lệ.") from exc

    with archive:
        infos = [item for item in archive.infolist() if not item.is_dir()]
        if len(infos) > 20_000:
            raise ValueError("ZIP có quá nhiều file entries; giới hạn là 20.000 file.")
        if sum(item.file_size for item in infos) > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("Dataset ZIP giải nén vượt quá 512 MB.")
        for item in infos:
            if item.file_size > 25 * 1024 * 1024:
                raise ValueError(f"File trong ZIP quá lớn (>25 MB): {item.filename}")
            if item.file_size > 1_000_000 and item.file_size / max(item.compress_size, 1) > 300:
                raise ValueError(f"Tỷ lệ nén bất thường, nghi ZIP bomb: {item.filename}")

        image_infos = []
        for item in infos:
            parts = _normalise_archive_path(item.filename)
            if parts and Path(parts[-1]).suffix.lower() in ALLOWED_EXTENSIONS:
                image_infos.append((item, parts))
        if len(image_infos) > MAX_IMAGES:
            raise ValueError(f"Dataset có {len(image_infos)} ảnh; giới hạn hiện tại là {MAX_IMAGES} ảnh/lần train.")
        if not image_infos:
            raise ValueError("Không tìm thấy ảnh JPG/JPEG/PNG trong ZIP.")

        # Accepted layouts: ClassName/image.jpg or wrapper/ClassName/image.jpg.
        # A common wrapper is detected only when every image is at least 3 levels deep.
        first_parts = [parts[0] for _, parts in image_infos if len(parts) >= 2]
        has_common_wrapper = (
            len(first_parts) == len(image_infos)
            and len(set(first_parts)) == 1
            and all(len(parts) >= 3 for _, parts in image_infos)
        )
        class_index = 1 if has_common_wrapper else 0

        features: list[np.ndarray] = []
        labels: list[str] = []
        accepted: list[tuple[str, bytes]] = []
        invalid: list[str] = []
        hashes: dict[str, tuple[str, str]] = {}

        for item, parts in image_infos:
            if len(parts) <= class_index + 1:
                invalid.append(item.filename)
                continue
            label = parts[class_index].strip()
            if not label or len(label) > 128 or ":" in label:
                invalid.append(item.filename)
                continue
            if label.casefold() == "other":
                raise ValueError("Tên thư mục 'Other' được dành riêng cho kết quả từ chối nhận diện; hãy đổi tên lớp này.")
            try:
                content = archive.read(item)
                digest = hashlib.sha256(content).hexdigest()
                if digest in hashes:
                    previous_label, previous_name = hashes[digest]
                    if previous_label != label:
                        raise DuplicateAcrossClassesError(
                            f"Một ảnh trùng nội dung xuất hiện ở hai lớp '{previous_label}' và '{label}' "
                            f"({previous_name}, {item.filename}). Hãy loại bỏ bản trùng hoặc sửa nhãn."
                        )
                    continue
                image = _decode_image(content)
                vector = _image_feature(image)
                hashes[digest] = (label, item.filename)
                features.append(vector)
                labels.append(label)
                safe_name = "/".join((label, *parts[class_index + 1:]))
                accepted.append((safe_name, content))
            except DuplicateAcrossClassesError:
                raise
            except ValueError:
                # Corrupt/unsupported image contents are counted and skipped.
                invalid.append(item.filename)
            except Exception:
                invalid.append(item.filename)

    counts = Counter(labels)
    if not counts:
        raise ValueError(
            "Không tìm thấy ảnh hợp lệ bên trong thư mục lớp. Hãy đặt ảnh trong folder có tên là nhãn, "
            "ví dụ Apple/image_001.jpg."
        )
    insufficient = {name: count for name, count in counts.items() if count < MIN_IMAGES_PER_CLASS}
    if insufficient:
        details = ", ".join(f"{name}: {count}" for name, count in sorted(insufficient.items()))
        raise ValueError(
            f"Mỗi thư mục lớp phải có ít nhất {MIN_IMAGES_PER_CLASS} ảnh hợp lệ, không trùng. "
            f"Số lượng hiện tại: {details}. Ảnh hỏng và ảnh trùng không được tính."
        )
    return features, labels, accepted, invalid


def _candidate_models():
    return [
        ("HOG + SVM (RBF)", SVC(C=2.0, kernel="rbf", gamma="scale", class_weight="balanced", probability=True, random_state=RANDOM_STATE)),
        ("HOG + Extra Trees", ExtraTreesClassifier(n_estimators=250, max_features="sqrt", class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1)),
        ("HOG + Random Forest", RandomForestClassifier(n_estimators=200, max_features="sqrt", class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=1)),
    ]



def _thresholded_predictions(estimator, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the user-facing reject rule to model probabilities."""
    if not hasattr(estimator, "predict_proba"):
        predicted = np.asarray(estimator.predict(X), dtype=str)
        return predicted, np.ones(len(predicted), dtype=float)
    probabilities = np.asarray(estimator.predict_proba(X), dtype=float)
    class_names = np.asarray([str(value) for value in estimator.classes_], dtype=object)
    best_indices = np.argmax(probabilities, axis=1)
    confidence = probabilities[np.arange(len(probabilities)), best_indices]
    candidates = class_names[best_indices].astype(str)
    predictions = np.where(confidence > CONFIDENCE_THRESHOLD, candidates, "Other")
    return predictions.astype(str), confidence.astype(float)


def _thresholded_f1_macro(estimator, X: np.ndarray, y_true: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    predicted, _ = _thresholded_predictions(estimator, X)
    labels = sorted(str(value) for value in np.unique(np.asarray(y_true, dtype=str)))
    return float(f1_score(y_true, predicted, labels=labels, average="macro", zero_division=0))


def _classification_metrics_with_reject(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
        f1_score, precision_score, recall_score,
    )

    known_labels = sorted(str(value) for value in np.unique(np.asarray(y_true, dtype=str)))
    display_labels = known_labels + (["Other"] if "Other" in set(map(str, y_pred)) and "Other" not in known_labels else [])
    recall_macro = float(recall_score(y_true, y_pred, labels=known_labels, average="macro", zero_division=0))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # For known classes, balanced accuracy is macro recall; using this avoids treating
        # the reject-only output label as a class with no ground-truth support.
        "balanced_accuracy": recall_macro,
        "f1_macro": float(f1_score(y_true, y_pred, labels=known_labels, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=known_labels, average="macro", zero_division=0)),
        "recall_macro": recall_macro,
        "rejected_count": int(np.sum(np.asarray(y_pred, dtype=str) == "Other")),
        "rejected_rate": float(np.mean(np.asarray(y_pred, dtype=str) == "Other")),
        "confusion_matrix": {
            "labels": display_labels,
            "matrix": confusion_matrix(y_true, y_pred, labels=display_labels).tolist(),
        },
        "classification_report": classification_report(
            y_true, y_pred, labels=display_labels, output_dict=True, zero_division=0
        ),
    }

def _group_importance(model: Pipeline, X_test: np.ndarray, y_test: np.ndarray, hog_dim: int) -> dict[str, float]:
    from sklearn.metrics import f1_score

    baseline = _thresholded_f1_macro(model, X_test, y_test)
    rng = np.random.default_rng(RANDOM_STATE)
    importance: dict[str, float] = {}
    groups = (("HOG texture", np.arange(hog_dim)), ("RGB color histogram", np.arange(hog_dim, X_test.shape[1])))
    for name, indices in groups:
        drops = []
        for _ in range(3):
            permuted = X_test.copy()
            order = rng.permutation(len(X_test))
            permuted[:, indices] = permuted[order][:, indices]
            drops.append(float(baseline - f1_score(y_test, model.predict(permuted), average="macro", zero_division=0)))
        importance[name] = float(np.mean(drops))
    positive = {key: max(value, 0.0) for key, value in importance.items()}
    total = sum(positive.values())
    return {key: (round(value / total, 6) if total > 0 else 0.0) for key, value in positive.items()}


def _novelty_confidence(score: float, lower_score: float, upper_score: float) -> float:
    """Map an inlier score to an acceptance score whose decision boundary is 0.64."""
    scale = max(float(upper_score - lower_score), 1e-4)
    logit_at_boundary = math.log(CONFIDENCE_THRESHOLD / (1.0 - CONFIDENCE_THRESHOLD))
    z = logit_at_boundary + 4.0 * (float(score) - float(lower_score)) / scale
    z = max(-30.0, min(30.0, z))
    return float(1.0 / (1.0 + math.exp(-z)))


def _persist_image_artifacts(
    job_dir: Path,
    accepted: list[tuple[str, bytes]],
    classes: list[str],
    confidence_mode: str,
) -> None:
    with zipfile.ZipFile(job_dir / "processed_dataset.zip", "w", compression=zipfile.ZIP_DEFLATED) as cleaned_zip:
        for archive_name, content in accepted:
            cleaned_zip.writestr(archive_name, content)
        cleaned_zip.writestr(
            "DATASET_INFO.txt",
            "Dataset đã được xác thực và loại ảnh trùng SHA-256.\n"
            "Nhãn được suy ra từ tên thư mục lớp bên trong ZIP.\n"
            f"Các lớp: {', '.join(classes)}\n"
            f"Chế độ mô hình: {confidence_mode}\n"
            f"Ngưỡng từ chối: confidence > {CONFIDENCE_THRESHOLD:.2f}; nếu không trả về Other.\n"
            "Đặc trưng ảnh: HOG texture + RGB color histogram; ảnh gốc không bị chỉnh sửa.\n",
        )


def _train_single_class(
    X: np.ndarray,
    label: str,
    job_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    """Train a one-class novelty detector and calibrate it on held-out inliers."""
    X_train, X_calibration = train_test_split(X, test_size=0.20, random_state=RANDOM_STATE)
    model = Pipeline([
        ("scale", StandardScaler()),
        ("model", IsolationForest(n_estimators=300, contamination=0.05, random_state=RANDOM_STATE, n_jobs=1)),
    ])
    model.fit(X_train)

    # Use images not seen during fitting to set the inlier-score operating boundary.
    calibration_scores = model.score_samples(X_calibration)
    center = float(np.mean(calibration_scores))
    spread = max(float(np.std(calibration_scores)), 1e-4)
    lower = center - 3.0 * spread
    upper = max(float(np.quantile(calibration_scores, 0.95)), lower + spread)
    calibration_confidences = np.asarray([
        _novelty_confidence(score, lower, upper) for score in calibration_scores
    ])
    accepted_rate = float(np.mean(calibration_confidences > CONFIDENCE_THRESHOLD))

    bundle = {
        "modality": "image",
        "mode": "one_class_novelty",
        # Keep the same 80%-fit model used to derive the calibration boundary so score
        # scales remain consistent at inference time.
        "pipeline": model,
        "classes": [label],
        "feature_type": "HOG + RGB histograms",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "novelty_score_lower": lower,
        "novelty_score_upper": upper,
    }
    metrics = {
        "known_class_acceptance_rate": accepted_rate,
        "calibration_known_images": int(len(X_calibration)),
        "accepted_known_images": int(np.sum(calibration_confidences > CONFIDENCE_THRESHOLD)),
        "note": "Tỷ lệ trên tập calibration gồm ảnh cùng lớp được giữ riêng khi fit. Dataset một lớp không có ảnh âm tính nên không đo được độ chính xác khi từ chối lớp lạ.",
    }
    history = {
        "metrics": metrics,
        "primary_metric": "Calibration known-class acceptance rate",
        "primary_value": accepted_rate,
        "cv_metric": "Not applicable (one-class novelty detection)",
        "cv_score": None,
        "cv_std": None,
        "cv_folds": 0,
        "baseline_algorithm": "Not applicable",
        "baseline_metric": "Not applicable",
        "baseline_value": None,
        "improvement_vs_baseline": None,
    }
    xai = {
        "method": "One-class novelty detection",
        "top_contributors": "Isolation Forest learns the distribution of HOG texture and RGB color features for the single supplied folder.",
        "importanceMap": {},
        "model_used": "Isolation Forest",
        "final_features": ["HOG texture", "RGB color histogram"],
    }
    return bundle, metrics, history | {"xai": xai}, "Isolation Forest (one-class novelty detection)"


def train_images(raw_zip: bytes, filename: str, storage_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    features, labels, accepted, invalid = _extract_archive(raw_zip)
    X = np.vstack(features)
    y = np.asarray(labels, dtype=str)
    classes = sorted(set(labels))
    counts = Counter(labels)
    job_id = new_job_id()
    job_dir = storage_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=False)

    try:
        if len(classes) == 1:
            bundle, metrics, history, algorithm = _train_single_class(X, classes[0], job_dir)
            xai = history.pop("xai")
            cv_score = cv_std = None
            folds = 0
            baseline_metrics = None
            improvement_vs_baseline = None
            mode = "one_class_novelty"
            confidence_note = (
                "Một lớp: confidence là điểm phù hợp với phân bố ảnh đã học từ folder; "
                "không phải xác suất đã hiệu chuẩn. Ảnh không đủ phù hợp sẽ trả về Other."
            )
        else:
            test_count = max(int(math.ceil(0.20 * len(y))), len(classes))
            if test_count >= len(y):
                raise ValueError("Không đủ ảnh để tạo tập kiểm tra độc lập.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_count, random_state=RANDOM_STATE, stratify=y
            )
            folds = min(3, int(min(Counter(y_train).values())))
            if folds < 2:
                raise ValueError("Không đủ ảnh mỗi lớp trong tập train để cross-validation.")
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
            scored = []
            for name, estimator in _candidate_models():
                pipeline = Pipeline([("scale", StandardScaler()), ("model", estimator)])
                try:
                    scores = cross_val_score(
                        pipeline, X_train, y_train, cv=cv, scoring=_thresholded_f1_macro, n_jobs=1, error_score="raise"
                    )
                    scored.append((float(np.mean(scores)), name, estimator, float(np.std(scores))))
                except Exception:
                    continue
            if not scored:
                raise ValueError("Các mô hình phân loại ảnh đều thất bại trong cross-validation.")
            scored.sort(key=lambda row: row[0], reverse=True)
            cv_score, algorithm, selected_estimator, cv_std = scored[0]
            holdout_model = Pipeline([("scale", StandardScaler()), ("model", selected_estimator)])
            holdout_model.fit(X_train, y_train)
            prediction, _ = _thresholded_predictions(holdout_model, X_test)
            metrics = _classification_metrics_with_reject(y_test, prediction)
            baseline = DummyClassifier(strategy="prior")
            baseline.fit(X_train, y_train)
            baseline_prediction, _ = _thresholded_predictions(baseline, X_test)
            baseline_metrics = _classification_metrics_with_reject(y_test, baseline_prediction)
            improvement_vs_baseline = float(metrics["f1_macro"] - baseline_metrics["f1_macro"])
            hog_dim = X.shape[1] - 48
            importance = _group_importance(holdout_model, X_test, y_test, hog_dim)
            final_model = Pipeline([("scale", StandardScaler()), ("model", selected_estimator)])
            final_model.fit(X, y)
            bundle = {
                "modality": "image",
                "mode": "multiclass",
                "pipeline": final_model,
                "classes": classes,
                "feature_type": "HOG + RGB histograms",
                "confidence_threshold": CONFIDENCE_THRESHOLD,
            }
            history = {
                "metrics": metrics,
                "primary_metric": "F1-macro",
                "primary_value": float(metrics["f1_macro"]),
                "cv_score": float(cv_score),
                "cv_metric": "F1-macro",
                "cv_std": float(cv_std),
                "cv_folds": int(folds),
                "baseline_algorithm": "DummyClassifier (prior)",
                "baseline_metric": "F1-macro",
                "baseline_value": float(baseline_metrics["f1_macro"]),
                "improvement_vs_baseline": float(improvement_vs_baseline),
                "baseline_metrics": baseline_metrics,
                "candidate_scores": [
                    {"algorithm": name, "cv_score": float(score), "cv_std": float(std)}
                    for score, name, _, std in scored
                ],
            }
            xai = {
                "method": "Group Permutation Importance trên tập kiểm tra",
                "top_contributors": "So sánh mức suy giảm F1-macro khi xáo trộn nhóm đặc trưng kết cấu và màu sắc.",
                "importanceMap": importance,
                "model_used": algorithm,
                "final_features": ["HOG texture", "RGB color histogram"],
            }
            mode = "multiclass"
            confidence_note = (
                f"Chỉ trả nhãn lớp khi xác suất lớp cao nhất > {CONFIDENCE_THRESHOLD:.0%}; "
                "nếu không, kết quả là Other. Xác suất là ước lượng của model, chưa được calibration riêng."
            )

        joblib.dump(bundle, job_dir / "model.joblib", compress=3)
        _persist_image_artifacts(job_dir, accepted, classes, mode)
        elapsed = time.perf_counter() - started
        history["execution_time_sec"] = round(float(elapsed), 3)
        warnings = []
        if len(y) < 500:
            warnings.append("Dataset mới đạt mức tối thiểu; nên có thêm ảnh đa dạng cho mỗi lớp để cải thiện khả năng tổng quát hóa.")
        if mode == "one_class_novelty":
            warnings.append(
                "Dataset chỉ có một lớp: hệ thống học phạm vi ảnh của lớp này. Muốn phân biệt nhiều nhãn cụ thể, hãy thêm folder tương ứng cho từng lớp."
            )
        elif improvement_vs_baseline is not None and improvement_vs_baseline <= 0:
            warnings.append("Image model không vượt baseline majority-class trên holdout; cần thêm ảnh/đặc trưng hoặc kiểm tra nhãn.")
        if invalid:
            warnings.append(f"Bỏ qua {len(invalid)} file ảnh không đọc được hoặc đường dẫn không hợp lệ.")
        if len(counts) > 1 and max(counts.values()) / min(counts.values()) >= 3:
            warnings.append("Dataset mất cân bằng lớp; nên bổ sung ảnh cho lớp ít mẫu.")

        result = make_response_base(job_id, "image", algorithm, metrics, elapsed)
        result.update({
            "name": f"Image - {algorithm}",
            "sourceFileName": Path(filename).name,
            "labels": classes,
            "classes": classes,
            "taskType": "classification" if len(classes) > 1 else "one_class_novelty",
            "sampleCount": int(len(y)),
            "trainingSampleCount": int(round(len(y) * 0.8)),
            "testSampleCount": int(len(y) - round(len(y) * 0.8)),
            "featureType": "HOG texture + RGB color histograms (64×64)",
            "classCounts": dict(counts),
            "invalidImageCount": len(invalid),
            "confidenceThreshold": CONFIDENCE_THRESHOLD,
            "confidenceRule": confidence_note,
            "imageMode": mode,
            "warnings": warnings,
            "download": {"dataset": f"/api/jobs/{job_id}/dataset", "model": f"/api/jobs/{job_id}/model"},
            "outputs": {
                "output_1_dataset": {
                    "status_msg": f"Đã xác thực {len(y)} ảnh thuộc {len(classes)} folder nhãn; yêu cầu tối thiểu {MIN_IMAGES_PER_CLASS} ảnh hợp lệ/folder; bỏ qua {len(invalid)} file lỗi và loại ảnh trùng SHA-256.",
                    "preprocessing_applied": "Nhãn lấy từ tên folder; resize nội bộ 64×64; trích xuất HOG và RGB histogram; ảnh gốc không bị thay đổi.",
                    "artifact_path": "processed_dataset.zip",
                },
                "output_2_model": {
                    "algorithm": algorithm,
                    "format": "joblib",
                    "artifact_path": "model.joblib",
                    "mode": mode,
                    "confidence_threshold": CONFIDENCE_THRESHOLD,
                },
                "output_3_history": history,
                "output_4_xai": xai,
            },
        })
        write_json(job_dir / "metadata.json", result)
        return result
    except Exception:
        # Do not leave a seemingly trained but incomplete model in the registry.
        import shutil
        shutil.rmtree(job_dir, ignore_errors=True)
        raise


def predict_image(model_bundle: dict[str, Any], image_bytes: bytes) -> dict[str, Any]:
    image = _decode_image(image_bytes)
    vector = _image_feature(image).reshape(1, -1)
    pipeline = model_bundle["pipeline"]
    threshold = float(model_bundle.get("confidence_threshold", CONFIDENCE_THRESHOLD))
    mode = model_bundle.get("mode", "multiclass")

    if mode == "one_class_novelty":
        label = str(model_bundle["classes"][0])
        score = float(pipeline.score_samples(vector)[0])
        confidence = _novelty_confidence(
            score,
            float(model_bundle["novelty_score_lower"]),
            float(model_bundle["novelty_score_upper"]),
        )
        recognized = confidence > threshold
        return {
            "class": label if recognized else "Other",
            "candidate_class": label,
            "confidence": confidence,
            "confidence_type": "novelty_score",
            "confidence_threshold": threshold,
            "is_known": recognized,
            "probabilities": {label: confidence, "Other": 1.0 - confidence},
        }

    if hasattr(pipeline, "predict_proba"):
        probabilities = np.asarray(pipeline.predict_proba(vector)[0], dtype=float)
        class_names = [str(value) for value in pipeline.classes_]
        best_index = int(np.argmax(probabilities))
        candidate = class_names[best_index]
        confidence = float(probabilities[best_index])
        recognized = confidence > threshold
        probability_map = {name: float(value) for name, value in zip(class_names, probabilities)}
    else:
        candidate = str(pipeline.predict(vector)[0])
        confidence = 1.0
        recognized = True
        probability_map = {candidate: 1.0}
    return {
        "class": candidate if recognized else "Other",
        "candidate_class": candidate,
        "confidence": confidence,
        "confidence_type": "model_probability_estimate",
        "confidence_threshold": threshold,
        "is_known": recognized,
        "probabilities": probability_map,
    }
