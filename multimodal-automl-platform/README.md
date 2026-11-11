# Multimodal AutoML Platform — v2.2.0

A local AutoML workbench for **tabular classification/regression** and **image classification**. The backend performs real training and inference with scikit-learn, persists every successfully trained model as a downloadable `joblib` artifact, and maintains a disk-backed model registry. The web interface, API error messages, launch scripts, and this documentation are in English.

## 1. Architecture

```text
Browser (web-ui)
   | HTTP / FormData / JSON
   v
FastAPI backend (backend/app/main.py)
   |-- Tabular AutoML (backend/app/services/tabular.py)
   |     |-- CSV validation and cleaning
   |     |-- Imputation + RobustScaler + OneHotEncoder inside a Pipeline
   |     |-- CV model selection: Logistic Regression / Random Forest / Extra Trees
   |     |                         or Ridge / Random Forest / Extra Trees Regressor
   |     |-- Holdout metrics and permutation importance
   |     `-- Refit on all labeled rows for serving
   |-- Image AutoML (backend/app/services/image.py)
   |     |-- ZIP validation; folder names become class labels
   |     |-- HOG texture + RGB color histogram features
   |     |-- CV model selection: SVM / Random Forest / Extra Trees
   |     `-- 64% confidence rejection rule; lower-confidence outputs become Other
   `-- Persistent model registry: data/models/<job_id>/
         |-- model.joblib              # trained estimator/pipeline and inference metadata
         |-- metadata.json             # registry entry, evaluation, artifact checksum
         `-- processed_dataset.csv or processed_dataset.zip
```

The backend binds to `127.0.0.1` by default. The UI and API are served from the same origin; CORS is not opened globally.

## 2. Installation and startup

Requirements: Python 3.10–3.12 (Python 3.11 recommended). An Internet connection is required the first time dependencies are installed.

### Windows

1. Extract the ZIP archive.
2. Open the `multimodal-automl-platform` folder.
3. Run `scripts/start_all_services.bat`.
4. Open `http://127.0.0.1:8000` if the browser does not open automatically.

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./scripts/start_all_services.sh
```

Or start the backend manually on any supported operating system:

```bash
python -m pip install -r requirements.txt
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Do not open `web-ui/index.html` directly with `file://`. Access the application through FastAPI so that same-origin API requests work correctly.

## 3. Tabular training

- Input: a CSV file with a header row (current processing limits: 250,000 rows and 2,000 columns).
- `Target Column`: the exact target column name after surrounding whitespace is trimmed.
- `Task Type`: `Auto`, `Classification`, or `Regression`.
- At least 12 rows with valid target values are required. Classification requires at least 3 samples per class to create the holdout and cross-validation splits.

Preprocessing and evaluation:

1. Read CSV using encoding and delimiter fallbacks.
2. Normalize common missing-value tokens, remove duplicate rows, and remove rows with missing targets.
3. Remove features with at least 70% missing values, constant features, and duplicate features.
4. Numeric features: median imputation, missing indicators, and `RobustScaler`.
5. Categorical features: most-frequent imputation and `OneHotEncoder` with unknown-category handling.
6. All imputers, encoders, and scalers are fitted inside the scikit-learn `Pipeline` to avoid preprocessing leakage across validation splits.
7. Create a stratified holdout for classification; compare candidates using cross-validation on the training split; report final metrics on the holdout.
8. Compare against a simple prior/median baseline and warn when the selected model does not outperform it on the holdout.
9. Refit the selected pipeline on all labeled rows after evaluation for inference.

`Auto` treats text targets as classification. Integer-like numeric targets with a relatively small number of distinct values are usually treated as classification; continuous numeric targets are treated as regression. Choose the task explicitly when the target semantics are known.

## 4. Image training

**Folder names inside the ZIP archive are the class labels.** No manual label entry is required in the web interface. The archive may contain one or multiple class folders. Every class folder must contain at least **100 valid, unique JPEG/JPG/PNG images**. Corrupt images and duplicate image contents are not counted toward the minimum.

Example with multiple classes:

```text
dataset.zip
|-- Apple/
|   |-- image_001.jpg
|   |-- image_002.jpg
|   `-- ... (at least 100 images)
`-- Banana/
    |-- image_001.jpg
    |-- image_002.jpg
    `-- ... (at least 100 images)
```

An optional wrapper folder is supported:

```text
dataset.zip
`-- dataset/
    |-- Apple/  (at least 100 images)
    `-- Banana/ (at least 100 images)
```

Training behavior:

- **Two or more class folders:** train a multiclass classifier using folder names as labels (for example, `Apple` and `Banana`).
- **One class folder:** use Isolation Forest for one-class novelty detection. 80% of the images are used for fitting and the remaining 20% are reserved for calibration. Matching images can be returned as the folder label; insufficiently matching images are returned as `Other`. A one-class dataset does not contain negative examples, so unknown-class rejection performance cannot be fully evaluated from that dataset alone.
- **Minimum class size:** at least 100 valid, unique images in every class folder, not 100 images total. Current limits are 5,000 images and 512 MB of extracted data per ZIP.
- The folder name `Other` is reserved for rejected predictions and cannot be used as a training class name.
- The current image pipeline is a CPU-friendly classical computer vision baseline: resize to 64×64 internally, extract HOG texture and RGB histogram features, then select SVM / Random Forest / Extra Trees using macro F1 with the 64% rejection rule. One-class mode uses Isolation Forest. Source images are not modified.
- This is **not a pretrained CNN or Vision Transformer**. Real-world accuracy depends on dataset diversity, class balance, labels, backgrounds, camera angles, and lighting. Transfer learning with MobileNet or EfficientNet is a possible future upgrade when GPU resources are available.

### Image prediction and the `Other` rule

For multiclass models, the backend selects the class with the highest predicted probability. It returns that class only when confidence is **strictly greater than 64%**. If the highest confidence is **64% or lower**, the prediction is `Other`.

Examples:

| Candidate probabilities | Returned label |
|---|---|
| Apple 72%, Banana 28% | `Apple` |
| Apple 19%, Banana 81% | `Banana` |
| Apple 48%, Banana 52% | `Other` |
| Apple 64%, Banana 36% | `Other` |

The API response also includes `candidate_class`, `confidence`, `confidence_threshold`, and the per-class `probabilities` for review. For one-class novelty detection, confidence is a transformed novelty/similarity score on a 0–1 scale, not a statistically calibrated class probability.

## 5. Persistent model files and model registry

Every successful training run—tabular or image—creates a real trained model file on the backend:

```text
data/
`-- models/
    `-- <job_id>/
        |-- model.joblib
        |-- metadata.json
        `-- processed_dataset.csv  (tabular)
            or processed_dataset.zip (image)
```

- `model.joblib` contains the fitted estimator/pipeline and the metadata required for inference. It is the trained model output and can be downloaded from the training result panel or from the model registry.
- `metadata.json` stores the model name, modality, features/classes, evaluation results, artifact filename, file size, SHA-256 checksum, and download endpoint.
- The model registry reads from `data/models` on the backend. It does not depend on browser `localStorage`; saved models remain available after refreshing the page or restarting the application, as long as the `data/models` directory is preserved.
- Selecting a model in the registry uses the corresponding saved backend artifact for inference.
- Deleting a model in the web interface sends a backend `DELETE` request. The backend removes the entire `data/models/<job_id>/` directory, including `model.joblib`, `metadata.json`, and the processed dataset artifact. `Delete All` sends one backend request to remove all committed models and their files.
- If the project folder is moved or deleted, copy `data/models` with it to preserve trained models. The `.gitignore` excludes generated model files from source-control commits by default.
- Joblib uses Python pickle serialization internally. Only load model artifacts created by a trusted backend; never load untrusted `joblib` files.

### Inference inputs

- Tabular inference accepts a JSON array in the exact feature order or a JSON object keyed by feature name.
- Image inference uploads the actual JPEG/PNG image. The filename is not used to infer its class.
- Tabular classification returns a predicted class and confidence; regression returns a numeric target prediction.
- Image classification applies the strict confidence threshold described above and returns `Other` otherwise.

## 6. API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/gateway/submit` | Train `tabular` (multipart CSV) or `image` (multipart ZIP) |
| `POST` | `/api/gateway/predict` | Tabular inference (JSON) |
| `POST` | `/api/gateway/predict/image` | Image inference (multipart form) |
| `GET` | `/api/models` | List persistent model registry entries |
| `DELETE` | `/api/models/{job_id}` | Delete one model and all files in its job directory |
| `DELETE` | `/api/models` | Delete all committed models and their files |
| `GET` | `/api/jobs/{job_id}/dataset` | Download processed CSV/ZIP |
| `GET` | `/api/jobs/{job_id}/model` | Download the trained `joblib` artifact |

Interactive API documentation: `http://127.0.0.1:8000/docs`.

## 7. Tests and validation

Run the backend test suite:

```bash
python -m pytest backend/tests -q
```

Compile the backend modules:

```bash
python -m compileall backend
```

The test suite covers tabular classification/regression, multiclass image training, one-class novelty detection, real image inference, strict 64% rejection behavior, invalid image archives, persistent model downloads, and backend deletion of one or all model artifacts.

## 8. Accuracy and operational limitations

- AutoML cannot guarantee correct predictions when targets are mislabeled, data is noisy, leakage is present, the deployment domain differs from the training domain, or the dataset is not representative. Accuracy, F1, R², MAE, and RMSE are holdout results for the uploaded dataset, not a guarantee of real-world performance.
- A holdout is a single split; metrics may vary, particularly on small datasets. Review class distributions, baselines, confusion matrices, and an independent external test set.
- For time-dependent data, patients, users, production batches, or repeated measurements of the same subject, random splitting may cause leakage. Use group-based or time-based splitting where appropriate before reporting research results.
- The backend is designed for local, single-machine use. Do not expose it directly to the public Internet without authentication, authorization, rate limiting, and sandboxing.

## 9. Sample files

- `data/test_samples/example_tabular.csv`: synthetic tabular classification dataset; target column is `plant_status`.
- `data/test_samples/valid_images.zip`: 200 synthetic sample images (100 in each of two class folders, `Apple` and `Banana`) for a quick image-pipeline smoke test. It is not a real plant dataset; its scores must not be interpreted as real-world performance.
- `data/test_samples/invalid_count.zip`: intentionally insufficient image counts; expected to be rejected because every class folder requires at least 100 valid images.
- `data/test_samples/single_class_sample.zip`: 100 synthetic `Apple` images for testing one-folder novelty detection.
