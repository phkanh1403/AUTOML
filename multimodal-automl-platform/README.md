# Multimodal AutoML Platform — v2.1

Nền tảng AutoML chạy local, hỗ trợ **tabular classification/regression** và **image classification**. Khác với bản V13, backend mới thực hiện huấn luyện thật bằng scikit-learn, lưu model bằng `joblib`, đánh giá trên holdout độc lập và dùng chính model đã lưu khi suy luận. Không còn browser fallback, metric ngẫu nhiên, training memory giả, mock ONNX hoặc mock ZIP.

## 1. Kiến trúc

```text
Browser (web-ui)
   │  HTTP / FormData / JSON
   ▼
FastAPI backend (backend/app/main.py)
   ├── Tabular AutoML (backend/app/services/tabular.py)
   │     ├── CSV validation / cleaning
   │     ├── Imputation + RobustScaler + OneHotEncoder (inside Pipeline)
   │     ├── CV model selection: Logistic/RF/ExtraTrees or Ridge/RF/ExtraTrees
   │     ├── Holdout metrics + permutation importance
   │     └── Full-data refit for serving
   ├── Image AutoML (backend/app/services/image.py)
   │     ├── ZIP/path/image validation + class-folder labels
   │     ├── HOG texture + RGB histogram features
   │     ├── CV model selection: SVM/RF/ExtraTrees using the 64% reject rule
   │     └── Holdout metrics + grouped permutation importance
   └── Persistent model registry: data/models/<job_id>/
         ├── model.joblib
         ├── metadata.json
         └── processed_dataset.csv | processed_dataset.zip
```

The backend binds to `127.0.0.1` by default. The UI and API are served from the same origin; CORS is not opened globally.

## 2. Cài đặt và chạy

Yêu cầu: Python 3.10–3.12 (khuyến nghị Python 3.11), kết nối Internet ở lần cài dependencies đầu tiên.

### Windows

1. Giải nén toàn bộ ZIP.
2. Mở thư mục `multimodal-automl-platform`.
3. Chạy `scripts/start_all_services.bat`.
4. Truy cập `http://127.0.0.1:8000` nếu trình duyệt không tự mở.

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./scripts/start_all_services.sh
```

Hoặc chạy thủ công trên mọi hệ điều hành:

```bash
python -m pip install -r requirements.txt
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Không mở `web-ui/index.html` trực tiếp bằng `file://`; hãy truy cập qua FastAPI để API cùng origin hoạt động.

## 3. Huấn luyện dữ liệu bảng

- File đầu vào: CSV có header (giới hạn xử lý hiện tại: 250.000 dòng và 2.000 cột).
- `Target Column`: tên cột cần dự đoán, phân biệt đúng theo header sau khi trim khoảng trắng.
- `Task Type`: `Auto`, `Classification` hoặc `Regression`.
- Cần tối thiểu 12 dòng có target hợp lệ. Classification cần ít nhất 3 mẫu cho mỗi lớp để tách holdout và cross-validation.

Tiền xử lý:

1. Đọc CSV với encoding/delimiter fallback.
2. Chuẩn hóa giá trị thiếu phổ biến, loại dòng trùng lặp và dòng thiếu target.
3. Loại feature có tỷ lệ thiếu từ 70%, feature hằng và feature trùng giá trị.
4. Numeric: median imputation + missing indicator + RobustScaler.
5. Categorical: most-frequent imputation + OneHotEncoder với unknown category handling.
6. Toàn bộ imputer/encoder/scaler nằm trong scikit-learn `Pipeline` để tránh data leakage giữa train và validation.
7. Tách holdout stratified cho classification; so sánh thuật toán bằng cross-validation trên phần train; báo cáo metric trên holdout.
8. So sánh thêm với baseline đơn giản (prior/median); cảnh báo nếu model không vượt baseline trên holdout.
9. Sau đánh giá, pipeline được fit lại trên toàn bộ dữ liệu có nhãn để phục vụ inference.

`Auto` nhận diện target dạng text là classification; target số nguyên có số lớp thấp thường là classification; target numeric liên tục là regression. Khi bài toán có ý nghĩa nghiệp vụ rõ ràng, nên chọn task thủ công để tránh suy luận sai kiểu target.

## 4. Huấn luyện ảnh

**Tên folder bên trong ZIP chính là nhãn lớp.** Không cần khai báo hoặc nhập nhãn thủ công trên giao diện web. Hệ thống chấp nhận một hoặc nhiều folder nhãn. Mỗi folder phải có tối thiểu **100 ảnh JPEG/JPG/PNG hợp lệ, không trùng nội dung**. Ảnh hỏng và ảnh trùng không được tính vào số lượng tối thiểu.

Ví dụ nhiều lớp:

```text
dataset.zip
├── Apple/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ... (ít nhất 100 ảnh)
└── Banana/
    ├── image_001.jpg
    ├── image_002.jpg
    └── ... (ít nhất 100 ảnh)
```

Có thể đặt các folder lớp bên trong một folder gốc chung:

```text
dataset.zip
└── dataset/
    ├── Apple/ (ít nhất 100 ảnh)
    └── Banana/ (ít nhất 100 ảnh)
```

Quy tắc huấn luyện:

- **Từ 2 folder lớp trở lên:** huấn luyện phân loại đa lớp; nhãn lấy từ tên từng folder (ví dụ `Apple`, `Banana`).
- **Chỉ 1 folder lớp:** hệ thống dùng Isolation Forest ở chế độ one-class novelty detection để học phạm vi ảnh của folder đó; 80% ảnh được dùng để fit, 20% còn lại làm calibration boundary; ảnh đủ phù hợp được trả tên folder, ảnh không đủ phù hợp trả `Other`. Vì dataset một lớp không có ví dụ âm tính, không thể đánh giá độ chính xác phát hiện các lớp lạ một cách đầy đủ.
- **Mỗi folder tối thiểu 100 ảnh hợp lệ:** đây là điều kiện bắt buộc cho từng folder, không phải tổng số ảnh. Hệ thống hiện giới hạn 5.000 ảnh và 512 MB dữ liệu giải nén mỗi ZIP.
- Tên folder `Other` được dành riêng cho kết quả từ chối nhận diện nên không được dùng làm tên lớp huấn luyện.
- Mô hình hiện tại là baseline classical computer vision: resize nội bộ 64×64, HOG texture + RGB histogram, sau đó tự chọn SVM/Random Forest/Extra Trees cho đa lớp bằng F1-macro có áp dụng ngưỡng từ chối 64%. Chế độ một lớp dùng Isolation Forest. Ảnh gốc trong ZIP không bị sửa đổi.
- Mô hình này chạy được trên CPU, **không phải pretrained CNN/ViT**. Với ảnh thực tế có nền, góc chụp, ánh sáng và biến thiên lớn, cần dataset đa dạng, cân bằng và kiểm thử trên ảnh ngoài tập train; có thể nâng cấp bằng Transfer Learning (MobileNet/EfficientNet) khi có GPU.

### Quy tắc dự đoán ảnh và ngưỡng Other

Đối với mô hình nhiều lớp, backend lấy lớp có xác suất dự đoán cao nhất. Nếu confidence **lớn hơn 64%** thì trả về tên lớp; nếu confidence **nhỏ hơn hoặc bằng 64%** thì trả về `Other`. Ví dụ model phân loại `Apple` và `Banana`: Apple 72% → `Apple`; Banana 81% → `Banana`; Apple 48% và Banana 52% → `Other`. Kết quả API vẫn kèm `candidate_class`, confidence, `confidence_threshold` và xác suất từng lớp để dễ kiểm tra.

Với mô hình một lớp, confidence là điểm novelty/độ phù hợp với phân bố ảnh đã học, được ánh xạ về thang 0–1 để áp dụng cùng ngưỡng 64%; **không nên hiểu đây là xác suất thống kê đã calibration**.

## 5. Dự đoán và quản lý model

- Model sau train được lưu trong `data/models/<job_id>/model.joblib` cùng metadata và dataset đã xử lý.
- Tab Prediction lấy danh sách trực tiếp từ backend, không lưu weights hay training rows trong localStorage.
- Tabular nhận JSON array theo thứ tự features hoặc JSON object theo tên feature. Ví dụ:

```json
[12.5, "category A", 4]
```

```json
{"age": 12.5, "region": "category A", "count": 4}
```

- Image inference gửi file ảnh thật lên backend; không dùng tên file để suy diễn nhãn. Kết quả trả nhãn lớp chỉ khi confidence > 64%; trường hợp còn lại trả `Other`.
- Tabular classification trả lớp dự đoán và confidence. Image classification áp dụng ngưỡng từ chối 64%; xác suất model đa lớp chưa được calibration riêng. Regression trả giá trị dự đoán.
- Xóa model trong UI sẽ xóa model và artifacts tương ứng trên backend.

## 6. API endpoints

| Method | Endpoint | Mục đích |
|---|---|---|
| GET | `/api/health` | Health check |
| POST | `/api/gateway/submit` | Train `tabular` (multipart CSV) hoặc `image` (multipart ZIP) |
| POST | `/api/gateway/predict` | Tabular inference (JSON) |
| POST | `/api/gateway/predict/image` | Image inference (multipart) |
| GET | `/api/models` | Model registry |
| DELETE | `/api/models/{job_id}` | Xóa model và artifacts |
| GET | `/api/jobs/{job_id}/dataset` | Tải processed CSV/ZIP |
| GET | `/api/jobs/{job_id}/model` | Tải `joblib` model |

Swagger API docs: `http://127.0.0.1:8000/docs`.

## 7. Kiểm thử

```bash
python -m pytest backend/tests -q
```

Kiểm tra cú pháp backend:

```bash
python -m compileall backend
```

## 8. Giới hạn và lưu ý về độ chính xác

- AutoML không thể bảo đảm dự đoán đúng nếu target sai, dữ liệu nhiễu, leakage, domain shift hoặc dataset thiếu đại diện. Accuracy/F1/R²/MAE/RMSE là kết quả holdout trên chính dataset được upload, không phải cam kết chất lượng ngoài thực tế.
- Holdout chỉ là một lần chia dữ liệu; với dataset nhỏ, metric dao động. Cần xem phân bố lớp, baseline, confusion matrix/classification report và đánh giá ngoài tập dữ liệu độc lập.
- Với dữ liệu theo thời gian, bệnh viện, người dùng, lô sản xuất hoặc cùng đối tượng xuất hiện nhiều lần, random split có thể gây leakage. Hãy dùng group/time-based split phù hợp miền ứng dụng trước khi công bố kết quả.
- Joblib dùng pickle bên dưới; chỉ dùng artifacts do backend của chính bạn tạo. Không nạp model joblib từ nguồn không tin cậy.
- Backend hiện thiết kế cho chạy local, một máy và một người dùng. Không public trực tiếp ra Internet khi chưa bổ sung authentication, authorization, rate limiting và sandboxing.

### Sample files

- `data/test_samples/example_tabular.csv`: synthetic tabular classification example; target column is `plant_status`.
- `data/test_samples/valid_images.zip`: 200 synthetic sample images (100 in each of two class folders, `Apple` and `Banana`) for a quick image pipeline smoke test. It is not a real plant dataset and its scores must not be interpreted as real-world performance.
- `data/test_samples/invalid_count.zip`: intentionally insufficient image counts (3 images/class), expected to be rejected because each class folder needs at least 100 valid images.
- `data/test_samples/single_class_sample.zip`: 100 synthetic `Apple` images to test one-folder novelty detection.
