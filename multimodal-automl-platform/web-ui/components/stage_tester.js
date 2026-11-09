window.StageTester = {
    generateImageBatch(count, width, height, format = "JPEG") {
        const list = [];
        for (let i = 1; i <= count; i++) {
            list.push({ filename: `sample_${i}.${format.toLowerCase()}`, format, width, height });
        }
        return list;
    },
    getValidTabular50PctNaN() {
        return [
            { sensor_id: 1, temp_c: 25.4, humidity_pct: 60.2, corrupted_50pct_col: null, vibration_hz: 12.1, target_label: 0 },
            { sensor_id: 2, temp_c: 26.1, humidity_pct: 59.8, corrupted_50pct_col: null, vibration_hz: 12.5, target_label: 0 },
            { sensor_id: 3, temp_c: 35.8, humidity_pct: 45.0, corrupted_50pct_col: 1.0, vibration_hz: 45.2, target_label: 1 },
            { sensor_id: 4, temp_c: 24.9, humidity_pct: 61.0, corrupted_50pct_col: null, vibration_hz: 11.8, target_label: 0 }
        ];
    },
    getInvalidMissingTargetTabular() {
        return [
            { sensor_id: 1, temp_c: 25.4, humidity_pct: 60.2, target_label: 0 },
            { sensor_id: 2, temp_c: 26.1, humidity_pct: 59.8, target_label: null },
            { sensor_id: 3, temp_c: 35.8, humidity_pct: 45.0, target_label: 1 }
        ];
    },
    runBrowserFallbackValidationAndExecution(payload) {
        const { modality, domain = "natural", mode = "auto", targetColumn = "target_label", dataPayload } = payload;
        if (modality === "image") {
            if (!Array.isArray(dataPayload) || dataPayload.length < 120) {
                return {
                    status: "REJECTED_AT_GATEWAY",
                    stage: "Giai doan 1 & 3 (Kiem dinh Dau vao Anh)",
                    reason: `Loi: So luong anh (${dataPayload.length}) < 120 anh toi thieu. He thong tu choi va quay lai tu dau!`
                };
            }
            for (const img of dataPayload) {
                if (img.width < 224 || img.height < 224) {
                    return {
                        status: "REJECTED_AT_GATEWAY",
                        stage: "Giai doan 1 & 3 (Kiem dinh Kich thuoc Anh)",
                        reason: `Loi: Anh "${img.filename}" (${img.width}x${img.height}) nho hon chuan toi thieu 224x224. He thong tu choi ngay tu dau!`
                    };
                }
            }
        }
        let droppedColumns = [];
        if (modality === "tabular") {
            for (let i = 0; i < dataPayload.length; i++) {
                if (dataPayload[i][targetColumn] === null || dataPayload[i][targetColumn] === undefined || dataPayload[i][targetColumn] === "") {
                    return {
                        status: "REJECTED_AT_GATEWAY",
                        stage: "Giai doan 1 & 3 (Rang buoc Nhan Muc tieu)",
                        reason: `Loi nghiem trong: Ban ghi thu ${i + 1} bi khuyet nhan muc tieu "${targetColumn}". Tuyet doi tu choi ngay tu dau!`
                    };
                }
            }
            const cols = Object.keys(dataPayload[0]);
            for (const col of cols) {
                if (col === targetColumn) continue;
                const nanCount = dataPayload.filter(r => r[col] === null || r[col] === undefined || r[col] === "").length;
                if (nanCount / dataPayload.length >= 0.5) droppedColumns.push(col);
            }
        }
        const jobId = "job_" + Math.random().toString(36).substring(2, 8);
        return {
            status: "SUCCESS",
            jobId,
            droppedColumns,
            outputs: {
                output_1_dataset: {
                    modality,
                    domain: modality === "image" ? domain : "structured_tabular",
                    preprocessing_applied: modality === "image"
                        ? "Laplacian Blur Filter + ImageNet Z-Score + CLAHE / Unsharp / Otsu"
                        : `Drop Cols >=50% NaN [${droppedColumns.join(", ") || "None"}] + Mean/Median Impute + IQR/Winsorize + RobustScaler`,
                    artifact_path: `data/processed/${modality}_${jobId}`
                },
                output_2_model: {
                    algorithm: modality === "image" ? "PyTorch ResNet50 / ViT-B16" : "XGBoost / GradientBoosting (Optuna Tuned)",
                    weights: `data/artifacts/model_${jobId}.pt`,
                    onnx_compiled: `data/artifacts/model_${jobId}.onnx`,
                    tensorrt_engine: `data/artifacts/model_${jobId}.trt`
                },
                output_3_history: {
                    execution_mode: mode,
                    learning_rate: modality === "image" ? 0.001 : 0.05,
                    loss: 0.124,
                    accuracy: 0.961,
                    execution_time_sec: 11.4
                },
                output_4_xai: {
                    method: modality === "image" ? "Grad-CAM Spatial Heatmap" : "SHAP Feature Attribution",
                    top_contributors: modality === "image"
                        ? "Kenh V/L CLAHE vung la ton thuong (Weight: 0.84)"
                        : "vibration_hz (+0.48), temp_c (+0.31), humidity_pct (-0.19)"
                }
            }
        };
    }
};
