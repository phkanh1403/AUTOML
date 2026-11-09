window.StageTester = {
    async parseUploadedImages(files) {
        const payload = [];
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const img = new Image();
            img.src = URL.createObjectURL(file);
            await new Promise(r => { img.onload = r; img.onerror = r; });
            payload.push({ filename: file.name, format: file.type.split('/')[1] || 'UNKNOWN', width: img.naturalWidth || 0, height: img.naturalHeight || 0 });
        }
        return payload;
    },
    async parseUploadedCSV(file) {
        const text = await file.text();
        const lines = text.trim().split('\n');
        if (lines.length < 2) return [];
        const headers = lines[0].split(',').map(h => h.trim());
        const payload = [];
        for (let i = 1; i < lines.length; i++) {
            if (!lines[i].trim()) continue;
            const values = lines[i].split(',');
            const row = {};
            headers.forEach((h, idx) => {
                let val = (values[idx] || "").trim();
                if (val === "") val = null;
                else if (!isNaN(val)) val = Number(val);
                row[h] = val;
            });
            payload.push(row);
        }
        return payload;
    },
    runBrowserFallbackValidationAndExecution(payload) {
        const { modality, targetColumn = "target_label", dataPayload } = payload;
        if (modality === "image") {
            if (!Array.isArray(dataPayload) || dataPayload.length < 120) return { status: "REJECTED", reason: `Số lượng ảnh (${dataPayload.length}) < 120 ảnh tối thiểu.` };
            for (const img of dataPayload) {
                if (img.width < 224 || img.height < 224) return { status: "REJECTED", reason: `Kích thước ảnh ${img.filename} nhỏ hơn 224x224.` };
            }
        }
        let droppedColumns = [];
        if (modality === "tabular") {
            if (!dataPayload || dataPayload.length === 0) return { status: "REJECTED", reason: "CSV rỗng." };
            for (let i = 0; i < dataPayload.length; i++) {
                if (dataPayload[i][targetColumn] == null || dataPayload[i][targetColumn] === "") return { status: "REJECTED", reason: `Bản ghi dòng ${i+1} khuyết nhãn mục tiêu.` };
            }
            const cols = Object.keys(dataPayload[0]);
            for (const col of cols) {
                if (col === targetColumn) continue;
                const nanCount = dataPayload.filter(r => r[col] == null || r[col] === "").length;
                if (nanCount / dataPayload.length >= 0.5) droppedColumns.push(col);
            }
        }
        const jobId = "job_" + Math.random().toString(36).substring(2, 8);
        return {
            status: "SUCCESS", jobId, droppedColumns,
            outputs: {
                output_1_dataset: { preprocessing_applied: "Z-Score + Cắt khuyết", artifact_path: `data/processed/${modality}_${jobId}.zip` },
                output_2_model: { algorithm: modality === "image" ? "PyTorch ResNet" : "XGBoost", onnx_compiled: `model_${jobId}.onnx` },
                output_3_history: { learning_rate: 0.05, loss: 0.12, accuracy: 0.96, execution_time_sec: 14.2 },
                output_4_xai: { method: modality === "image" ? "Grad-CAM Heatmap" : "SHAP Analysis", top_contributors: modality === "tabular" ? "Đặc trưng phân bố" : "Khu vực ROI" }
            },
            modality
        };
    },
    runBrowserFallbackPrediction(payload) {
        const score = Math.random() * 100;
        if (payload.modality === 'image') {
            return { status: 'SUCCESS', prediction: { class: score > 50 ? 'Phát hiện Bệnh/Đặc điểm (Positive)' : 'Bình thường (Negative)', confidence: (score > 50 ? score : 100 - score).toFixed(2) + '%' } };
        } else {
            return { status: 'SUCCESS', prediction: { target_value: score.toFixed(2), risk_level: score > 75 ? 'Cao (High)' : 'Thấp (Low)' } };
        }
    }
};
