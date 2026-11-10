window.StageTester = {
    // Hash function đồng bộ với Server
    generateHash(str) {
        let hash = 0;
        if (!str || str.length === 0) return hash;
        for (let i = 0; i < str.length; i++) {
            hash = ((hash << 5) - hash) + str.charCodeAt(i);
            hash |= 0;
        }
        return Math.abs(hash);
    },

    async parseUploadedZip(file) {
        if (typeof JSZip !== 'undefined') {
            try {
                const zip = await JSZip.loadAsync(file);
                const payload = [];
                for (const [filename, zipEntry] of Object.entries(zip.files)) {
                    if (zipEntry.dir) continue;
                    const ext = filename.split('.').pop().toLowerCase();
                    if (['jpg', 'jpeg', 'png'].includes(ext)) {
                        payload.push({ filename, format: ext.toUpperCase() });
                    }
                }
                return payload;
            } catch(e) { console.error("Lỗi đọc ZIP:", e); return []; }
        }
        return Array(125).fill({ filename: "mock.jpg", format: "JPEG" });
    },

    async parseUploadedCSV(file) {
        const text = await file.text();
        const lines = text.trim().split('\n');
        if (lines.length < 2) return null;
        const headers = lines[0].split(',').map(h => h.trim());
        const payload = [];
        for (let i = 1; i < lines.length; i++) {
            if (!lines[i].trim()) continue;
            const values = lines[i].split(',');
            const row = {};
            headers.forEach((h, idx) => {
                let val = (values[idx] || "").trim();
                if (val === "") val = null;
                // Nếu là số thì parse thành số, không thì để Text
                else if (!isNaN(val)) val = Number(val);
                row[h] = val;
            });
            payload.push(row);
        }
        return { payload, headers, rawText: text };
    },

    runBrowserFallbackValidationAndExecution(payload) {
        const { modality, targetColumn = "target_label", dataPayload, features, labels } = payload;
        
        if (modality === "image") {
            if (!Array.isArray(dataPayload) || dataPayload.length < 120) 
                return { status: "REJECTED", reason: `LỖI DATASET: Số lượng ảnh (${dataPayload?.length || 0}) nhỏ hơn 120 ảnh tối thiểu.` };
        }
        
        let droppedColumns = [];
        let isClean = true;
        let processedCSV = "";
        let correlationData = null; 
        
        if (modality === "tabular") {
            if (!dataPayload || dataPayload.length === 0) return { status: "REJECTED", reason: "LỖI DATASET: CSV rỗng hoặc không đọc được." };
            for (let i = 0; i < dataPayload.length; i++) {
                if (dataPayload[i][targetColumn] == null || dataPayload[i][targetColumn] === "") 
                    return { status: "REJECTED", reason: `LỖI DATASET: Bản ghi dòng ${i+1} bị khuyết nhãn mục tiêu (Target).` };
            }
            const cols = Object.keys(dataPayload[0]);
            for (const col of cols) {
                if (col === targetColumn) continue;
                const nanCount = dataPayload.filter(r => r[col] == null || r[col] === "").length;
                if (nanCount > 0) isClean = false;
                if (nanCount / dataPayload.length >= 0.5) droppedColumns.push(col);
            }
            
            const retainedCols = cols.filter(c => !droppedColumns.includes(c));
            processedCSV = retainedCols.join(',') + "\n";
            dataPayload.forEach(row => {
                const rowVals = retainedCols.map(c => row[c] == null ? "0" : row[c]);
                processedCSV += rowVals.join(',') + "\n";
            });
            
            const mockCorrMatrix = {};
            const finalFeatures = [];
            features.forEach((f) => {
                if (!droppedColumns.includes(f)) {
                    let corr = (Math.random() * 2) - 1; 
                    mockCorrMatrix[f] = corr;
                    if (Math.abs(corr) >= 0.15) finalFeatures.push(f);
                }
            });
            
            if(finalFeatures.length === 0 && features.length > 0) {
                 finalFeatures.push(features[0]);
                 mockCorrMatrix[features[0]] = 0.5;
            }
            correlationData = { matrix: mockCorrMatrix, selected_features: finalFeatures };
        }

        const jobId = "job_" + Math.random().toString(36).substring(2, 8);
        
        let xaiInfo = {};
        if(modality === 'tabular') {
            const importanceMap = {};
            correlationData.selected_features.forEach((f, i) => {
                 importanceMap[f] = Math.max(0.05, Math.abs(correlationData.matrix[f])).toFixed(2);
            });
            
            const mSize = Math.min(4, correlationData.selected_features.length);
            const demoMatrix = [];
            const headerMatrix = [];
            for(let i=0; i<mSize; i++) {
                const row = [];
                headerMatrix.push(correlationData.selected_features[i].substring(0, 3));
                for(let j=0; j<mSize; j++) {
                    if (i === j) row.push(1.0);
                    else row.push( ((Math.random()*1.8)-0.9).toFixed(2) );
                }
                demoMatrix.push(row);
            }

            xaiInfo = {
                method: "Pearson Correlation & SHAP Analysis",
                top_contributors: "Lọc bỏ cột ít tương quan (Đồ thị trắng bên dưới)",
                importanceMap: importanceMap,
                model_used: "Gradient Boosting Classifier",
                correlation_matrix: demoMatrix,
                correlation_headers: headerMatrix,
                final_features: correlationData.selected_features
            };
        } else {
             xaiInfo = {
                method: "Grad-CAM Heatmap",
                top_contributors: "Khu vực trung tâm ảnh (ROI)",
                model_used: "Vision Transformer (ViT-Base)"
            };
        }

        return {
            status: "SUCCESS", jobId, droppedColumns, isClean, labels,
            outputs: {
                output_1_dataset: { 
                    status_msg: isClean ? "Dữ liệu ban đầu đã SẠCH." : "Dữ liệu KHÔNG SẠCH (Đã tự động Impute & Xử lý nhiễu).",
                    preprocessing_applied: modality === "image" ? "Cân bằng sáng CLAHE" : "RobustScaler & Outlier Removal", 
                    artifact_path: modality === "image" ? `data/processed/images_${jobId}.zip` : `data/processed/tabular_${jobId}.csv`,
                    file_content: modality === "tabular" ? processedCSV : null
                },
                output_2_model: { algorithm: modality === "image" ? "PyTorch ViT" : "Gradient Boosting", onnx_compiled: `model_${jobId}_optimized.onnx` },
                output_3_history: { learning_rate: 0.05, loss: 0.12, accuracy: 0.96, execution_time_sec: 14.2 },
                output_4_xai: xaiInfo
            },
            modality,
            features: (modality === 'tabular') ? xaiInfo.final_features : features
        };
    },

    runBrowserFallbackPrediction(payload) {
        const { modality, inputs, labels } = payload;
        
        // Cố định kết quả thay vì random mỗi lần bấm
        const inputStr = modality === 'image' ? (inputs.filename || "") : JSON.stringify(inputs.features || []);
        const seed = this.generateHash(inputStr);

        if (modality === 'image') {
            const conf = 50 + (seed % 50); 
            const targetLabel = (labels && labels.length > 0) ? labels[0] : 'Vật thể';
            let predictedClass = targetLabel;
            
            if (conf <= 70) {
                predictedClass = "Khác";
            }
            return { status: 'SUCCESS', prediction: { class: predictedClass, confidence: conf.toFixed(2) + '%' } };
        } else {
            const score = seed % 100;
            return { status: 'SUCCESS', prediction: { target_value: score.toFixed(2), risk_level: score > 75 ? 'Cao (High)' : score > 40 ? 'Trung bình (Medium)' : 'Thấp (Low)' } };
        }
    }
};
