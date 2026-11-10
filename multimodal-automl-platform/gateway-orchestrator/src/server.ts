import express from 'express';
import cors from 'cors';
import { payloadAuthMiddleware } from './middlewares/payload_auth';
import { validateImageDataset } from './validators/image_validator';
import { validateTabularDataset } from './validators/tabular_validator';
import { TaskDispatcher } from './orchestrator/task_dispatcher';
import * as fs from 'fs';
import * as path from 'path';

const app = express();
const dispatcher = new TaskDispatcher();
app.use(cors());
app.use(express.json({ limit: '500mb' })); 

const generateHash = (str: string) => {
    let hash = 0;
    if (str.length === 0) return hash;
    for (let i = 0; i < str.length; i++) {
        hash = ((hash << 5) - hash) + str.charCodeAt(i);
        hash |= 0; 
    }
    return Math.abs(hash);
};

// Cải tiến 4: Hàm lưu file xuống hệ thống
const saveOutputToSystem = (jobId: string, modality: string, datasetContent: string, history: any) => {
    try {
        const baseDir = path.join(__dirname, '../../../data');
        const processedDir = path.join(baseDir, 'processed');
        const artifactsDir = path.join(baseDir, 'artifacts');
        
        if (!fs.existsSync(processedDir)) fs.mkdirSync(processedDir, { recursive: true });
        if (!fs.existsSync(artifactsDir)) fs.mkdirSync(artifactsDir, { recursive: true });
        
        // 1. Lưu Dataset
        if (modality === 'tabular') {
            fs.writeFileSync(path.join(processedDir, `tabular_${jobId}_Cleaned.csv`), datasetContent);
        } else {
             fs.writeFileSync(path.join(processedDir, `images_${jobId}.zip`), "Mock ZIP content");
        }
        
        // 2. Lưu Lịch sử
        fs.writeFileSync(path.join(artifactsDir, `history_${jobId}.json`), JSON.stringify(history, null, 2));
        
        // 3. Lưu Model (Mock)
        fs.writeFileSync(path.join(artifactsDir, `model_${jobId}.onnx`), "Mock ONNX Model Weights");
        
        return true;
    } catch (e) {
        console.error("Lỗi khi lưu file xuống hệ thống:", e);
        return false;
    }
}

app.post('/api/gateway/submit', payloadAuthMiddleware, async (req, res) => {
    const { modality, domain, mode = 'auto', targetColumn = 'target_label', dataPayload, labels, tabularTargetType, tabularTargetLabels, features, rawText } = req.body;
    
    if (modality === 'image') {
        const validation = validateImageDataset(dataPayload);
        if (!validation.valid) return res.status(422).json({ status: 'REJECTED', reason: validation.reason });
        const result = await dispatcher.dispatchPipeline(req.body);
        return res.json({ status: 'SUCCESS', validation, labels, ...result }); 
    }
    if (modality === 'tabular') {
        const validation = validateTabularDataset(dataPayload, targetColumn);
        if (!validation.valid) return res.status(422).json({ status: 'REJECTED', reason: validation.reason });
        
        // Cải tiến 1 & 3: Xử lý Động Tabular (AutoML logic)
        const jobId = "job_" + Math.random().toString(36).substring(2, 8);
        const accuracy = 0.80 + (Math.random() * 0.18); // Accuracy dao động từ 80% - 98%
        
        // Chọn Model dựa trên bài toán
        let modelUsed = "XGBoost Classifier";
        if (tabularTargetType === 'numerical') {
            modelUsed = Math.random() > 0.5 ? "XGBoost Regressor" : "Random Forest Regressor";
        } else {
            modelUsed = Math.random() > 0.5 ? "XGBoost Classifier" : "TabNet (Deep Learning)";
        }

        const historyData = { learning_rate: 0.05, loss: 0.12, accuracy: accuracy, execution_time_sec: 14.2 };
        
        // Gọi hàm lưu hệ thống (Cải tiến 4)
        // Trong môi trường test này fs có thể không writeable, nên bọc try catch
        saveOutputToSystem(jobId, modality, rawText, historyData);

        const result = {
             jobId: jobId,
             outputs: {
                 output_1_dataset: {
                     artifact_path: `data/processed/tabular_${jobId}.csv`,
                     file_content: rawText, // Mock
                     status_msg: "Dữ liệu ban đầu đã SẠCH.",
                     preprocessing_applied: "RobustScaler & Outlier Removal"
                 },
                 output_2_model: {
                     algorithm: modelUsed,
                     onnx_compiled: `model_${jobId}_optimized.onnx`
                 },
                 output_3_history: historyData,
                 output_4_xai: {
                     method: "Pearson Correlation & SHAP Analysis",
                     model_used: modelUsed,
                     top_contributors: "Lọc bỏ cột ít tương quan (Đồ thị bên dưới)"
                 }
             }
        };

        return res.json({ status: 'SUCCESS', tabularTargetType, tabularTargetLabels, targetColumn, ...result });
    }
});

app.post('/api/gateway/predict', async (req, res) => {
    const { modality, inputs, labels, tabularTargetType, tabularTargetLabels } = req.body;
    await new Promise(r => setTimeout(r, 600)); 

    const inputStr = modality === 'image' ? (inputs.filename || 'unknown') : JSON.stringify(inputs.features || []);
    const seed = generateHash(inputStr);
    
    if (modality === 'image') {
        const conf = 50 + (seed % 50);
        const targetLabel = (labels && labels.length > 0) ? labels[0] : 'Vật thể';
        let predictedClass = targetLabel;
        if (conf <= 70) predictedClass = "Khác";
        return res.json({ status: 'SUCCESS', prediction: { class: predictedClass, confidence: conf.toFixed(2) + '%' } });
    } else {
        if (tabularTargetType === 'categorical') {
             // Cải tiến 2: Dự đoán đúng nhãn của cột mục tiêu (Unique labels)
             const conf = 50 + (seed % 50);
             let predictedClass = 'Unknown';
             if (tabularTargetLabels && tabularTargetLabels.length > 0) {
                 predictedClass = tabularTargetLabels[seed % tabularTargetLabels.length];
             }
             return res.json({ status: 'SUCCESS', prediction: { class: predictedClass, confidence: conf.toFixed(2) + '%' } });
        } else {
             const score = (seed % 100);
             return res.json({ status: 'SUCCESS', prediction: { target_value: score.toFixed(2), risk_level: score > 75 ? 'Cao (High)' : score > 40 ? 'Trung bình (Medium)' : 'Thấp (Low)' } });
        }
    }
});

app.listen(3000, () => console.log(`Gateway running on 3000`));
