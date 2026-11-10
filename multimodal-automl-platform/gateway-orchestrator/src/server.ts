import express from 'express';
import cors from 'cors';
import { payloadAuthMiddleware } from './middlewares/payload_auth';
import { validateImageDataset } from './validators/image_validator';
import { validateTabularDataset } from './validators/tabular_validator';
import { TaskDispatcher } from './orchestrator/task_dispatcher';

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

app.post('/api/gateway/submit', payloadAuthMiddleware, async (req, res) => {
    // Luồng này ở backend sẽ gọi dispatcher thật. Ở Mock, ta skip cho ngắn gọn vì FE đã bọc logic fallback.
    return res.json({ status: 'SUCCESS' });
});

app.post('/api/gateway/predict', async (req, res) => {
    const { modality, inputs, labels, tabularTargetType, tabularTargetLabels, trainingMemory } = req.body;
    await new Promise(r => setTimeout(r, 600)); 

    if (modality === 'image') {
        const inputStr = inputs.filename || 'unknown';
        const seed = generateHash(inputStr);
        const conf = 50 + (seed % 50);
        const targetLabel = (labels && labels.length > 0) ? labels[0] : 'Vật thể';
        let predictedClass = targetLabel;
        
        // CẬP NHẬT 1: Ngưỡng 60%
        if (conf <= 60) predictedClass = "Khác";
        return res.json({ status: 'SUCCESS', prediction: { class: predictedClass, confidence: conf.toFixed(2) + '%' } });
    } else {
        const inputKey = (inputs.features || []).join('|');
        
        // CẬP NHẬT 2: KIỂM TRA BỘ NHỚ HUẤN LUYỆN (MEMORIZATION)
        if (trainingMemory && trainingMemory[inputKey] !== undefined) {
             const exactAnswer = trainingMemory[inputKey];
             if (tabularTargetType === 'categorical') {
                 return res.json({ status: 'SUCCESS', prediction: { class: exactAnswer, confidence: '99.80%' } });
             } else {
                 return res.json({ status: 'SUCCESS', prediction: { target_value: Number(exactAnswer).toFixed(2), risk_level: 'Khớp với tập huấn luyện' } });
             }
        }
        
        // Nếu không có trong dữ liệu huấn luyện, dự đoán bằng Hash
        const seed = generateHash(JSON.stringify(inputs.features || []));
        if (tabularTargetType === 'categorical') {
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
