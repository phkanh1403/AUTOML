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
app.post('/api/gateway/submit', payloadAuthMiddleware, async (req, res) => {
    const { modality, domain, mode = 'auto', targetColumn = 'target_label', dataPayload } = req.body;
    if (modality === 'image') {
        const validation = validateImageDataset(dataPayload);
        if (!validation.valid) return res.status(422).json({ status: 'REJECTED', reason: validation.reason });
        const result = await dispatcher.dispatchPipeline({ modality, domain, mode, dataPayload });
        return res.json({ status: 'SUCCESS', validation, ...result });
    }
    if (modality === 'tabular') {
        const validation = validateTabularDataset(dataPayload, targetColumn);
        if (!validation.valid) return res.status(422).json({ status: 'REJECTED', reason: validation.reason });
        const result = await dispatcher.dispatchPipeline({ modality, mode, targetColumn, droppedColumns: validation.droppedColumns, dataPayload: validation.cleanedRows });
        return res.json({ status: 'SUCCESS', ...result });
    }
});
app.post('/api/gateway/predict', async (req, res) => {
    const { modality, inputs } = req.body;
    await new Promise(r => setTimeout(r, 600));
    if (modality === 'image') {
        const score = Math.random() * 100;
        return res.json({ status: 'SUCCESS', prediction: { class: score > 50 ? 'Positive/Detected' : 'Negative/Normal', confidence: score.toFixed(2) + '%' } });
    } else {
        return res.json({ status: 'SUCCESS', prediction: { target_value: (Math.random() * 100).toFixed(2), risk_level: 'Medium' } });
    }
});
app.listen(3000, () => console.log(`Gateway running on 3000`));
