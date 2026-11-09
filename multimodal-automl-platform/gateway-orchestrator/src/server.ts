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
        if (!validation.valid) {
            return res.status(422).json({
                status: 'REJECTED_AT_GATEWAY',
                stage: 'Stage 1 & 3 Validation',
                reason: validation.reason
            });
        }
        const result = await dispatcher.dispatchPipeline({ modality, domain, mode, dataPayload });
        return res.json({ status: 'SUCCESS', validation, ...result });
    }

    if (modality === 'tabular') {
        const validation = validateTabularDataset(dataPayload, targetColumn);
        if (!validation.valid) {
            return res.status(422).json({
                status: 'REJECTED_AT_GATEWAY',
                stage: 'Stage 1 & 3 Validation',
                reason: validation.reason
            });
        }
        const result = await dispatcher.dispatchPipeline({
            modality,
            mode,
            targetColumn,
            droppedColumns: validation.droppedColumns,
            dataPayload: validation.cleanedRows
        });
        return res.json({
            status: 'SUCCESS',
            droppedColumns: validation.droppedColumns,
            retainedColumns: validation.retainedColumns,
            ...result
        });
    }
});

// Mock Prediction Endpoint
app.post('/api/gateway/predict', async (req, res) => {
    const { modality, inputs } = req.body;
    
    // Giả lập độ trễ suy luận
    await new Promise(r => setTimeout(r, 800));

    if (modality === 'image') {
        const score = Math.random() * 100;
        return res.json({
            status: 'SUCCESS',
            prediction: {
                class: score > 50 ? 'Bệnh lá đốm (Leaf Spot)' : 'Lá khỏe mạnh (Healthy)',
                confidence: (score > 50 ? score : 100 - score).toFixed(2) + '%'
            }
        });
    } else {
        const risk = Math.random() * 100;
        return res.json({
            status: 'SUCCESS',
            prediction: {
                target_value: risk.toFixed(2),
                risk_level: risk > 75 ? 'Cao (High)' : risk > 40 ? 'Trung bình (Medium)' : 'Thấp (Low)'
            }
        });
    }
});


const PORT = process.env.NODE_GATEWAY_PORT || 3000;
app.listen(PORT, () => {
    console.log(`[API Gateway] Listening on http://127.0.0.1:${PORT}`);
});
