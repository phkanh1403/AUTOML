import express from 'express';
import cors from 'cors';
import { payloadAuthMiddleware } from './middlewares/payload_auth';
import { validateImageDataset } from './validators/image_validator';
import { validateTabularDataset } from './validators/tabular_validator';
import { TaskDispatcher } from './orchestrator/task_dispatcher';

const app = express();
const dispatcher = new TaskDispatcher();

app.use(cors());
app.use(express.json({ limit: '50mb' }));

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

const PORT = process.env.NODE_GATEWAY_PORT || 3000;
app.listen(PORT, () => {
    console.log(`[API Gateway] Listening on http://127.0.0.1:${PORT}`);
});
