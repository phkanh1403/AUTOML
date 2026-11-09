import { Request, Response, NextFunction } from 'express';

export function payloadAuthMiddleware(req: Request, res: Response, next: NextFunction) {
    const apiKey = req.headers['x-api-key'] || 'local-dev-key';
    if (!apiKey) {
        return res.status(401).json({ status: 'ERROR', stage: 3, message: 'Missing Authentication Key' });
    }

    const { modality, dataPayload } = req.body;
    if (!modality || !['image', 'tabular'].includes(modality)) {
        return res.status(400).json({
            status: 'REJECTED',
            stage: 1,
            message: 'Modality khong hop le. Chi chap nhan "image" hoac "tabular".'
        });
    }

    if (!dataPayload) {
        return res.status(400).json({
            status: 'REJECTED',
            stage: 1,
            message: 'Thieu khoi du lieu dau vao (dataPayload).'
        });
    }

    next();
}
