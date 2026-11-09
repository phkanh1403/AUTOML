import { Request, Response, NextFunction } from 'express';
export function payloadAuthMiddleware(req: Request, res: Response, next: NextFunction) {
    if (!req.body.dataPayload) return res.status(400).json({ status: 'REJECTED', stage: 1, message: 'Thieu khoi du lieu (dataPayload).' });
    next();
}
