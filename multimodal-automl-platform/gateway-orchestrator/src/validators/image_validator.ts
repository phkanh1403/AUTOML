export interface ImageMeta {
    filename: string;
    format: string;
    width: number;
    height: number;
}
export interface ValidationResult { valid: boolean; reason?: string; validCount?: number; }
export function validateImageDataset(images: ImageMeta[]): ValidationResult {
    if (!Array.isArray(images) || images.length < 120) {
        return { valid: false, reason: `Vi pham rang buoc so luong: Can toi thieu 120 anh (Hien co: ${images ? images.length : 0} anh). He thong tu choi.` };
    }
    const allowedFormats = ['JPEG', 'JPG', 'PNG'];
    for (const img of images) {
        let fmt = (img.format || '').toUpperCase();
        if (fmt === "JPG") fmt = "JPEG";
        if (!allowedFormats.includes(fmt)) {
            return { valid: false, reason: `Dinh dang anh "${img.filename}" khong hop le (${img.format}). Chi ho tro JPEG/PNG.` };
        }
        if (img.width < 224 || img.height < 224) {
            return { valid: false, reason: `Kich thuoc anh "${img.filename}" (${img.width}x${img.height}) nho hon 224x224. He thong tu choi.` };
        }
    }
    return { valid: true, validCount: images.length };
}
