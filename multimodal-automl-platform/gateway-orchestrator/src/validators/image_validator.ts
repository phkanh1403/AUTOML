export interface ImageMeta { filename: string; format: string; }
export interface ValidationResult { valid: boolean; reason?: string; validCount?: number; }
export function validateImageDataset(images: ImageMeta[]): ValidationResult {
    if (!Array.isArray(images) || images.length < 120) {
        return { valid: false, reason: `Vi phạm số lượng: Cần tối thiểu 120 ảnh (Hiện có: ${images ? images.length : 0} ảnh).` };
    }
    const allowedFormats = ['JPEG', 'JPG', 'PNG'];
    for (const img of images) {
        let fmt = (img.format || '').toUpperCase();
        if (fmt === "JPG") fmt = "JPEG";
        if (!allowedFormats.includes(fmt)) return { valid: false, reason: `Định dạng ảnh "${img.filename}" không hợp lệ. Chỉ hỗ trợ JPEG/PNG.` };
        // ĐÃ XÓA KIỂM TRA KÍCH THƯỚC ẢNH
    }
    return { valid: true, validCount: images.length };
}
