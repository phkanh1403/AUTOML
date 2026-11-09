export interface ImageMeta {
    filename: string;
    format: string;
    width: number;
    height: number;
}

export interface ValidationResult {
    valid: boolean;
    reason?: string;
    validCount?: number;
}

export function validateImageDataset(images: ImageMeta[]): ValidationResult {
    if (!Array.isArray(images) || images.length < 120) {
        return {
            valid: false,
            reason: `Vi pham rang buoc so luong: Can toi thieu 120 anh (Hien co: ${images ? images.length : 0} anh). He thong tu choi va quay lai tu dau.`
        };
    }

    const allowedFormats = ['JPEG', 'JPG', 'PNG'];
    for (const img of images) {
        const fmt = img.format.toUpperCase();
        if (!allowedFormats.includes(fmt)) {
            return {
                valid: false,
                reason: `Dinh dang anh khong hop le o tep "${img.filename}" (${img.format}). Chi ho tro JPEG va PNG.`
            };
        }
        if (img.width < 224 || img.height < 224) {
            return {
                valid: false,
                reason: `Kich thuoc anh "${img.filename}" (${img.width}x${img.height}) nho hon muc toi thieu 224x224. He thong tu choi ngay tu dau.`
            };
        }
    }

    return { valid: true, validCount: images.length };
}
