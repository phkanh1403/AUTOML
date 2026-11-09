export interface TabularValidationResult {
    valid: boolean;
    reason?: string;
    droppedColumns: string[];
    retainedColumns: string[];
    cleanedRows: Record<string, any>[];
}

export function validateTabularDataset(
    rows: Record<string, any>[],
    targetColumn: string
): TabularValidationResult {
    if (!Array.isArray(rows) || rows.length === 0) {
        return {
            valid: false,
            reason: 'Tap du lieu bang rong.',
            droppedColumns: [],
            retainedColumns: [],
            cleanedRows: []
        };
    }

    for (let i = 0; i < rows.length; i++) {
        const targetVal = rows[i][targetColumn];
        if (targetVal === null || targetVal === undefined || targetVal === '' || Number.isNaN(targetVal)) {
            return {
                valid: false,
                reason: `Vi pham nghiem trong: Ban ghi thu ${i + 1} bi khuyet nhan muc tieu ("${targetColumn}"). Tuyet doi tu choi ngay tu dau!`,
                droppedColumns: [],
                retainedColumns: [],
                cleanedRows: []
            };
        }
    }

    const allColumns = Object.keys(rows[0]);
    const droppedColumns: string[] = [];
    const retainedColumns: string[] = [];
    const totalRows = rows.length;

    for (const col of allColumns) {
        if (col === targetColumn) {
            retainedColumns.push(col);
            continue;
        }
        let missingCount = 0;
        for (const row of rows) {
            const val = row[col];
            if (val === null || val === undefined || val === '' || Number.isNaN(val)) {
                missingCount++;
            }
        }
        if (missingCount / totalRows >= 0.5) {
            droppedColumns.push(col);
        } else {
            retainedColumns.push(col);
        }
    }

    const cleanedRows = rows.map(row => {
        const filtered: Record<string, any> = {};
        for (const col of retainedColumns) {
            filtered[col] = row[col];
        }
        return filtered;
    });

    return {
        valid: true,
        droppedColumns,
        retainedColumns,
        cleanedRows
    };
}
