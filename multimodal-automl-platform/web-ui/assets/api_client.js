(() => {
    async function parseResponse(response) {
        let body;
        try { body = await response.json(); }
        catch { body = { detail: `Server trả về HTTP ${response.status} nhưng nội dung không phải JSON.` }; }
        if (!response.ok) {
            return {
                status: "ERROR",
                reason: body.detail || body.reason || `Yêu cầu thất bại (HTTP ${response.status}).`,
                httpStatus: response.status
            };
        }
        return body;
    }

    window.AutoMLApiClient = {
        async health() {
            const response = await fetch("/api/health");
            return parseResponse(response);
        },

        async submitToGateway(payload) {
            const form = new FormData();
            form.append("modality", payload.modality);
            form.append("file", payload.file, payload.file.name);
            if (payload.modality === "tabular") {
                form.append("target_column", payload.targetColumn || "target_label");
                form.append("task_type", payload.taskType || "auto");
            }
            const response = await fetch("/api/gateway/submit", { method: "POST", body: form });
            return parseResponse(response);
        },

        async predictTabular(payload) {
            const response = await fetch("/api/gateway/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            return parseResponse(response);
        },

        async predictImage(modelId, file) {
            const form = new FormData();
            form.append("model_id", modelId);
            form.append("file", file, file.name);
            const response = await fetch("/api/gateway/predict/image", { method: "POST", body: form });
            return parseResponse(response);
        },

        async listModels() {
            const response = await fetch("/api/models");
            return parseResponse(response);
        },

        async deleteModel(modelId) {
            const response = await fetch(`/api/models/${encodeURIComponent(modelId)}`, { method: "DELETE" });
            return parseResponse(response);
        }
    };
})();
