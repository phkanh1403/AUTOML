(() => {
    async function parseResponse(response) {
        let body;
        try { body = await response.json(); }
        catch { body = { detail: `The server returned HTTP ${response.status} with a non-JSON response.` }; }
        if (!response.ok) {
            const detail = typeof body.detail === "string"
                ? body.detail
                : (body.detail?.message || body.reason || `Request failed (HTTP ${response.status}).`);
            return {
                status: "ERROR",
                reason: detail,
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
        },

        async deleteAllModels() {
            const response = await fetch("/api/models", { method: "DELETE" });
            return parseResponse(response);
        }
    };
})();
