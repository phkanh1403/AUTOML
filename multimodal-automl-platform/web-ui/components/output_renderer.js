(() => {
    const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    })[char]);

    const percent = (value) => Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(2)}%` : "N/A";
    const number = (value, digits = 4) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "N/A";
    const formatBytes = (value) => {
        const bytes = Number(value);
        if (!Number.isFinite(bytes) || bytes < 0) return "N/A";
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / 1024 ** 2).toFixed(2)} MB`;
    };

    window.OutputRenderer = {
        render(response) {
            const log = document.getElementById("pipeline-log");
            if (!response || response.status !== "SUCCESS") {
                const reason = response?.reason || "The backend did not return a valid training result.";
                log.textContent = `TRAINING FAILED\n${reason}`;
                document.getElementById("out-dataset").textContent = "No dataset artifact";
                document.getElementById("out-model").textContent = "No trained model";
                document.getElementById("out-history").textContent = "No evaluation metrics";
                document.getElementById("out-xai").textContent = "No explainability output";
                document.getElementById("test-predict-prompt").style.display = "none";
                return;
            }

            const outputs = response.outputs || {};
            const dataset = outputs.output_1_dataset || {};
            const model = outputs.output_2_model || {};
            const history = outputs.output_3_history || {};
            const xai = outputs.output_4_xai || {};
            const metrics = history.metrics || {};
            const artifact = response.modelArtifact || {};

            const selectionSummary = response.imageMode === "one_class_novelty"
                ? "The novelty detector was fitted on the single supplied image class (classification CV is not applicable)."
                : `The algorithm was selected using ${history.cv_folds || "—"}-fold cross-validation on the training split.`;
            log.textContent = `TRAINING COMPLETED\nJob: ${response.jobId}\n${selectionSummary}\nThe trained model has been saved to backend storage.`;
            if (response.warnings?.length) log.textContent += `\n\nWarnings:\n- ${response.warnings.join("\n- ")}`;

            const datasetUrl = response.download?.dataset || "#";
            const modelUrl = response.download?.model || "#";
            document.getElementById("out-dataset").innerHTML =
                `<b>Status:</b> ${escapeHtml(dataset.status_msg)}<br>` +
                `<b>Preprocessing:</b> ${escapeHtml(dataset.preprocessing_applied)}<br>` +
                `<a class="download-btn" href="${escapeHtml(datasetUrl)}">Download processed dataset</a>`;

            document.getElementById("out-model").innerHTML =
                `<b>Algorithm:</b> ${escapeHtml(model.algorithm)}<br>` +
                `<b>Artifact:</b> ${escapeHtml(artifact.file_name || "model.joblib")} (${escapeHtml(formatBytes(artifact.size_bytes))})<br>` +
                `<b>Format:</b> ${escapeHtml(model.format || "joblib")}<br>` +
                `<a class="download-btn" href="${escapeHtml(modelUrl)}">Download trained model (.joblib)</a>`;

            let metricHtml = `<b>Algorithm:</b> ${escapeHtml(model.algorithm)}<br>`;
            metricHtml += `<b>Primary metric:</b> ${escapeHtml(history.primary_metric || "—")} = ${escapeHtml(number(history.primary_value, 4))}<br>`;
            metricHtml += `<b>CV ${escapeHtml(history.cv_metric || "score")}:</b> ${escapeHtml(number(history.cv_score, 4))} ± ${escapeHtml(number(history.cv_std, 4))}<br>`;
            if (history.baseline_algorithm) metricHtml += `<b>Baseline ${escapeHtml(history.baseline_metric || "metric")}:</b> ${escapeHtml(number(history.baseline_value, 4))} (${escapeHtml(history.baseline_algorithm)})<br>`;
            if (Number.isFinite(Number(history.improvement_vs_baseline))) metricHtml += `<b>Improvement over baseline:</b> ${escapeHtml(number(history.improvement_vs_baseline, 4))}<br>`;
            metricHtml += `<b>Holdout:</b> ${escapeHtml(String(response.testSampleCount ?? "—"))} samples / ${escapeHtml(String(response.sampleCount ?? "—"))} total samples<br>`;
            if (response.imageMode === "one_class_novelty") {
                metricHtml += `<b>Known-class acceptance rate on calibration:</b> ${escapeHtml(percent(metrics.known_class_acceptance_rate))}<br>`;
                metricHtml += `<b>Note:</b> ${escapeHtml(metrics.note || "No negative-class samples are available to evaluate unknown-class rejection.")}<br>`;
                metricHtml += `<b>Prediction threshold:</b> confidence &gt; ${escapeHtml(percent(response.confidenceThreshold ?? 0.64))}<br>`;
            } else if (response.taskType === "classification" || response.modality === "image") {
                metricHtml += `<b>Accuracy:</b> ${escapeHtml(percent(metrics.accuracy))}<br>`;
                metricHtml += `<b>Balanced accuracy:</b> ${escapeHtml(percent(metrics.balanced_accuracy))}<br>`;
                metricHtml += `<b>Macro F1:</b> ${escapeHtml(number(metrics.f1_macro, 4))}<br>`;
                if (response.modality === "image") {
                    metricHtml += `<b>Rejected as Other:</b> ${escapeHtml(percent(metrics.rejected_rate))} (${escapeHtml(String(metrics.rejected_count ?? 0))} images)<br>`;
                    metricHtml += `<b>Label acceptance rule:</b> confidence &gt; ${escapeHtml(percent(response.confidenceThreshold ?? 0.64))}; otherwise Other<br>`;
                }
                const confusion = metrics.confusion_matrix;
                if (confusion?.labels?.length && confusion.labels.length <= 8) {
                    metricHtml += `<div class="confusion-wrap"><b>Confusion matrix (rows = actual, columns = predicted)</b><table class="matrix-table"><thead><tr><th>Actual \\ Predicted</th>${confusion.labels.map(label => `<th>${escapeHtml(label)}</th>`).join("")}</tr></thead><tbody>`;
                    confusion.matrix.forEach((row, i) => {
                        metricHtml += `<tr><th>${escapeHtml(confusion.labels[i])}</th>${row.map(value => `<td>${escapeHtml(value)}</td>`).join("")}</tr>`;
                    });
                    metricHtml += `</tbody></table></div>`;
                }
            } else {
                metricHtml += `<b>R²:</b> ${escapeHtml(number(metrics.r2, 4))}<br>`;
                metricHtml += `<b>MAE:</b> ${escapeHtml(number(metrics.mae, 4))}<br>`;
                metricHtml += `<b>RMSE:</b> ${escapeHtml(number(metrics.rmse, 4))}<br>`;
            }
            metricHtml += `<b>Training time:</b> ${escapeHtml(number(history.execution_time_sec, 2))} seconds`;
            document.getElementById("out-history").innerHTML = metricHtml;

            let xaiHtml = `<b>Method:</b> ${escapeHtml(xai.method || "—")}<br>`;
            xaiHtml += `<b>Interpretation:</b> ${escapeHtml(xai.top_contributors || "—")}<br>`;
            const importance = xai.importanceMap || {};
            const entries = Object.entries(importance).sort((a, b) => Number(b[1]) - Number(a[1]));
            if (entries.length) {
                xaiHtml += `<div class="chart-container"><div class="chart-title">Feature / Group Importance</div>`;
                for (const [feature, value] of entries) {
                    const width = Math.max(0, Math.min(100, Number(value) * 100));
                    xaiHtml += `<div class="importance-row"><span title="${escapeHtml(feature)}">${escapeHtml(feature)}</span><div class="importance-track"><div class="importance-fill" style="width:${width.toFixed(2)}%"></div></div><b>${escapeHtml(number(value, 3))}</b></div>`;
                }
                xaiHtml += `</div>`;
            }
            const matrix = xai.correlation_matrix || [];
            const headers = xai.correlation_headers || [];
            if (matrix.length && headers.length) {
                xaiHtml += `<div class="chart-container"><div class="chart-title">Spearman Correlation (descriptive only)</div><div class="matrix-wrap">`;
                xaiHtml += `<table class="matrix-table"><thead><tr><th></th>${headers.map(h => `<th>${escapeHtml(h)}</th>`).join("")}</tr></thead><tbody>`;
                matrix.forEach((row, i) => {
                    xaiHtml += `<tr><th>${escapeHtml(headers[i] || "")}</th>`;
                    row.forEach(value => {
                        const v = Number(value);
                        const shade = v > 0.5 ? "positive" : v < -0.5 ? "negative" : "neutral";
                        xaiHtml += `<td class="${shade}">${escapeHtml(number(v, 2))}</td>`;
                    });
                    xaiHtml += `</tr>`;
                });
                xaiHtml += `</tbody></table></div></div>`;
            }
            if (!entries.length && !matrix.length) xaiHtml += `<p>No detailed XAI analysis is available for this training run.</p>`;
            document.getElementById("out-xai").innerHTML = xaiHtml;
            document.getElementById("test-predict-prompt").style.display = "block";
            window.ModelManager?.saveModel(response);
        }
    };

    window.ModelManager = {
        models: [],
        currentModel: null,

        async init() { await this.refresh(); },

        async refresh() {
            try {
                const response = await window.AutoMLApiClient.listModels();
                if (response.status === "ERROR") {
                    console.error("Unable to load the model registry:", response.reason);
                    return;
                }
                this.models = response.items || [];
                this.renderModelList();
                if (this.currentModel) {
                    const refreshed = this.models.find(model => model.jobId === this.currentModel.jobId);
                    this.setTrainedModel(refreshed || null);
                } else if (this.models.length) {
                    this.setTrainedModel(this.models[0]);
                } else {
                    this.setTrainedModel(null);
                }
            } catch (error) {
                console.error("Unable to refresh the model registry:", error);
            }
        },

        async saveModel(modelData) {
            await this.refresh();
            const stored = this.models.find(model => model.jobId === modelData.jobId);
            if (stored) this.setTrainedModel(stored);
        },

        async deleteModel(id) {
            if (!confirm("Delete this trained model and all associated files from backend storage?")) return;
            const result = await window.AutoMLApiClient.deleteModel(id);
            if (result.status !== "SUCCESS") {
                alert(result.reason || "The model could not be deleted from backend storage.");
                return;
            }
            if (this.currentModel?.jobId === id) this.currentModel = null;
            await this.refresh();
        },

        async deleteAll() {
            if (!this.models.length) return;
            if (!confirm(`Delete all ${this.models.length} trained models and their artifacts from backend storage?`)) return;
            const result = await window.AutoMLApiClient.deleteAllModels();
            if (result.status !== "SUCCESS") {
                alert(result.reason || "One or more models could not be deleted.");
                await this.refresh();
                return;
            }
            this.currentModel = null;
            await this.refresh();
        },

        renderModelList() {
            const list = document.getElementById("model-history-list");
            if (!list) return;
            list.replaceChildren();
            if (!this.models.length) {
                list.textContent = "No trained models are currently stored on the backend.";
                return;
            }
            for (const model of this.models) {
                const wrapper = document.createElement("div");
                wrapper.className = "model-list-item";
                const details = document.createElement("div");
                details.className = "model-list-details";
                const title = document.createElement("b");
                title.textContent = model.name || model.jobId;
                const artifact = model.modelArtifact || {};
                const meta = document.createElement("small");
                meta.textContent = `${model.modality} · ${model.outputs?.output_2_model?.algorithm || "model"} · ${model.jobId} · ${formatBytes(artifact.size_bytes)}`;
                details.append(title, meta);

                const actions = document.createElement("div");
                actions.className = "model-list-actions";
                const select = document.createElement("button");
                select.className = "outline compact";
                select.textContent = "Use for testing";
                select.addEventListener("click", () => this.setTrainedModel(model));
                const download = document.createElement("a");
                download.className = "download-btn compact";
                download.href = model.download?.model || `/api/jobs/${encodeURIComponent(model.jobId)}/model`;
                download.textContent = "Download .joblib";
                download.setAttribute("download", artifact.file_name || `${model.jobId}_model.joblib`);
                const remove = document.createElement("button");
                remove.className = "danger compact";
                remove.textContent = "Delete";
                remove.addEventListener("click", () => this.deleteModel(model.jobId));
                actions.append(select, download, remove);
                wrapper.append(details, actions);
                list.append(wrapper);
            }
        },

        setTrainedModel(model) {
            this.currentModel = model;
            this.renderModelList();
            const status = document.getElementById("predict-model-status");
            const imageForm = document.getElementById("predict-image-form");
            const tabularForm = document.getElementById("predict-tabular-form");
            const resultBox = document.getElementById("predict-result-box");
            resultBox.style.display = "none";
            if (!model) {
                status.textContent = "No model selected. Train a model first.";
                status.style.color = "#f87171";
                imageForm.style.display = "none";
                tabularForm.style.display = "none";
                return;
            }
            const modality = model.modality;
            status.textContent = `Selected model: ${model.name} (${model.jobId})`;
            status.style.color = "#34d399";
            imageForm.style.display = modality === "image" ? "block" : "none";
            tabularForm.style.display = modality === "tabular" ? "block" : "none";
            if (modality === "tabular") {
                const features = model.features || [];
                const types = model.featureTypes || {};
                document.getElementById("dynamic-tabular-instruction").textContent =
                    `Enter exactly ${features.length} features in this order: ${features.map(feature => `${feature} (${types[feature] || "value"})`).join(", ")}. Use JSON arrays or objects to preserve strings containing commas.`;
                document.getElementById("predict-features").placeholder = `[12.5, "category A", 4] or {"${features[0] || "feature"}": 12.5}`;
            } else {
                const labels = model.labels || model.classes || [];
                const threshold = Number(model.confidenceThreshold ?? 0.64);
                document.getElementById("dynamic-image-instruction").textContent =
                    `Class labels come from folder names: ${labels.join(", ")}. Upload a JPEG/PNG image for content-based inference. A label is returned only when confidence is greater than ${(threshold * 100).toFixed(0)}%; otherwise the result is Other.`;
            }
        }
    };

    document.addEventListener("DOMContentLoaded", () => window.ModelManager.init());
})();
