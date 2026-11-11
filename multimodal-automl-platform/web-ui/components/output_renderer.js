(() => {
    const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    })[char]);

    const percent = (value) => Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(2)}%` : "N/A";
    const number = (value, digits = 4) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "N/A";

    window.OutputRenderer = {
        render(response) {
            const log = document.getElementById("pipeline-log");
            if (!response || response.status !== "SUCCESS") {
                const reason = response?.reason || "Backend không trả về kết quả hợp lệ.";
                log.textContent = `❌ TRAINING THẤT BẠI\n${reason}`;
                document.getElementById("out-dataset").textContent = "Không có artifact";
                document.getElementById("out-model").textContent = "Không có model";
                document.getElementById("out-history").textContent = "Không có metric";
                document.getElementById("out-xai").textContent = "Không có giải thích";
                document.getElementById("test-predict-prompt").style.display = "none";
                return;
            }

            const outputs = response.outputs || {};
            const dataset = outputs.output_1_dataset || {};
            const model = outputs.output_2_model || {};
            const history = outputs.output_3_history || {};
            const xai = outputs.output_4_xai || {};
            const metrics = history.metrics || {};

            const selectionSummary = response.imageMode === "one_class_novelty"
                ? "Mô hình novelty detection được fit trên một lớp ảnh (không có CV phân lớp)."
                : `Thuật toán được chọn bằng ${history.cv_folds || "—"}-fold cross-validation trên tập train.`;
            log.textContent = `✅ HUẤN LUYỆN HOÀN TẤT\nJob: ${response.jobId}\n${selectionSummary}\nModel được lưu ở backend.`;
            if (response.warnings?.length) log.textContent += `\n\nLưu ý:\n• ${response.warnings.join("\n• ")}`;

            const datasetUrl = response.download?.dataset || "#";
            const modelUrl = response.download?.model || "#";
            document.getElementById("out-dataset").innerHTML =
                `<b>Trạng thái:</b> ${escapeHtml(dataset.status_msg)}<br>` +
                `<b>Tiền xử lý:</b> ${escapeHtml(dataset.preprocessing_applied)}<br>` +
                `<a class="download-btn" href="${escapeHtml(datasetUrl)}">⬇️ Tải dataset đã xử lý</a>`;

            document.getElementById("out-model").innerHTML =
                `<b>Model:</b> ${escapeHtml(model.algorithm)}<br>` +
                `<b>Định dạng:</b> ${escapeHtml(model.format || "joblib")}<br>` +
                `<a class="download-btn" href="${escapeHtml(modelUrl)}">⬇️ Tải model đã train</a>`;

            let metricHtml = `<b>Thuật toán:</b> ${escapeHtml(model.algorithm)}<br>`;
            metricHtml += `<b>Metric chính:</b> ${escapeHtml(history.primary_metric || "—")} = ${escapeHtml(number(history.primary_value, 4))}<br>`;
            metricHtml += `<b>CV ${escapeHtml(history.cv_metric || "score")}:</b> ${escapeHtml(number(history.cv_score, 4))} ± ${escapeHtml(number(history.cv_std, 4))}<br>`;
            if (history.baseline_algorithm) metricHtml += `<b>Baseline ${escapeHtml(history.baseline_metric || "metric")}:</b> ${escapeHtml(number(history.baseline_value, 4))} (${escapeHtml(history.baseline_algorithm)})<br>`;
            if (Number.isFinite(Number(history.improvement_vs_baseline))) metricHtml += `<b>Cải thiện so với baseline:</b> ${escapeHtml(number(history.improvement_vs_baseline, 4))}<br>`;
            metricHtml += `<b>Holdout:</b> ${escapeHtml(String(response.testSampleCount ?? "—"))} mẫu / ${escapeHtml(String(response.sampleCount ?? "—"))} mẫu tổng<br>`;
            if (response.imageMode === "one_class_novelty") {
                metricHtml += `<b>Tỷ lệ giữ lại ảnh cùng lớp trên calibration:</b> ${escapeHtml(percent(metrics.known_class_acceptance_rate))}<br>`;
                metricHtml += `<b>Lưu ý:</b> ${escapeHtml(metrics.note || "Không có dữ liệu lớp âm tính để đo khả năng từ chối lớp lạ.")}<br>`;
                metricHtml += `<b>Ngưỡng dự đoán:</b> confidence &gt; ${escapeHtml(percent(response.confidenceThreshold ?? 0.64))}<br>`;
            } else if (response.taskType === "classification" || response.modality === "image") {
                metricHtml += `<b>Accuracy:</b> ${escapeHtml(percent(metrics.accuracy))}<br>`;
                metricHtml += `<b>Balanced accuracy:</b> ${escapeHtml(percent(metrics.balanced_accuracy))}<br>`;
                metricHtml += `<b>F1-macro:</b> ${escapeHtml(number(metrics.f1_macro, 4))}<br>`;
                if (response.modality === "image") {
                    metricHtml += `<b>Tỷ lệ bị từ chối (Other):</b> ${escapeHtml(percent(metrics.rejected_rate))} (${escapeHtml(String(metrics.rejected_count ?? 0))} ảnh)<br>`;
                    metricHtml += `<b>Ngưỡng trả nhãn:</b> confidence &gt; ${escapeHtml(percent(response.confidenceThreshold ?? 0.64))}; nếu không trả Other<br>`;
                }
                const confusion = metrics.confusion_matrix;
                if (confusion?.labels?.length && confusion.labels.length <= 8) {
                    metricHtml += `<div class="confusion-wrap"><b>Confusion matrix (rows = actual, columns = predicted)</b><table class="matrix-table"><thead><tr><th>Actual \ Pred.</th>${confusion.labels.map(label => `<th>${escapeHtml(label)}</th>`).join("")}</tr></thead><tbody>`;
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
            metricHtml += `<b>Thời gian train:</b> ${escapeHtml(number(history.execution_time_sec, 2))} giây`;
            document.getElementById("out-history").innerHTML = metricHtml;

            let xaiHtml = `<b>Phương pháp:</b> ${escapeHtml(xai.method || "—")}<br>`;
            xaiHtml += `<b>Diễn giải:</b> ${escapeHtml(xai.top_contributors || "—")}<br>`;
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
                xaiHtml += `<div class="chart-container"><div class="chart-title">Spearman Correlation (mang tính mô tả)</div><div class="matrix-wrap">`;
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
            if (!entries.length && !matrix.length) xaiHtml += `<p>Không có phân tích XAI chi tiết cho lượt train này.</p>`;
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
            const response = await window.AutoMLApiClient.listModels();
            if (response.status === "ERROR") {
                console.error("Không tải được model history:", response.reason);
                return;
            }
            this.models = response.items || [];
            this.renderModelList();
            if (this.currentModel) {
                const refreshed = this.models.find(m => m.jobId === this.currentModel.jobId);
                this.setTrainedModel(refreshed || null);
            } else if (this.models.length) {
                this.setTrainedModel(this.models[0]);
            } else {
                this.setTrainedModel(null);
            }
        },

        async saveModel(modelData) {
            await this.refresh();
            const stored = this.models.find(m => m.jobId === modelData.jobId);
            if (stored) this.setTrainedModel(stored);
        },

        async deleteModel(id) {
            if (!confirm("Xóa model và toàn bộ artifact được lưu trên backend?")) return;
            const result = await window.AutoMLApiClient.deleteModel(id);
            if (result.status !== "SUCCESS") {
                alert(result.reason || "Không thể xóa model.");
                return;
            }
            if (this.currentModel?.jobId === id) this.currentModel = null;
            await this.refresh();
        },

        async deleteAll() {
            if (!this.models.length) return;
            if (!confirm(`Xóa toàn bộ ${this.models.length} model và artifact trên backend?`)) return;
            for (const model of [...this.models]) {
                const result = await window.AutoMLApiClient.deleteModel(model.jobId);
                if (result.status !== "SUCCESS") console.error("Không thể xóa", model.jobId, result.reason);
            }
            this.currentModel = null;
            await this.refresh();
        },

        renderModelList() {
            const list = document.getElementById("model-history-list");
            if (!list) return;
            list.replaceChildren();
            if (!this.models.length) {
                list.textContent = "Chưa có mô hình nào được lưu trên backend.";
                return;
            }
            for (const model of this.models) {
                const wrapper = document.createElement("div");
                wrapper.className = "model-list-item";
                const details = document.createElement("div");
                details.className = "model-list-details";
                const title = document.createElement("b");
                title.textContent = model.name || model.jobId;
                const meta = document.createElement("small");
                meta.textContent = `${model.modality} · ${model.outputs?.output_2_model?.algorithm || "model"} · ${model.jobId}`;
                details.append(title, meta);
                const actions = document.createElement("div");
                const select = document.createElement("button");
                select.className = "outline compact";
                select.textContent = "Chọn test";
                select.addEventListener("click", () => this.setTrainedModel(model));
                const remove = document.createElement("button");
                remove.className = "danger compact";
                remove.textContent = "Xóa";
                remove.addEventListener("click", () => this.deleteModel(model.jobId));
                actions.append(select, remove);
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
                status.textContent = "❌ Chưa có model. Hãy train một model trước.";
                status.style.color = "#f87171";
                imageForm.style.display = "none";
                tabularForm.style.display = "none";
                return;
            }
            const modality = model.modality;
            status.textContent = `✅ Đang chọn: ${model.name} (${model.jobId})`;
            status.style.color = "#34d399";
            imageForm.style.display = modality === "image" ? "block" : "none";
            tabularForm.style.display = modality === "tabular" ? "block" : "none";
            if (modality === "tabular") {
                const features = model.features || [];
                const types = model.featureTypes || {};
                document.getElementById("dynamic-tabular-instruction").textContent =
                    `Nhập đúng thứ tự ${features.length} đặc trưng: ${features.map(f => `${f} (${types[f] || "value"})`).join(", ")}. Có thể nhập JSON array/object để giữ chuỗi có dấu phẩy.`;
                document.getElementById("predict-features").placeholder = `[12.5, "category A", 4] hoặc {"${features[0] || "feature"}": 12.5}`;
            } else {
                const labels = model.labels || model.classes || [];
                const threshold = Number(model.confidenceThreshold ?? 0.64);
                document.getElementById("dynamic-image-instruction").textContent =
                    `Các nhãn lấy từ tên folder: ${labels.join(", ")}. Tải ảnh JPEG/PNG để dự đoán theo nội dung ảnh. Chỉ trả nhãn khi confidence > ${(threshold * 100).toFixed(0)}%; nếu không sẽ trả Other.`;
            }
        }
    };

    document.addEventListener("DOMContentLoaded", () => window.ModelManager.init());
})();
