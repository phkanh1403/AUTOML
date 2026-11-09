window.OutputRenderer = {
    downloadMockFile(filename, content) {
        const blob = new Blob([content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    },

    render(response) {
        const logEl = document.getElementById("pipeline-log");
        
        if (response.status !== "SUCCESS") {
            logEl.innerHTML = `<span style="color:#ef4444; font-weight:bold;">❌ HỆ THỐNG TỪ CHỐI HUẤN LUYỆN:</span>\n${response.reason}`;
            document.getElementById("out-dataset").innerHTML = `<span style="color:#ef4444">Hủy</span>`;
            document.getElementById("out-model").textContent = "Hủy";
            document.getElementById("out-history").textContent = "Hủy";
            document.getElementById("out-xai").textContent = "Hủy";
            document.getElementById("test-predict-prompt").style.display = "none";
            return;
        }
        
        logEl.innerHTML = `<span style="color:#10b981; font-weight:bold;">✅ HUẤN LUYỆN THÀNH CÔNG!</span>\nToàn bộ 5 giai đoạn đã hoàn tất.`;
        const out = response.outputs;
        const dsPath = out.output_1_dataset.artifact_path;
        const mdPath = out.output_2_model.onnx_compiled;
        
        document.getElementById("out-dataset").innerHTML = 
            `<b>Trạng thái:</b> ${out.output_1_dataset.status_msg}<br>` +
            `<b>Biến đổi:</b> ${out.output_1_dataset.preprocessing_applied}<br>` + 
            `<button class="download-btn" onclick="window.OutputRenderer.downloadMockFile('${dsPath}', 'Du lieu mock da xu ly')">⬇️ Tải Dataset đã xử lý</button>`; // LUÔN HIỆN NÚT TẢI
            
        document.getElementById("out-model").innerHTML = 
            `<b>Model:</b> ${out.output_2_model.algorithm}<br>` + 
            `<button class="download-btn" onclick="window.OutputRenderer.downloadMockFile('${mdPath}', 'ONNX Weights')">⬇️ Tải Mô hình (.onnx)</button>`;
            
        document.getElementById("out-history").innerHTML = `<b>Accuracy:</b> ${(out.output_3_history.accuracy * 100).toFixed(2)}%<br><b>Time:</b> ${out.output_3_history.execution_time_sec}s`;
        document.getElementById("out-xai").innerHTML = `<b>XAI:</b> ${out.output_4_xai.method}`;
        
        document.getElementById("test-predict-prompt").style.display = "block";
        window.ModelManager.saveModel(response);
    }
};

window.ModelManager = {
    models: [],
    currentModel: null,
    
    saveModel(response) {
        response.id = Date.now().toString();
        response.name = `Mô hình ${response.modality.toUpperCase()} - ${new Date().toLocaleTimeString()}`;
        this.models.push(response);
        this.renderModelList();
        this.setTrainedModel(response);
    },
    
    deleteModel(id) {
        this.models = this.models.filter(m => m.id !== id);
        this.renderModelList();
        if(this.currentModel && this.currentModel.id === id) {
            this.setTrainedModel(this.models.length > 0 ? this.models[this.models.length - 1] : null);
        }
    },
    
    renderModelList() {
        const listEl = document.getElementById("model-history-list");
        if(!listEl) return;
        listEl.innerHTML = "";
        if(this.models.length === 0) {
            listEl.innerHTML = "<p style='color:#94a3b8'>Chưa có mô hình nào.</p>";
            return;
        }
        this.models.forEach(m => {
            const div = document.createElement("div");
            div.className = "model-list-item";
            const isActive = this.currentModel && this.currentModel.id === m.id;
            div.innerHTML = `
                <div>
                    <b>${m.name}</b> (Acc: ${(m.outputs.output_3_history.accuracy*100).toFixed(2)}%)
                    ${isActive ? '<span style="color:#10b981; font-size:12px; margin-left:8px;">[Đang chọn]</span>' : ''}
                </div>
                <div>
                    <button class="outline" style="width:auto; padding:4px 8px; margin:0;" onclick="window.ModelManager.setTrainedModelById('${m.id}')">Chọn Test</button>
                    <button class="danger" style="padding:4px 8px; margin:0; margin-left:5px;" onclick="window.ModelManager.deleteModel('${m.id}')">Xóa</button>
                </div>
            `;
            listEl.appendChild(div);
        });
    },

    setTrainedModelById(id) {
        const m = this.models.find(x => x.id === id);
        if(m) this.setTrainedModel(m);
    },

    setTrainedModel(modelData) {
        this.currentModel = modelData;
        this.renderModelList();
        
        const status = document.getElementById("predict-model-status");
        if(status) {
            status.textContent = modelData ? 
                `✅ Đang Test: ${modelData.name} (${modelData.outputs.output_2_model.algorithm})` : 
                "❌ Chưa có mô hình. Vui lòng Huấn luyện (Tab 1) hoặc Chọn ở danh sách.";
            status.style.color = modelData ? "#10b981" : "#ef4444";
            
            const mod = modelData ? (modelData.modality || modelData.outputs.output_1_dataset.modality) : null;
            document.getElementById('predict-image-form').style.display = (mod === 'image') ? 'block' : 'none';
            document.getElementById('predict-tabular-form').style.display = (mod === 'tabular') ? 'block' : 'none';
            
            // Xử lý báo tên cột Tabular
            if (mod === 'tabular') {
                const features = modelData.outputs.output_4_xai.features || [];
                document.getElementById('dynamic-tabular-instruction').innerHTML = 
                    `Các cột ảnh hưởng đến Predict theo thứ tự: <br><b style="color:#38bdf8;">${features.join(', ')}</b><br>` + 
                    `<i>(Vui lòng nhập các giá trị lần lượt theo đúng thứ tự trên, cách nhau bởi dấu phẩy)</i>`;
                document.getElementById('predict-features').placeholder = "Ví dụ: 8.5, 3.2, 10...";
            }
        }
    }
};
