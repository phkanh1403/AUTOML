window.OutputRenderer = {
    downloadMockFile(filename, content, type) {
        let blob;
        if (type === 'csv') {
            blob = new Blob([content], { type: 'text/csv' });
        } else if (type === 'zip') {
             blob = new Blob(["Mock zip content..."], { type: 'application/zip' });
        } else {
             blob = new Blob([content], { type: 'application/octet-stream' }); 
        }
        
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
            document.getElementById("out-xai").innerHTML = `<span style="color:#ef4444">Hủy</span>`;
            document.getElementById("test-predict-prompt").style.display = "none";
            return;
        }
        
        logEl.innerHTML = `<span style="color:#10b981; font-weight:bold;">✅ HUẤN LUYỆN THÀNH CÔNG!</span>\nToàn bộ 5 giai đoạn đã hoàn tất.`;
        const out = response.outputs;
        const dsPath = out.output_1_dataset.artifact_path;
        const mdPath = out.output_2_model.onnx_compiled;
        
        let downloadBtn = '';
        if (response.modality === 'tabular') {
            downloadBtn = `<button class="download-btn" onclick="window.OutputRenderer.downloadMockFile('${dsPath}', \`${out.output_1_dataset.file_content}\`, 'csv')">⬇️ Tải CSV đã xử lý</button>`;
        } else {
             downloadBtn = `<button class="download-btn" onclick="window.OutputRenderer.downloadMockFile('${dsPath}', '', 'zip')">⬇️ Tải ZIP Ảnh đã xử lý</button>`;
        }

        document.getElementById("out-dataset").innerHTML = 
            `<b>Trạng thái:</b> ${out.output_1_dataset.status_msg}<br>` +
            `<b>Biến đổi:</b> ${out.output_1_dataset.preprocessing_applied}<br>` + 
            downloadBtn;
            
        document.getElementById("out-model").innerHTML = 
            `<b>Model:</b> ${out.output_2_model.algorithm}<br>` + 
            `<button class="download-btn" onclick="window.OutputRenderer.downloadMockFile('${mdPath}', 'Model Weights Mock', 'model')">⬇️ Tải Mô hình (.onnx)</button>`;
            
        document.getElementById("out-history").innerHTML = `<b>Accuracy:</b> ${(out.output_3_history.accuracy * 100).toFixed(2)}%<br><b>Time:</b> ${out.output_3_history.execution_time_sec}s`;
        
        let xaiContent = `<b>XAI:</b> ${out.output_4_xai.method}<br>`;
        xaiContent += `<b>Thuật toán Train:</b> <span style="color:#10b981; font-weight:bold;">${out.output_4_xai.model_used}</span><br>`;
        xaiContent += `<b>Đóng góp:</b> ${out.output_4_xai.top_contributors}<br>`;
        
        if (response.modality === 'tabular' && out.output_4_xai.importanceMap) {
             let bars = "";
             for (const [feat, val] of Object.entries(out.output_4_xai.importanceMap)) {
                  const width = Math.max(10, val * 100);
                  bars += `<div style="display:flex; align-items:center; margin-top:4px; font-size:11px;">
                              <div style="width:70px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; text-align:right; margin-right:8px;" title="${feat}">${feat}</div>
                              <div style="background:#38bdf8; height:12px; width:${width}%; border-radius:2px;"></div>
                              <div style="margin-left:8px;">${val}</div>
                           </div>`;
             }
             xaiContent += `<div class="chart-container" style="margin-top:10px;">
                                <div style="font-size:12px; color:#333; font-weight:bold; margin-bottom:5px;">SHAP Feature Importance</div>
                                ${bars}
                            </div>`;
        }
        document.getElementById("out-xai").innerHTML = xaiContent;
        
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
            document.getElementById('predict-result-box').style.display = 'none';
            
            if (mod === 'tabular') {
                const features = modelData.features || [];
                const dropped = modelData.outputs.output_1_dataset.dropped_columns || [];
                const validFeatures = features.filter(f => !dropped.includes(f));

                document.getElementById('dynamic-tabular-instruction').innerHTML = 
                    `Các cột ảnh hưởng đến Predict theo thứ tự: <br><b style="color:#38bdf8;">${validFeatures.join(', ')}</b><br>` + 
                    `<i>(Vui lòng nhập các giá trị lần lượt theo đúng thứ tự trên, cách nhau bởi dấu phẩy)</i>`;
                document.getElementById('predict-features').placeholder = "Ví dụ: 8.5, 3.2, 10...";
            }
        }
    }
};
