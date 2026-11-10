window.OutputRenderer = {
    tempCSVContent: "",

    downloadMockFile(filename, type) {
        if (type === 'zip') {
            if (typeof JSZip !== 'undefined') {
                const zip = new JSZip();
                zip.file("INFO.txt", "Tập dữ liệu Ảnh của bạn đã qua bước tiền xử lý.\n- Đã chuẩn hóa kích thước và hệ màu.\n- Đã loại bỏ ảnh mờ.\n- Áp dụng các bộ lọc Data Augmentation.");
                zip.file("sample_01.txt", "Mocked image binary 1");
                
                zip.generateAsync({type:"blob"}).then(function(blob) {
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url; a.download = filename;
                    document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(url);
                });
                return;
            } else {
                alert("JSZip chưa được tải. Cần kết nối Internet để tải thư viện ZIP.");
                return;
            }
        }
        
        let blob;
        if (type === 'csv') {
            blob = new Blob([window.OutputRenderer.tempCSVContent], { type: 'text/csv' });
        } else {
            blob = new Blob(["Mock Model Weights Binary"], { type: 'application/octet-stream' }); 
        }
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = filename;
        document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(url);
    },

    render(response) {
        const logEl = document.getElementById("pipeline-log");
        
        try {
            if (response.status !== "SUCCESS") {
                logEl.innerHTML = `<span style="color:#ef4444; font-weight:bold;">❌ HỆ THỐNG TỪ CHỐI:</span>\n${response.reason}`;
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
                window.OutputRenderer.tempCSVContent = out.output_1_dataset.file_content || "Mock,CSV";
                downloadBtn = `<button class="download-btn" onclick="window.OutputRenderer.downloadMockFile('${dsPath}', 'csv')">⬇️ Tải CSV đã xử lý</button>`;
            } else {
                downloadBtn = `<button class="download-btn" onclick="window.OutputRenderer.downloadMockFile('${dsPath}', 'zip')">⬇️ Tải ZIP Ảnh đã xử lý</button>`;
            }

            document.getElementById("out-dataset").innerHTML = 
                `<b>Trạng thái:</b> ${out.output_1_dataset.status_msg}<br>` +
                `<b>Biến đổi:</b> ${out.output_1_dataset.preprocessing_applied}<br>` + 
                downloadBtn;
                
            document.getElementById("out-model").innerHTML = 
                `<b>Model:</b> ${out.output_2_model.algorithm}<br>` + 
                `<button class="download-btn" onclick="window.OutputRenderer.downloadMockFile('${mdPath}', 'model')">⬇️ Tải Mô hình (.onnx)</button>`;
                
            document.getElementById("out-history").innerHTML = `<b>Accuracy:</b> ${(out.output_3_history.accuracy * 100).toFixed(2)}%<br><b>Time:</b> ${out.output_3_history.execution_time_sec}s`;
            
            let xaiContent = `<b>XAI:</b> ${out.output_4_xai.method}<br>`;
            xaiContent += `<b>Thuật toán Train:</b> <span style="color:#10b981; font-weight:bold;">${out.output_4_xai.model_used}</span><br>`;
            xaiContent += `<b>Đóng góp:</b> ${out.output_4_xai.top_contributors}<br>`;
            
            if (response.modality === 'tabular' && out.output_4_xai.importanceMap) {
                 let bars = "";
                 for (const [feat, val] of Object.entries(out.output_4_xai.importanceMap)) {
                      const width = Math.max(5, val * 100);
                      bars += `<div style="display:flex; align-items:center; margin-top:4px; font-size:11px;">
                                  <div style="width:70px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; text-align:right; margin-right:8px; color:black;" title="${feat}">${feat}</div>
                                  <div style="background:#38bdf8; height:12px; width:${width}%; border-radius:2px;"></div>
                                  <div style="margin-left:8px; color:black;">${val}</div>
                               </div>`;
                 }
                 
                 let matrixHtml = `<div style="margin-top:15px; font-size:10px;"><div style="margin-left:35px;">`;
                 out.output_4_xai.correlation_headers.forEach(h => {
                     matrixHtml += `<span style="display:inline-block; width:35px; text-align:center; overflow:hidden; color:black;">${h}</span>`;
                 });
                 matrixHtml += `</div>`;
                 
                 out.output_4_xai.correlation_matrix.forEach((row, i) => {
                     matrixHtml += `<div style="display:flex; align-items:center;">`;
                     matrixHtml += `<span style="width:35px; text-align:right; margin-right:5px; overflow:hidden; color:black;">${out.output_4_xai.correlation_headers[i]}</span>`;
                     row.forEach(val => {
                         let bg = "white";
                         if (val == 1.0) bg = "#38bdf8";
                         else if (val > 0.5) bg = "#bae6fd";
                         else if (val < -0.5) bg = "#fecaca";
                         matrixHtml += `<span class="matrix-cell" style="background:${bg}; color:black;">${val}</span>`;
                     });
                     matrixHtml += `</div>`;
                 });
                 matrixHtml += `</div>`;

                 xaiContent += `<div class="chart-container">
                                    <div class="chart-title">SHAP Feature Importance</div>
                                    ${bars}
                                    <div class="chart-title" style="margin-top:15px;">Correlation Matrix (Lọc Đặc Trưng)</div>
                                    ${matrixHtml}
                                    <p style="font-size:10px; color:#666; margin-top:5px;">*Chỉ các cột tương quan cao mới được đưa vào Dự đoán</p>
                                </div>`;
            }
            document.getElementById("out-xai").innerHTML = xaiContent;
            
            document.getElementById("test-predict-prompt").style.display = "block";
            
            const modelToSave = JSON.parse(JSON.stringify(response));
            delete modelToSave.dataPayload;
            delete modelToSave.rawText;
            if (modelToSave.outputs && modelToSave.outputs.output_1_dataset) {
                delete modelToSave.outputs.output_1_dataset.file_content;
            }
            window.ModelManager.saveModel(modelToSave);
            
        } catch (error) {
            console.error("LỖI RENDER UI:", error);
            logEl.innerHTML = `<span style="color:#ef4444; font-weight:bold;">❌ LỖI HỆ THỐNG GIAO DIỆN:</span>\nQuá trình kết xuất UI gặp sự cố. Có thể do dữ liệu trả về lỗi cấu trúc. Chi tiết: ${error.message}`;
        }
    }
};

window.ModelManager = {
    models: [],
    currentModel: null,
    
    init() {
        try {
            const saved = localStorage.getItem('automl_models_v12');
            if (saved) {
                this.models = JSON.parse(saved);
                if (this.models.length > 0) this.setTrainedModel(this.models[this.models.length - 1]);
            }
        } catch(e) { console.warn("Không load được cache model", e); }
        this.renderModelList();
    },

    saveState() { 
        try {
            localStorage.setItem('automl_models_v12', JSON.stringify(this.models)); 
        } catch(e) {
            console.error("Lỗi lưu Model vào Cache:", e);
        }
    },
    
    saveModel(modelData) {
        modelData.id = Date.now().toString();
        modelData.name = `Mô hình ${modelData.modality.toUpperCase()} - ${new Date().toLocaleTimeString()} (Acc: ${(modelData.outputs.output_3_history.accuracy*100).toFixed(2)}%)`;
        this.models.push(modelData);
        this.saveState();
        this.renderModelList();
        this.setTrainedModel(modelData);
    },
    
    deleteModel(id) {
        this.models = this.models.filter(m => m.id !== id);
        this.saveState();
        this.renderModelList();
        if(this.currentModel && this.currentModel.id === id) {
            this.setTrainedModel(this.models.length > 0 ? this.models[this.models.length - 1] : null);
        }
    },

    deleteAll() {
        if(confirm("Bạn có chắc chắn muốn xóa toàn bộ lịch sử mô hình không?")) {
            this.models = [];
            this.currentModel = null;
            this.saveState();
            this.renderModelList();
            this.setTrainedModel(null);
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
                <div style="font-size: 13px;">
                    <b>${m.name}</b>
                    ${isActive ? '<span style="color:#10b981; font-size:12px; margin-left:8px;">[Đang chọn]</span>' : ''}
                </div>
                <div>
                    <button class="outline" style="width:auto; padding:4px 8px; margin:0; font-size: 12px;" onclick="window.ModelManager.setTrainedModelById('${m.id}')">Chọn Test</button>
                    <button class="danger" style="padding:4px 8px; margin:0; margin-left:5px; font-size: 12px;" onclick="window.ModelManager.deleteModel('${m.id}')">Xóa</button>
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
                `✅ Đang Test: ${modelData.name} (${modelData.outputs?.output_2_model?.algorithm})` : 
                "❌ Chưa có mô hình. Vui lòng Huấn luyện (Tab 1) hoặc Chọn ở danh sách.";
            status.style.color = modelData ? "#10b981" : "#ef4444";
            
            const mod = modelData ? (modelData.modality || modelData.outputs?.output_1_dataset?.modality) : null;
            document.getElementById('predict-image-form').style.display = (mod === 'image') ? 'block' : 'none';
            document.getElementById('predict-tabular-form').style.display = (mod === 'tabular') ? 'block' : 'none';
            document.getElementById('predict-result-box').style.display = 'none';
            
            if (mod === 'tabular') {
                const validFeatures = modelData.features || [];
                document.getElementById('dynamic-tabular-instruction').innerHTML = 
                    `Các cột ảnh hưởng đến Predict theo thứ tự: <br><b style="color:#38bdf8;">${validFeatures.join(', ')}</b><br>` + 
                    `<i>(Vui lòng nhập các giá trị (Chữ hoặc Số) lần lượt theo đúng thứ tự trên, cách nhau bởi dấu phẩy)</i>`;
                document.getElementById('predict-features').placeholder = "Ví dụ: United States, education, female, 6000...";
            } else if (mod === 'image') {
                // CẢI TIẾN 1: Ngưỡng hiển thị UI chuyển thành 60%
                const lbls = modelData.labels || [];
                if(lbls.length > 0) {
                     document.getElementById('dynamic-image-instruction').innerHTML = 
                        `Mô hình được huấn luyện để phát hiện: <b style="color:#38bdf8;">${lbls[0]}</b>. <br><i style="color:#cbd5e1; font-size:12px;">Nếu độ tin cậy <= 60%, kết quả sẽ là "Khác".</i>`;
                }
            }
        }
    }
};

document.addEventListener("DOMContentLoaded", () => {
    if(window.ModelManager) window.ModelManager.init();
});
