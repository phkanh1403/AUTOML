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
        document.getElementById("pipeline-log").textContent = JSON.stringify(response, null, 2);
        if (response.status !== "SUCCESS") {
            document.getElementById("out-dataset").innerHTML = `<span style="color:#ef4444">${response.reason}</span>`;
            document.getElementById("out-model").textContent = "Đã hủy do vi phạm ràng buộc.";
            window.PredictionManager.setTrainedModel(null);
            return;
        }
        
        const out = response.outputs;
        const dsPath = out.output_1_dataset.artifact_path;
        const mdPath = out.output_2_model.onnx_compiled;
        
        document.getElementById("out-dataset").innerHTML = 
            `<b>Xử lý:</b> ${out.output_1_dataset.preprocessing_applied}<br>` + 
            `<button class="download-btn" onclick="window.OutputRenderer.downloadMockFile('${dsPath}', 'Du lieu mock da xu ly')">⬇️ Tải Dataset đã xử lý</button>`;
            
        document.getElementById("out-model").innerHTML = 
            `<b>Model:</b> ${out.output_2_model.algorithm}<br>` + 
            `<button class="download-btn" onclick="window.OutputRenderer.downloadMockFile('${mdPath}', 'ONNX Weights')">⬇️ Tải Mô hình (.onnx)</button>`;
            
        document.getElementById("out-history").innerHTML = `<b>Accuracy:</b> ${(out.output_3_history.accuracy * 100).toFixed(2)}%<br><b>Time:</b> ${out.output_3_history.execution_time_sec}s`;
        document.getElementById("out-xai").innerHTML = `<b>XAI:</b> ${out.output_4_xai.method}`;
        
        // Hiện nút chuyển sang tab Predict
        document.getElementById("test-predict-prompt").style.display = "block";
        window.PredictionManager.setTrainedModel(response);
    }
};

window.PredictionManager = {
    currentModel: null,
    setTrainedModel(modelData) {
        this.currentModel = modelData;
        const status = document.getElementById("predict-model-status");
        if(status) {
            status.textContent = modelData ? 
                `✅ Đã nạp Mô hình ${modelData.outputs.output_2_model.algorithm} (Độ chính xác: ${(modelData.outputs.output_3_history.accuracy * 100).toFixed(2)}%)` : 
                "❌ Chưa có mô hình. Vui lòng Huấn luyện (Tab 1) trước.";
            status.style.color = modelData ? "#10b981" : "#ef4444";
            
            const mod = modelData ? (modelData.modality || modelData.outputs.output_1_dataset.modality) : null;
            document.getElementById('predict-image-form').style.display = (mod === 'image') ? 'block' : 'none';
            document.getElementById('predict-tabular-form').style.display = (mod === 'tabular') ? 'block' : 'none';
        }
    }
};
