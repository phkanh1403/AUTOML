window.OutputRenderer = {
    render(response) {
        document.getElementById("pipeline-log").textContent = "Phản hồi từ API Gateway:\n" + JSON.stringify(response, null, 2);
        if (response.status !== "SUCCESS") {
            document.getElementById("out-dataset").innerHTML = `<span style="color:#ef4444">${response.reason}</span>`;
            document.getElementById("out-model").textContent = "Đã hủy do dữ liệu vi phạm ràng buộc.";
            document.getElementById("out-history").textContent = "Không kích hoạt huấn luyện.";
            document.getElementById("out-xai").textContent = "Không trích xuất XAI.";
            window.PredictionManager.setTrainedModel(null);
            return;
        }
        const out = response.outputs;
        document.getElementById("out-dataset").innerHTML = `<b>Tiền xử lý:</b><br>${out.output_1_dataset.preprocessing_applied}<br><br><b>Path:</b> ${out.output_1_dataset.artifact_path}`;
        document.getElementById("out-model").innerHTML = `<b>Thuật toán:</b> ${out.output_2_model.algorithm}<br><b>Biên dịch:</b> ${out.output_2_model.onnx_compiled}`;
        document.getElementById("out-history").innerHTML = `<b>Accuracy:</b> ${(out.output_3_history.accuracy * 100).toFixed(2)}%<br><b>Time:</b> ${out.output_3_history.execution_time_sec}s`;
        document.getElementById("out-xai").innerHTML = `<b>XAI:</b> ${out.output_4_xai.method}<br><b>Đóng góp:</b> ${out.output_4_xai.top_contributors}`;
        
        window.RAGWidget.recordLatestRun(out);
        
        // Cập nhật trạng thái cho tab Predict
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
                `Sẵn sàng: Mô hình ${modelData.outputs.output_2_model.algorithm} (Accuracy: ${(modelData.outputs.output_3_history.accuracy * 100).toFixed(2)}%)` : 
                "Chưa có mô hình nào được huấn luyện thành công. Vui lòng chạy Pipeline trước.";
            status.style.color = modelData ? "#34d399" : "#f87171";
            
            // Hiện đúng form predict
            if(modelData) {
                document.getElementById('predict-image-form').style.display = modelData.outputs.output_1_dataset.modality === 'image' ? 'block' : 'none';
                document.getElementById('predict-tabular-form').style.display = modelData.outputs.output_1_dataset.modality === 'tabular' ? 'block' : 'none';
            } else {
                document.getElementById('predict-image-form').style.display = 'none';
                document.getElementById('predict-tabular-form').style.display = 'none';
            }
        }
    }
};
