window.OutputRenderer = {
    render(response) {
        document.getElementById("pipeline-log").textContent = JSON.stringify(response, null, 2);
        if (response.status !== "SUCCESS") {
            document.getElementById("out-dataset").innerHTML = `<span style="color:#ef4444">${response.reason}</span>`;
            document.getElementById("out-model").textContent = "Da huy do du lieu dau vao vi pham rang buoc.";
            document.getElementById("out-history").textContent = "Khong kich hoat huan luyen.";
            document.getElementById("out-xai").textContent = "Khong trich xuat XAI.";
            return;
        }
        const out = response.outputs;
        document.getElementById("out-dataset").innerHTML =
            `<b>Trang thai:</b> Da chuan hoa & tang cuong<br>` +
            `<b>Chi tiet:</b> ${out.output_1_dataset.preprocessing_applied || out.output_1_dataset.status}<br>` +
            `<b>Duong dan DVC:</b> ${out.output_1_dataset.artifact_path || out.output_1_dataset.path}`;
        document.getElementById("out-model").innerHTML =
            `<b>Thuat toan:</b> ${out.output_2_model.algorithm}<br>` +
            `<b>Trong so goc:</b> ${out.output_2_model.weights || out.output_2_model.weights_path}<br>` +
            `<b>Bien dich Edge:</b> ${out.output_2_model.onnx_compiled || out.output_2_model.onnx_path} & TensorRT (.trt)`;
        document.getElementById("out-history").innerHTML =
            `<b>Learning Rate:</b> ${out.output_3_history.learning_rate}<br>` +
            `<b>Loss / Accuracy:</b> ${out.output_3_history.loss} / ${(out.output_3_history.accuracy * 100).toFixed(2)}%<br>` +
            `<b>Thoi gian:</b> ${out.output_3_history.execution_time_sec}s`;
        document.getElementById("out-xai").innerHTML =
            `<b>Phuong phap:</b> ${out.output_4_xai.method || out.output_4_xai.type}<br>` +
            `<b>Giai thich:</b> ${out.output_4_xai.top_contributors || JSON.stringify(out.output_4_xai.details)}`;
        window.RAGWidget.recordLatestRun(out);
    }
};
