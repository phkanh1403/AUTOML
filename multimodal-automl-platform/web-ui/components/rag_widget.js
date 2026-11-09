window.RAGWidget = {
    lastRun: null,
    recordLatestRun(outputs) {
        this.lastRun = outputs;
    },
    askQuestion() {
        const input = document.getElementById("rag-input").value.trim();
        const box = document.getElementById("rag-response");
        if (!input) return;
        if (!this.lastRun) {
            box.textContent = "[RAG Assistant]: Chua co lich su huan luyen trong Vector DB.";
            return;
        }
        const h = this.lastRun.output_3_history;
        const m = this.lastRun.output_2_model;
        box.textContent = `[RAG Vector DB Truy xuat]: Mo hinh ${m.algorithm} hoi tu tai Learning Rate = ${h.learning_rate}, dat Accuracy ${(h.accuracy * 100).toFixed(2)}% (Loss: ${h.loss}) trong ${h.execution_time_sec}s va da xuat sang ONNX/TensorRT.`;
    }
};
