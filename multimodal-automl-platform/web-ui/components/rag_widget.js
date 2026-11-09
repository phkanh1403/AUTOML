window.RAGWidget = {
    lastRun: null,
    recordLatestRun(outputs) { this.lastRun = outputs; },
    askQuestion() {
        const input = document.getElementById("rag-input").value.trim();
        const box = document.getElementById("rag-response");
        if (!input) return;
        if (!this.lastRun) {
            box.textContent = "[RAG Assistant]: Chưa có lịch sử huấn luyện thành công. Hãy chạy Pipeline trước.";
            return;
        }
        const h = this.lastRun.output_3_history;
        const m = this.lastRun.output_2_model;
        box.textContent = `[RAG Retrieval]: Dựa trên vector log, mô hình ${m.algorithm} đạt độ chính xác ${(h.accuracy * 100).toFixed(2)}% (Loss: ${h.loss}). Thời gian chạy là ${h.execution_time_sec} giây với LR=${h.learning_rate}.`;
    }
};
