import numpy as np
from .log_ingestor import TrainingLogIngestor

class NaturalLanguageRAGAssistant:
    def __init__(self, ingestor: TrainingLogIngestor):
        self.ingestor = ingestor

    def query(self, question: str) -> str:
        if not self.ingestor.memory_store:
            return "Chua co nhat ky huan luyen nao trong Vector DB."
        q_vec = self.ingestor.compute_embedding(question)
        best_item = max(self.ingestor.memory_store, key=lambda item: float(np.dot(q_vec, item["embedding"])))
        rec = best_item["record"]
        return (
            f"[RAG Phan tich]: Mo hinh {rec['algorithm']} dat do chinh xac {rec['accuracy']*100:.2f}%, "
            f"Loss = {rec['loss']}, Learning Rate = {rec['learning_rate']} trong {rec['execution_time_sec']}s."
        )
