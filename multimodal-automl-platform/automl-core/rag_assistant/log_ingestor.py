import hashlib
import numpy as np
from typing import Dict, Any, List

class TrainingLogIngestor:
    def __init__(self):
        self.memory_store: List[Dict[str, Any]] = []

    @staticmethod
    def compute_embedding(text: str, dim: int = 384) -> np.ndarray:
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(dim)
        return vec / np.linalg.norm(vec)

    def ingest_run(self, run_record: Dict[str, Any]):
        summary = (
            f"Run {run_record.get('job_id')}: Algorithm={run_record.get('algorithm')}, "
            f"LR={run_record.get('learning_rate')}, Accuracy={run_record.get('accuracy')}, "
            f"Loss={run_record.get('loss')}, Time={run_record.get('execution_time_sec')}s."
        )
        self.memory_store.append({
            "summary": summary,
            "record": run_record,
            "embedding": self.compute_embedding(summary)
        })
