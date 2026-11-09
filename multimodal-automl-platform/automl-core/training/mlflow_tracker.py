import time
import os
import json
from typing import Dict, Any

class MLflowExperimentTracker:
    def __init__(self, artifact_dir: str = "data/artifacts"):
        self.artifact_dir = artifact_dir
        os.makedirs(self.artifact_dir, exist_ok=True)

    def log_experiment(self, job_id: str, algorithm: str, lr: float, loss: float, accuracy: float, exec_time: float) -> Dict[str, Any]:
        record = {
            "job_id": job_id,
            "algorithm": algorithm,
            "learning_rate": lr,
            "loss": round(loss, 4),
            "accuracy": round(accuracy, 4),
            "execution_time_sec": round(exec_time, 2),
            "weights_path": f"{self.artifact_dir}/model_{job_id}.pt",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(os.path.join(self.artifact_dir, f"mlflow_run_{job_id}.json"), "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        return record
