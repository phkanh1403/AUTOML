from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Optional, List
import pandas as pd
import uvicorn

from preprocessing.tabular.column_router import TabularColumnRouter
from training.oop_trainer import OOPAutoMLTrainer
from explainability.shap_analyzer import SHAPFeatureAnalyzer
from rag_assistant.log_ingestor import TrainingLogIngestor
from rag_assistant.nl_query_engine import NaturalLanguageRAGAssistant

app = FastAPI(title="Multimodal AutoML Core Worker")
ingestor = TrainingLogIngestor()
rag_engine = NaturalLanguageRAGAssistant(ingestor)

class PipelinePayload(BaseModel):
    job_id: str
    modality: str
    domain: Optional[str] = "natural"
    mode: str = "auto"
    targetColumn: Optional[str] = "target_label"
    droppedColumns: Optional[List[str]] = []
    dataPayload: Any

class RAGQueryPayload(BaseModel):
    question: str

@app.post("/api/v1/execute-pipeline")
def execute_pipeline(payload: PipelinePayload):
    trainer = OOPAutoMLTrainer(mode=payload.mode)
    history = trainer.execute_training(payload.job_id, payload.modality)
    ingestor.ingest_run(history)

    xai_details = {"gradcam_focus_region": "Spatial ROI Layer4 Conv2d"}
    if payload.modality == "tabular" and isinstance(payload.dataPayload, list) and len(payload.dataPayload) > 0:
        df = pd.DataFrame(payload.dataPayload)
        router = TabularColumnRouter(target_col=payload.targetColumn)
        df_processed, _ = router.process(df)
        xai_details = SHAPFeatureAnalyzer.compute_feature_importance(df_processed, payload.targetColumn)

    return {
        "output_1_dataset": {
            "status": "Cleaned, Normalized & Augmented",
            "modality": payload.modality,
            "domain": payload.domain,
            "dropped_columns": payload.droppedColumns,
            "path": f"data/processed/{payload.modality}_{payload.job_id}"
        },
        "output_2_model": {
            "algorithm": history["algorithm"],
            "weights_path": history["weights_path"],
            "onnx_path": f"data/artifacts/model_{payload.job_id}.onnx",
            "tensorrt_path": f"data/artifacts/model_{payload.job_id}.trt"
        },
        "output_3_history": history,
        "output_4_xai": {
            "type": "Grad-CAM Heatmap" if payload.modality == "image" else "SHAP Values",
            "details": xai_details
        }
    }

@app.post("/api/v1/rag-query")
def query_rag(payload: RAGQueryPayload):
    return {"answer": rag_engine.query(payload.question)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
