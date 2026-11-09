@echo off
echo Khoi chay toan bo he thong Multimodal AutoML Platform...
start "MLflow Tracking Server" cmd /k "automl-core\venv\Scripts\mlflow ui --port 5000"
start "FastAPI AutoML Worker" cmd /k "cd automl-core && venv\Scripts\python api_worker.py"
start "Node.js API Gateway" cmd /k "cd gateway-orchestrator && npm run dev"
start "Web UI Server" cmd /k "cd web-ui && python -m http.server 8080"
echo Tat ca dich vu da khoi chay!
