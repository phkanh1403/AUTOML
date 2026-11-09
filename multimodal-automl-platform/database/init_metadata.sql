CREATE TABLE IF NOT EXISTS pipeline_jobs (
    job_id UUID PRIMARY KEY,
    modality VARCHAR(20) NOT NULL CHECK (modality IN ('image', 'tabular')),
    domain_type VARCHAR(50),
    execution_mode VARCHAR(20) DEFAULT 'auto',
    status VARCHAR(30) NOT NULL DEFAULT 'PENDING',
    dropped_columns TEXT[],
    input_path TEXT NOT NULL,
    processed_dataset_path TEXT,
    model_artifact_path TEXT,
    onnx_path TEXT,
    tensorrt_path TEXT,
    xai_artifact_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS training_history (
    run_id VARCHAR(64) PRIMARY KEY,
    job_id UUID REFERENCES pipeline_jobs(job_id) ON DELETE CASCADE,
    algorithm VARCHAR(50) NOT NULL,
    learning_rate DOUBLE PRECISION,
    loss_value DOUBLE PRECISION,
    accuracy DOUBLE PRECISION,
    execution_time_sec DOUBLE PRECISION,
    weights_path TEXT,
    logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
