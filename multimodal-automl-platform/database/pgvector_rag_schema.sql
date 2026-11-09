CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS training_log_embeddings (
    log_id SERIAL PRIMARY KEY,
    run_id VARCHAR(64) REFERENCES training_history(run_id) ON DELETE CASCADE,
    log_summary TEXT NOT NULL,
    hyperparameters JSONB,
    metrics JSONB,
    embedding vector(384),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_log_embedding_hnsw 
ON training_log_embeddings USING hnsw (embedding vector_cosine_ops);
