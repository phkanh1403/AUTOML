SELECT 
    algorithm,
    COUNT(run_id) AS total_runs,
    ROUND(AVG(accuracy)::numeric, 4) AS avg_accuracy,
    ROUND(MAX(accuracy)::numeric, 4) AS best_accuracy,
    ROUND(AVG(loss_value)::numeric, 4) AS avg_loss,
    ROUND(AVG(execution_time_sec)::numeric, 2) AS avg_time_sec
FROM training_history
GROUP BY algorithm
ORDER BY best_accuracy DESC;

SELECT 
    job_id, modality, status, dropped_columns, created_at
FROM pipeline_jobs
WHERE array_length(dropped_columns, 1) > 0 OR status = 'REJECTED';
