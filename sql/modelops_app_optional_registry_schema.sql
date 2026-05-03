-- Optional RDS tables for model registry metadata.
-- The MVP can use S3 registry files. Use these tables only if you want RDS
-- to store model run metadata shown in the dashboard.

CREATE TABLE IF NOT EXISTS model_runs (
    model_run_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source_prefix TEXT,
    model_uri TEXT NOT NULL,
    forecast_uri TEXT NOT NULL,
    metrics_uri TEXT NOT NULL,
    rmse_model DOUBLE PRECISION,
    rmse_naive DOUBLE PRECISION,
    wape_model DOUBLE PRECISION,
    wape_naive DOUBLE PRECISION,
    segments_beating_naive_rate DOUBLE PRECISION,
    n_segments INTEGER,
    status TEXT NOT NULL,
    is_champion BOOLEAN NOT NULL DEFAULT FALSE,
    promotion_reason TEXT
);

CREATE TABLE IF NOT EXISTS model_segment_metrics (
    id BIGSERIAL PRIMARY KEY,
    model_run_id TEXT REFERENCES model_runs(model_run_id),
    segment_key TEXT,
    segment_name TEXT,
    n_obs INTEGER,
    rmse_model DOUBLE PRECISION,
    rmse_naive DOUBLE PRECISION,
    mae_model DOUBLE PRECISION,
    mae_naive DOUBLE PRECISION,
    wape_model DOUBLE PRECISION,
    wape_naive DOUBLE PRECISION,
    model_beats_naive BOOLEAN
);
