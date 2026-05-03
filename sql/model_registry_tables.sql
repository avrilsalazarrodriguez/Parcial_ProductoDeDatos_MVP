-- Optional RDS tables for model registry visibility and governance.
-- The app can use S3 registry directly, but these tables document the relational
-- contract for a production-grade MVP.

CREATE TABLE IF NOT EXISTS model_runs (
    model_run_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status TEXT NOT NULL,
    is_champion BOOLEAN NOT NULL DEFAULT FALSE,
    source_prefix TEXT NOT NULL,
    model_uri TEXT NOT NULL,
    forecast_uri TEXT NOT NULL,
    metrics_uri TEXT NOT NULL,
    weighted_mae DOUBLE PRECISION,
    weighted_naive_mae DOUBLE PRECISION,
    item_weighted_mae DOUBLE PRECISION,
    beats_naive_rate DOUBLE PRECISION,
    n_segments INTEGER,
    n_items_evaluated INTEGER,
    promotion_reason TEXT
);

CREATE TABLE IF NOT EXISTS model_segment_metrics (
    id BIGSERIAL PRIMARY KEY,
    model_run_id TEXT REFERENCES model_runs(model_run_id),
    segment_key TEXT,
    n INTEGER,
    mae DOUBLE PRECISION,
    naive_mae DOUBLE PRECISION,
    beats_naive_rate DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model_item_metrics (
    id BIGSERIAL PRIMARY KEY,
    model_run_id TEXT REFERENCES model_runs(model_run_id),
    item_id INTEGER,
    segment_key TEXT,
    n INTEGER,
    y_true_mean DOUBLE PRECISION,
    pred_mean DOUBLE PRECISION,
    mae DOUBLE PRECISION,
    naive_mae DOUBLE PRECISION,
    beats_naive_rate DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS data_batches (
    batch_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    s3_uri TEXT NOT NULL,
    status TEXT NOT NULL,
    records_count INTEGER,
    notes TEXT
);
