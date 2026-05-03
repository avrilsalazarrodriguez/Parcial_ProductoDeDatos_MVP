-- Relational schema for the 1C Company Streamlit MVP.
-- Large datasets and model outputs live in S3/Glue. RDS stores operational data.

CREATE TABLE IF NOT EXISTS business_feedback (
    feedback_id SERIAL PRIMARY KEY,
    shop_id INTEGER NOT NULL,
    item_id INTEGER NOT NULL,
    issue_type TEXT NOT NULL,
    comment TEXT NOT NULL,
    analyst_name TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS batch_exports (
    export_id SERIAL PRIMARY KEY,
    scope TEXT NOT NULL,
    shop_id INTEGER,
    records_count INTEGER NOT NULL,
    total_prediction DOUBLE PRECISION NOT NULL,
    s3_uri TEXT,
    status TEXT NOT NULL DEFAULT 'generated',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS app_usage_events (
    event_id SERIAL PRIMARY KEY,
    event_type TEXT NOT NULL,
    shop_id INTEGER,
    item_id INTEGER,
    records_count INTEGER,
    status TEXT NOT NULL DEFAULT 'success',
    message TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS problem_products (
    problem_id SERIAL PRIMARY KEY,
    shop_id INTEGER NOT NULL,
    item_id INTEGER NOT NULL,
    y_true DOUBLE PRECISION,
    prediction DOUBLE PRECISION,
    abs_error DOUBLE PRECISION,
    reason TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS forecast_results (
    forecast_id SERIAL PRIMARY KEY,
    shop_id INTEGER NOT NULL,
    item_id INTEGER NOT NULL,
    prediction DOUBLE PRECISION NOT NULL,
    forecast_month TEXT NOT NULL DEFAULT 'next_month',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_metadata (
    model_id SERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    model_uri TEXT,
    metrics_uri TEXT,
    notes TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_business_feedback_shop_item
    ON business_feedback (shop_id, item_id);

CREATE INDEX IF NOT EXISTS idx_batch_exports_created_at
    ON batch_exports (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_app_usage_events_created_at
    ON app_usage_events (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_problem_products_abs_error
    ON problem_products (abs_error DESC);
