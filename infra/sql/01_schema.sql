-- 01_schema.sql
-- Modelo relacional mínimo para el MVP de pronóstico de ventas.

-- Esta base de datos no guarda el dataset completo de entrenamiento.
-- Los datos grandes viven en S3/Glue. RDS guarda información operacional:
-- feedback de negocio, archivos batch generados y metadata del modelo.

DROP TABLE IF EXISTS business_feedback;
DROP TABLE IF EXISTS batch_exports;
DROP TABLE IF EXISTS model_metadata;
DROP TABLE IF EXISTS problem_products;
DROP TABLE IF EXISTS forecast_results;

-- Tabla con predicciones del mes futuro que consulta la aplicación.
-- Se carga desde data/prep/test_pairs.parquet + data/predictions/submission.csv.
CREATE TABLE forecast_results (
    forecast_id SERIAL PRIMARY KEY,
    shop_id INTEGER NOT NULL,
    item_id INTEGER NOT NULL,
    prediction DOUBLE PRECISION NOT NULL,
    forecast_month TEXT NOT NULL DEFAULT 'next_month',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Tabla con productos o pares tienda-producto que conviene revisar.
-- Se genera desde los errores altos del conjunto de validación.
CREATE TABLE problem_products (
    problem_id SERIAL PRIMARY KEY,
    shop_id INTEGER NOT NULL,
    item_id INTEGER NOT NULL,
    y_true DOUBLE PRECISION,
    prediction DOUBLE PRECISION,
    abs_error DOUBLE PRECISION,
    reason TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para capturar observaciones del negocio desde la UI.
-- Esta es la tabla más importante para cumplir el requisito de feedback.
CREATE TABLE business_feedback (
    feedback_id SERIAL PRIMARY KEY,
    shop_id INTEGER NOT NULL,
    item_id INTEGER NOT NULL,
    issue_type TEXT NOT NULL,
    comment TEXT,
    analyst_name TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para registrar archivos batch generados para negocio/CFO.
-- En AWS guardaremos el path de S3 donde queda el CSV.
CREATE TABLE batch_exports (
    export_id SERIAL PRIMARY KEY,
    scope TEXT NOT NULL,
    shop_id INTEGER,
    records_count INTEGER NOT NULL,
    total_prediction DOUBLE PRECISION NOT NULL,
    s3_uri TEXT,
    status TEXT NOT NULL DEFAULT 'generated',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Tabla con metadata del modelo activo.
-- Sirve para documentar qué modelo usa la app y cómo se evaluó.
CREATE TABLE model_metadata (
    model_id SERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,
    model_type TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    rmse_model DOUBLE PRECISION,
    rmse_naive DOUBLE PRECISION,
    mae_model DOUBLE PRECISION,
    notes TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
