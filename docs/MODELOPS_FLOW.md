# Flujo ModelOps con datos traducidos de Kaggle

## Interpretación correcta

Este no es un problema de población ni de usuarios. Es un problema de forecasting retail por `shop_id` e `item_id`.

Con los archivos traducidos disponibles:

- `sales_train.csv`
- `test.csv`
- `items_en.csv`
- `item_categories_en.csv`
- `shops_en.csv`
- `sample_submission.csv`

la segmentación principal debe hacerse por `item_category_id` y mostrarse con `item_category_name`. Si por alguna razón no estuvieran las categorías, el flujo cae a un `segment_key` derivado de demanda y precio.

## Estrategia de modelos

1. Construir features mensuales de ventas.
2. Entrenar un modelo global con todos los pares tienda-producto.
3. Entrenar modelos por categoría/segmento solo si hay suficiente volumen.
4. Conservar un modelo de segmento únicamente si mejora el RMSE frente al modelo global y el naive.
5. Usar `cnt_lag_1` como baseline naive.
6. Generar forecast batch para todos los pares de `test.csv`.
7. Guardar outputs en S3 para el dashboard.

## Flujo AWS

1. Subir CSVs a S3 en `raw/current/`.
2. Ejecutar SageMaker Processing Job 1: `BuildFeatures`.
3. Ejecutar SageMaker Processing Job 2: `TrainSegmentedModels`.
4. Ejecutar SageMaker Processing Job 3: `ScoreBatch`.
5. Correr Glue Crawler sobre `s3://bucket/modelops/`.
6. Entregar al dashboard los prefijos:
   - `modelops/predictions/forecast_detail.parquet`
   - `modelops/predictions/forecast_summary_by_shop_segment.parquet`
   - `modelops/predictions/forecast_summary_by_category.parquet`
   - `modelops/evaluation/evaluation_detail.parquet`
   - `modelops/evaluation/evaluation_by_segment.parquet`
   - `modelops/evaluation/evaluation_by_item.parquet`
   - `modelops/evaluation/model_metrics.json`

## Outputs principales

- `features/train.parquet`
- `features/valid.parquet`
- `features/inference_features.parquet`
- `features/inference_pairs.parquet`
- `features/product_segments.parquet`
- `features/shop_dimension.parquet`
- `model/model_bundle.joblib`
- `evaluation/evaluation_detail.parquet`
- `evaluation/evaluation_by_segment.parquet`
- `evaluation/evaluation_by_item.parquet`
- `evaluation/model_metrics.json`
- `predictions/forecast_detail.parquet`
- `predictions/forecast_summary_by_shop_segment.parquet`
- `predictions/forecast_summary_by_category.parquet`
- `predictions/submission.csv`
