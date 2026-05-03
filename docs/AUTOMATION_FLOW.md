# Flujo de automatización ModelOps

Este overlay convierte el flujo que ya corriste manualmente en un flujo repetible para nuevos CSVs y para ejecución programada.

## Arquitectura

```text
Nuevo CSV manual o programado
        ↓
S3 raw/runs/<run_id>/ y raw/current/
        ↓
Ejecución manual o CodeBuild programado
        ↓
SageMaker Processing jobs
  1. Build features
  2. Train global + segmented models
  3. Score batch
        ↓
S3 modelops/runs/<run_id>/
        ↓
S3 modelops/latest/
        ↓
Glue Crawler
        ↓
Dashboard Streamlit lee latest
```

## Por qué no usar datos sintéticos para métricas

Para evaluación del modelo, usa datos reales históricos y compara contra baseline naive. Los datos sintéticos solo deben usarse, si acaso, para probar que el pipeline reacciona ante una nueva carga. Una alternativa más defendible es crear un batch de entrada usando un mes histórico real, por ejemplo `date_block_num=33`.

## Qué se automatiza

- Preparación de un nuevo `run_id`.
- Carga de CSVs a S3.
- Ejecución de SageMaker Processing.
- Publicación de outputs en `modelops/latest/`.
- Ejecución de Glue Crawler.
- Ejecución programada con EventBridge + CodeBuild.

## Qué NO hace este overlay

- No modifica el dashboard.
- No crea RDS ni ECS.
- No crea ECR para modelos.
- No reemplaza el flujo de ModelOps que ya funciona.
- No genera datos sintéticos para afirmar mejora del modelo.

## Rutas estables para el dashboard

```text
s3://<MODEL_BUCKET>/modelops/latest/predictions/forecast_detail.parquet
s3://<MODEL_BUCKET>/modelops/latest/predictions/forecast_summary_by_category.parquet
s3://<MODEL_BUCKET>/modelops/latest/predictions/forecast_summary_by_shop_segment.parquet
s3://<MODEL_BUCKET>/modelops/latest/evaluation/evaluation_by_segment.parquet
s3://<MODEL_BUCKET>/modelops/latest/evaluation/evaluation_by_item.parquet
s3://<MODEL_BUCKET>/modelops/latest/evaluation/model_metrics.json
```

## Rutas históricas

Cada corrida queda versionada por `run_id`:

```text
s3://<MODEL_BUCKET>/raw/runs/<run_id>/
s3://<MODEL_BUCKET>/modelops/runs/<run_id>/features/
s3://<MODEL_BUCKET>/modelops/runs/<run_id>/model/
s3://<MODEL_BUCKET>/modelops/runs/<run_id>/evaluation/
s3://<MODEL_BUCKET>/modelops/runs/<run_id>/predictions/
```
