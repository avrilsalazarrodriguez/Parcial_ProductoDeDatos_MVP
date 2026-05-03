# Flujo final: ModelOps → S3 latest → Streamlit en ECS

```text
CSV + metadata traducida
        ↓
S3 raw/current/
        ↓
SageMaker Processing
  1. Build features
  2. Train global + segmented models
  3. Score batch
        ↓
S3 modelops/predictions, modelops/evaluation, modelops/model
        ↓
S3 modelops/latest/
        ↓
ECS Fargate Streamlit app
        ├── Batch CFO lee forecast_detail.parquet
        ├── Evaluación lee evaluation_by_segment/item + model_metrics
        ├── Exports CFO se guardan en S3 app/batch_exports
        └── Feedback + historial se guarda en RDS
```

## Decisiones de diseño

- Inferencia individual: se mantiene dentro del contenedor con `artifacts/model.joblib`.
- Batch grande: se lee desde S3 como predicción precomputada para no degradar la experiencia de usuario.
- Evaluación: se lee desde S3 para mostrar performance global, por segmento y por producto.
- RDS: se conserva para feedback, historial de exports y productos problemáticos.
- ECR/ECS: siguen siendo responsabilidad del despliegue de la app.
- SageMaker Processing: se conserva como cómputo pesado fuera de Streamlit.

## Variables de entorno de ECS

```text
USE_MODELOPS_S3=true
MODEL_BUCKET=<bucket de pfs-modelops>
MODELOPS_PREFIX=modelops/latest
MODELOPS_REGISTRY_PREFIX=modelops/registry
APP_EXPORT_BUCKET=<bucket de pfs-modelops>
APP_EXPORT_PREFIX=app/batch_exports
AWS_REGION=us-east-1
```

## Rutas que consume la app

```text
s3://<MODEL_BUCKET>/modelops/latest/predictions/forecast_detail.parquet
s3://<MODEL_BUCKET>/modelops/latest/predictions/forecast_summary_by_category.parquet
s3://<MODEL_BUCKET>/modelops/latest/predictions/forecast_summary_by_shop_segment.parquet
s3://<MODEL_BUCKET>/modelops/latest/evaluation/evaluation_by_segment.parquet
s3://<MODEL_BUCKET>/modelops/latest/evaluation/evaluation_by_item.parquet
s3://<MODEL_BUCKET>/modelops/latest/evaluation/model_metrics.json
```

## Qué probar en AWS

- La URL pública abre.
- `Batch CFO` usa datos desde S3.
- `Generar archivo CFO` escribe en `app/batch_exports/`.
- RDS registra historial del batch.
- La evaluación muestra performance por segmento/producto.
- CloudWatch muestra logs sin credenciales ni secretos.
