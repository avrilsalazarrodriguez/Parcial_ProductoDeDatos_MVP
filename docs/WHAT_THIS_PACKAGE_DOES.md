# Qué hace este paquete

Este paquete replica la capa de aplicación en tu cuenta de AWS:

```text
S3 ModelOps existente
    ↓
Streamlit Docker image en ECR
    ↓
ECS Fargate + ALB público
    ↓
RDS PostgreSQL + Secrets Manager
    ↓
CloudWatch Logs
```

No borra tu bucket de ModelOps. No reentrena modelos. Solo conecta la app con los outputs ya generados en S3.

## Servicios creados

- RDS PostgreSQL.
- Secret de credenciales de RDS.
- ECR repository.
- ECS Cluster.
- ECS Task Definition.
- ECS Service en Fargate.
- Application Load Balancer público.
- CloudWatch Log Group.
- IAM roles para ECS.

## Rutas S3 esperadas

```text
s3://<MODEL_BUCKET>/modelops/latest/predictions/forecast_detail.parquet
s3://<MODEL_BUCKET>/modelops/latest/evaluation/evaluation_by_segment.parquet
s3://<MODEL_BUCKET>/modelops/latest/evaluation/evaluation_by_item.parquet
s3://<MODEL_BUCKET>/modelops/latest/evaluation/model_metrics.json
s3://<MODEL_BUCKET>/modelops/registry/champion.json
s3://<MODEL_BUCKET>/modelops/registry/model_runs.csv
s3://<MODEL_BUCKET>/app/batch_exports/
```
