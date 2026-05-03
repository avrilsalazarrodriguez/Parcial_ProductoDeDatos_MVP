# Checklist de pruebas en AWS

## S3 / ModelOps

```bash
aws s3 ls s3://$MODEL_BUCKET/modelops/latest/predictions/
aws s3 ls s3://$MODEL_BUCKET/modelops/latest/evaluation/
aws s3 ls s3://$MODEL_BUCKET/modelops/latest/model/
```

## Loader local

```bash
PYTHONPATH=. uv run python scripts/smoke_test_modelops_loader.py \
  --bucket "$MODEL_BUCKET" \
  --prefix modelops/latest
```

## ECS

```bash
aws ecs describe-services \
  --cluster "$ECS_CLUSTER" \
  --services "$ECS_SERVICE" \
  --query "services[0].{status:status,running:runningCount,desired:desiredCount,deployments:deployments[*].rolloutState}"
```

## CloudWatch logs

```bash
aws logs describe-log-groups --query "logGroups[].logGroupName" --output table
aws logs tail <LOG_GROUP_NAME> --follow
```

## App browser tests

- URL pública abre.
- Resumen muestra métricas ModelOps.
- Batch CFO muestra registros de forecast.
- Botón CFO guarda CSV en S3.
- RDS muestra historial de batch exports.
- Evaluación muestra `evaluation_by_segment` y `evaluation_by_item`.
- Feedback guarda observaciones en RDS.
- Productos problemáticos se leen desde RDS.

## Screenshots

- URL pública abierta.
- ECS service running.
- ECR con imagen publicada.
- Task Definition con env vars ModelOps.
- S3 `modelops/latest`.
- S3 `app/batch_exports` después de exportar.
- RDS disponible.
- Secrets Manager, sin mostrar secretos.
- CloudWatch logs.
- Glue Data Catalog.
