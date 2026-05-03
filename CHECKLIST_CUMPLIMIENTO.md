# Checklist de cierre total del MVP

## Restricciones duras del caso

| Punto | Estado objetivo | Evidencia esperada |
|---|---:|---|
| Streamlit con URL pública | Completo | URL pública abierta en browser |
| ECS Fargate | Completo | ECS service running=1 desired=1 |
| ECR | Completo | Repo ECR con tags `latest` y `modelops-s3` |
| S3 | Completo | `modelops/latest/`, `modelops/registry/`, `app/batch_exports/` |
| RDS | Completo | Tablas con feedback/export history/model registry opcional |
| Secrets Manager | Completo | Secret de RDS creado y usado por app |
| CloudFormation | Completo | stacks `pfs-mvp-rds`, `pfs-mvp-ecr`, `pfs-mvp-app`, `pfs-modelops-automation-complete` |
| Glue Data Catalog | Completo | crawler ejecutado y tablas creadas |
| Batch CFO | Completo | CSV generado y guardado en S3 |
| Evaluación vs ground truth | Completo | App muestra RMSE/MAE/model metrics |
| KPIs por segmento/producto | Completo | App muestra `evaluation_by_segment` y `evaluation_by_item` |
| Feedback en RDS | Completo | Observación guardada y leída en app |
| Nuevo CSV / retraining | Completo | CSV en `raw/incoming/` dispara o permite ejecutar CodeBuild |
| Champion/challenger | Completo | `champion.json` y `model_runs.csv` con promoted/rejected |
| Model registry visible | Completo | App muestra model_id/champion/model_runs |
| Logs | Completo | CloudWatch log group con eventos sin secretos |
| Diagramas draw.io | Completo | `docs/architecture_final.drawio` y `docs/erd_rds_modelops.drawio` |
| Reporte | Completo | `reporte.md` o `reporte.pdf` con screenshots |
