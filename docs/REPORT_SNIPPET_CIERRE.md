# Snippet para reporte: cierre de automatización y MLOps

## Ejecución de modelos desde Streamlit

La solución combina dos patrones de inferencia. Para inferencia individual, la app carga el modelo serializado dentro del contenedor de Streamlit y predice en el momento con cache de recursos. Para batch grande, como catálogo completo o todos los productos de una tienda, los pronósticos se precomputan offline con SageMaker Processing y se guardan en S3 bajo `modelops/latest/`. Streamlit solo lee esas tablas, lo que reduce latencia, evita que el usuario espere entrenamiento o scoring masivo, y controla costos.

## Actualización con nuevos datos

Los datos crudos viven en S3. La ruta estable es `raw/current/` y la ruta de llegada de nuevos lotes es `raw/incoming/`. Cuando llega un CSV nuevo, EventBridge puede iniciar un proyecto de CodeBuild que valida el archivo, lo integra con `raw/current/sales_train.csv`, ejecuta los jobs de SageMaker Processing, genera predicciones y métricas, y actualiza Glue Data Catalog.

## Champion/challenger

Cada corrida se escribe en `modelops/runs/<run_id>/`. Antes de publicar un modelo a `modelops/latest/`, el script `register_and_promote_model.py` calcula métricas agregadas por segmento y producto, compara el candidato contra el champion vigente y solo lo promueve si mejora de acuerdo con la regla configurada. El catálogo queda en S3 como `modelops/registry/champion.json` y `modelops/registry/model_runs.csv`. El dashboard puede mostrar el champion vigente, corridas históricas y métricas segmentadas.

## Capa operacional

RDS PostgreSQL almacena feedback del negocio, historial de exports CFO, eventos de uso y, opcionalmente, tablas relacionales del registro de modelos. Secrets Manager guarda las credenciales de RDS. CloudWatch Logs permite diagnosticar errores de la app, fallas de conexión, problemas de S3 y eventos de uso sin exponer secretos.

## Próximos pasos de producción

Para producción se recomienda activar el trigger S3 de EventBridge, configurar alertas de CloudWatch/SNS, endurecer permisos IAM de mínimo privilegio, agregar retención de logs y monitoreo de drift de datos/modelo. El MVP deja esas extensiones preparadas sin sobrecargar la arquitectura inicial.
