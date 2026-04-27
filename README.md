# README — Handoff de ModelOps y rutas para integración con Streamlit

## 1. Objetivo de este README

Este documento resume lo que se hizo en la rama de modelos, qué rutas S3 debe consumir el dashboard de Streamlit, qué recursos AWS quedaron creados y qué falta para cerrar la integración del MVP.

La idea central de esta parte es separar el cómputo pesado del dashboard:

```text
Datos CSV reales
  ↓
S3 raw/current/
  ↓
SageMaker Processing
  ↓
Feature engineering + entrenamiento + evaluación + batch scoring
  ↓
S3 modelops/features/
S3 modelops/model/
S3 modelops/evaluation/
S3 modelops/predictions/
  ↓
Glue Data Catalog
  ↓
Streamlit consume forecast y métricas como tablas precomputadas
```

Esto evita que Streamlit entrene o prediga todo en cada clic. El dashboard solo debe leer resultados ya calculados.

---

## 2. Rama y contexto

Rama de trabajo de modelos:

```text
feature/modelops-segmented-retraining
```

Rama base del dashboard/app:

```text
partial-mvp
```

Rama común de integración:

```text
develop
```

Flujo esperado de ramas:

```text
main
  ↓
develop
  ↑
feature/modelops-segmented-retraining
  ↑
partial-mvp / dashboard work
```

Este README está pensado para incluirse en el repo y compartirlo con la compañera que desarrolla Streamlit.

---

## 3. Cambios hechos en la parte de modelos

### 3.1 Paquete principal de ModelOps

Se agregó el paquete:

```text
modelops/
```

Archivos esperados:

```text
modelops/__init__.py
modelops/config.py
modelops/feature_builder.py
modelops/io.py
modelops/score_batch.py
modelops/segments.py
modelops/train_segmented.py
```

Propósito de cada archivo:

| Archivo | Propósito |
|---|---|
| `modelops/config.py` | Configuración de features, columnas numéricas y parámetros mínimos del flujo. |
| `modelops/io.py` | Funciones de lectura/escritura local y S3. |
| `modelops/segments.py` | Construcción de segmentos usando `item_category_id` / `item_category_name`, con fallback derivado si no hay metadata. |
| `modelops/feature_builder.py` | Genera features mensuales desde ventas, test y metadata traducida. |
| `modelops/train_segmented.py` | Entrena modelo global y modelos por categoría/segmento; evalúa contra baseline naive. |
| `modelops/score_batch.py` | Genera forecast batch, summaries y `submission.csv`. |

---

### 3.2 Scripts agregados

Se agregaron scripts para ejecución local, subida a S3 y ejecución en AWS:

```text
scripts/run_model_flow_local.py
scripts/upload_model_inputs_to_s3.py
scripts/run_model_flow_aws_jobs.py
```

También se agregaron entrypoints específicos para SageMaker Processing:

```text
scripts/sagemaker_entrypoints/build_features.py
scripts/sagemaker_entrypoints/train_segmented.py
scripts/sagemaker_entrypoints/score_batch.py
```

Estos entrypoints resuelven el problema de imports dentro del contenedor de SageMaker, porque agregan el root del proyecto al `sys.path` antes de importar `modelops`.

---

### 3.3 Infraestructura agregada

Se agregó:

```text
infra/modelops-stack.yaml
```

Este stack crea:

- bucket S3 para datos y outputs de ModelOps;
- Glue Database;
- Glue Crawler;
- SageMaker Execution Role.

También se preparó un overlay opcional de automatización:

```text
infra/modelops-automation-stack.yaml
buildspec-modelops.yml
scripts/prepare_new_model_run.py
scripts/run_manual_retraining_cycle.sh
scripts/create_holdout_incoming_batch.py
scripts/append_incoming_batch_to_sales.py
scripts/print_modelops_outputs.sh
```

Ese overlay sirve para automatizar futuras corridas manuales o programadas con CodeBuild + EventBridge.

---

### 3.4 Documentación agregada

Se agregó documentación de apoyo:

```text
docs/MODELOPS_FLOW.md
docs/AUTOMATION_FLOW.md
README_MODELING_BRANCH.md
README_AUTOMATION.md
```

Este archivo actual, `README_MODELOPS_HANDOFF.md`, resume todo para integración con el dashboard.

---

### 3.5 Dependencias

Se agregaron o usaron dependencias para:

```text
sagemaker
boto3
s3fs
awswrangler
pyyaml
pytest
```

Para SageMaker Processing se simplificó `requirements.txt` para evitar problemas de resolución de dependencias dentro del contenedor:

```text
pandas
pyarrow
scikit-learn
joblib
boto3
typing
```

---

## 4. Datos usados

Los archivos CSV reales usados para el flujo fueron:

```text
data/raw/sales_train.csv
data/raw/test.csv
data/raw/items_en.csv
data/raw/item_categories_en.csv
data/raw/shops_en.csv
data/raw/sample_submission.csv
```

Estos archivos **no deben subirse a GitHub**.

El `.gitignore` debe excluir como mínimo:

```gitignore
data/raw/*.csv
data/modelops/
*.parquet
*.joblib
.env
.venv/
__pycache__/
```

---

## 5. Correcciones hechas durante la ejecución

### 5.1 Problema de imports locales

Al correr localmente, algunos scripts no encontraban el paquete `modelops`.

Solución usada:

```bash
PYTHONPATH=. uv run python scripts/run_model_flow_local.py ...
```

Y para otros scripts:

```bash
PYTHONPATH=. uv run python scripts/upload_model_inputs_to_s3.py ...
PYTHONPATH=. uv run python scripts/run_model_flow_aws_jobs.py ...
```

---

### 5.2 Problema de bucket default de SageMaker

SageMaker intentó subir código al bucket default:

```text
sagemaker-us-east-1-494321812137
```

pero el rol no tenía permisos ahí.

Solución aplicada en `scripts/run_model_flow_aws_jobs.py`:

```python
boto_session = boto3.Session(region_name=args.region)
session = sagemaker.Session(
    boto_session=boto_session,
    default_bucket=args.bucket,
)
```

Así SageMaker usa el bucket del stack de ModelOps.

---

### 5.3 Problema de imports dentro de SageMaker Processing

SageMaker ejecutaba directamente:

```text
modelops/feature_builder.py
```

Eso hacía que el contenedor no pudiera importar `modelops`.

Solución aplicada:

Se crearon wrappers en:

```text
scripts/sagemaker_entrypoints/
```

y se modificó `scripts/run_model_flow_aws_jobs.py` para ejecutar:

```python
code="scripts/sagemaker_entrypoints/build_features.py"
code="scripts/sagemaker_entrypoints/train_segmented.py"
code="scripts/sagemaker_entrypoints/score_batch.py"
```

---

## 6. Recursos AWS creados para ModelOps

### 6.1 Stack principal

Stack:

```text
pfs-modelops
```

Recursos principales:

- S3 bucket;
- SageMaker Execution Role;
- Glue Database;
- Glue Crawler;
- CloudWatch Logs de los jobs de SageMaker Processing.

---

### 6.2 Bucket principal

Bucket usado:

```text
pfs-modelops-494321812137-us-east-1
```

Variable recomendada:

```bash
export MODEL_BUCKET=pfs-modelops-494321812137-us-east-1
```

---

### 6.3 SageMaker role

Para obtenerlo automáticamente:

```bash
export SAGEMAKER_ROLE_ARN=$(aws cloudformation describe-stacks \
  --stack-name pfs-modelops \
  --query "Stacks[0].Outputs[?OutputKey=='SageMakerExecutionRoleArn'].OutputValue" \
  --output text)
```

---

### 6.4 Glue Crawler

Crawler esperado:

```text
pfs-modelops-crawler
```

Para obtenerlo automáticamente:

```bash
export GLUE_CRAWLER_NAME=$(aws cloudformation describe-stacks \
  --stack-name pfs-modelops \
  --query "Stacks[0].Outputs[?OutputKey=='GlueCrawlerName'].OutputValue" \
  --output text)
```

---

### 6.5 Glue Database

No asumir el nombre. Obtenerlo con:

```bash
export GLUE_DATABASE_NAME=$(aws cloudformation describe-stacks \
  --stack-name pfs-modelops \
  --query "Stacks[0].Outputs[?OutputKey=='GlueDatabaseName'].OutputValue" \
  --output text)
```

Si sale vacío, usar:

```bash
export GLUE_DATABASE_NAME=$(aws glue get-crawler \
  --name "$GLUE_CRAWLER_NAME" \
  --query "Crawler.DatabaseName" \
  --output text)
```

Verificar:

```bash
echo $GLUE_DATABASE_NAME
```

---

## 7. Comandos que ya se ejecutaron o deben poder repetirse

### 7.1 Obtener outputs del stack

```bash
export MODEL_BUCKET=$(aws cloudformation describe-stacks \
  --stack-name pfs-modelops \
  --query "Stacks[0].Outputs[?OutputKey=='BucketName'].OutputValue" \
  --output text)

export SAGEMAKER_ROLE_ARN=$(aws cloudformation describe-stacks \
  --stack-name pfs-modelops \
  --query "Stacks[0].Outputs[?OutputKey=='SageMakerExecutionRoleArn'].OutputValue" \
  --output text)

export GLUE_CRAWLER_NAME=$(aws cloudformation describe-stacks \
  --stack-name pfs-modelops \
  --query "Stacks[0].Outputs[?OutputKey=='GlueCrawlerName'].OutputValue" \
  --output text)

export GLUE_DATABASE_NAME=$(aws glue get-crawler \
  --name "$GLUE_CRAWLER_NAME" \
  --query "Crawler.DatabaseName" \
  --output text)
```

---

### 7.2 Subir inputs CSV a S3

```bash
PYTHONPATH=. uv run python scripts/upload_model_inputs_to_s3.py \
  --bucket "$MODEL_BUCKET" \
  --sales-path data/raw/sales_train.csv \
  --test-path data/raw/test.csv \
  --items-path data/raw/items_en.csv \
  --categories-path data/raw/item_categories_en.csv \
  --shops-path data/raw/shops_en.csv \
  --sample-submission-path data/raw/sample_submission.csv
```

Esto deja los CSV en:

```text
s3://$MODEL_BUCKET/raw/current/
```

---

### 7.3 Correr flujo completo en SageMaker Processing

```bash
PYTHONPATH=. uv run python scripts/run_model_flow_aws_jobs.py \
  --role-arn "$SAGEMAKER_ROLE_ARN" \
  --bucket "$MODEL_BUCKET" \
  --region us-east-1 \
  --instance-type ml.m5.xlarge
```

El flujo corre tres jobs:

```text
1. Build features
2. Train segmented models
3. Score batch
```

Los jobs se observaron como `Completed`.

---

## 8. Rutas S3 para entregar al dashboard

Bucket base:

```text
s3://pfs-modelops-494321812137-us-east-1
```

### 8.1 Rutas principales para Streamlit

Estas son las rutas más importantes que debe usar el dashboard:

```text
FORECAST_DETAIL=s3://pfs-modelops-494321812137-us-east-1/modelops/predictions/forecast_detail.parquet
FORECAST_SUMMARY_CATEGORY=s3://pfs-modelops-494321812137-us-east-1/modelops/predictions/forecast_summary_by_category.parquet
FORECAST_SUMMARY_SHOP_SEGMENT=s3://pfs-modelops-494321812137-us-east-1/modelops/predictions/forecast_summary_by_shop_segment.parquet
SUBMISSION=s3://pfs-modelops-494321812137-us-east-1/modelops/predictions/submission.csv

EVALUATION_DETAIL=s3://pfs-modelops-494321812137-us-east-1/modelops/evaluation/evaluation_detail.parquet
EVALUATION_BY_SEGMENT=s3://pfs-modelops-494321812137-us-east-1/modelops/evaluation/evaluation_by_segment.parquet
EVALUATION_BY_ITEM=s3://pfs-modelops-494321812137-us-east-1/modelops/evaluation/evaluation_by_item.parquet
MODEL_METRICS=s3://pfs-modelops-494321812137-us-east-1/modelops/evaluation/model_metrics.json
```

---

### 8.2 Rutas de features, modelo y metadata

Estas rutas no son necesariamente para el dashboard, pero sirven para auditoría y debugging:

```text
FEATURES_TRAIN=s3://pfs-modelops-494321812137-us-east-1/modelops/features/train.parquet
FEATURES_VALID=s3://pfs-modelops-494321812137-us-east-1/modelops/features/valid.parquet
INFERENCE_FEATURES=s3://pfs-modelops-494321812137-us-east-1/modelops/features/inference_features.parquet
INFERENCE_PAIRS=s3://pfs-modelops-494321812137-us-east-1/modelops/features/inference_pairs.parquet
PRODUCT_SEGMENTS=s3://pfs-modelops-494321812137-us-east-1/modelops/features/product_segments.parquet
SHOP_DIMENSION=s3://pfs-modelops-494321812137-us-east-1/modelops/features/shop_dimension.parquet

MODEL_BUNDLE=s3://pfs-modelops-494321812137-us-east-1/modelops/model/model_bundle.joblib
```

---

### 8.3 Comando para imprimir rutas con variables

```bash
cat <<EOT
Ya terminó el flujo ModelOps en SageMaker Processing.

Glue Database:
$GLUE_DATABASE_NAME

Forecast:
s3://$MODEL_BUCKET/modelops/predictions/forecast_detail.parquet
s3://$MODEL_BUCKET/modelops/predictions/forecast_summary_by_category.parquet
s3://$MODEL_BUCKET/modelops/predictions/forecast_summary_by_shop_segment.parquet

Evaluación:
s3://$MODEL_BUCKET/modelops/evaluation/evaluation_by_segment.parquet
s3://$MODEL_BUCKET/modelops/evaluation/evaluation_by_item.parquet
s3://$MODEL_BUCKET/modelops/evaluation/model_metrics.json

Submission:
s3://$MODEL_BUCKET/modelops/predictions/submission.csv
EOT
```

---

## 9. Rutas recomendadas si se activa automatización con `latest/`

Si se copia el overlay de automatización y se usa `run_manual_retraining_cycle.sh`, el dashboard debería leer rutas estables bajo `modelops/latest/`:

```text
LATEST_FORECAST_DETAIL=s3://pfs-modelops-494321812137-us-east-1/modelops/latest/predictions/forecast_detail.parquet
LATEST_FORECAST_SUMMARY_CATEGORY=s3://pfs-modelops-494321812137-us-east-1/modelops/latest/predictions/forecast_summary_by_category.parquet
LATEST_FORECAST_SUMMARY_SHOP_SEGMENT=s3://pfs-modelops-494321812137-us-east-1/modelops/latest/predictions/forecast_summary_by_shop_segment.parquet
LATEST_SUBMISSION=s3://pfs-modelops-494321812137-us-east-1/modelops/latest/predictions/submission.csv

LATEST_EVALUATION_BY_SEGMENT=s3://pfs-modelops-494321812137-us-east-1/modelops/latest/evaluation/evaluation_by_segment.parquet
LATEST_EVALUATION_BY_ITEM=s3://pfs-modelops-494321812137-us-east-1/modelops/latest/evaluation/evaluation_by_item.parquet
LATEST_MODEL_METRICS=s3://pfs-modelops-494321812137-us-east-1/modelops/latest/evaluation/model_metrics.json
```

Estas rutas son mejores para la app porque no cambian entre corridas.

---

## 10. Cómo puede leer el dashboard estos archivos

### Opción rápida: leer directo desde S3 con pandas

```python
import pandas as pd

forecast_detail = pd.read_parquet(
    "s3://pfs-modelops-494321812137-us-east-1/modelops/predictions/forecast_detail.parquet"
)

evaluation_by_segment = pd.read_parquet(
    "s3://pfs-modelops-494321812137-us-east-1/modelops/evaluation/evaluation_by_segment.parquet"
)
```

Para que esto funcione dentro de ECS/Fargate, el Task Role de la app debe tener permisos:

```text
s3:GetObject
s3:ListBucket
```

sobre el bucket:

```text
pfs-modelops-494321812137-us-east-1
```

---

### Opción alternativa: descargar a local para pruebas

```bash
aws s3 cp s3://pfs-modelops-494321812137-us-east-1/modelops/predictions/forecast_detail.parquet data/app/forecast_detail.parquet
aws s3 cp s3://pfs-modelops-494321812137-us-east-1/modelops/evaluation/model_metrics.json data/app/model_metrics.json
```

---

## 11. Validaciones pendientes

### 11.1 Validar outputs en S3

```bash
aws s3 ls s3://$MODEL_BUCKET/modelops/features/ --recursive
aws s3 ls s3://$MODEL_BUCKET/modelops/model/ --recursive
aws s3 ls s3://$MODEL_BUCKET/modelops/evaluation/ --recursive
aws s3 ls s3://$MODEL_BUCKET/modelops/predictions/ --recursive
```

---

### 11.2 Descargar métricas

```bash
aws s3 cp s3://$MODEL_BUCKET/modelops/evaluation/model_metrics.json .
cat model_metrics.json
```

Revisar:

```text
rmse_naive
rmse_global
rmse_final
n_segment_models
```

---

### 11.3 Correr Glue Crawler

```bash
aws glue start-crawler --name "$GLUE_CRAWLER_NAME"
```

Ver estado:

```bash
aws glue get-crawler \
  --name "$GLUE_CRAWLER_NAME" \
  --query "Crawler.State"
```

Listar tablas:

```bash
aws glue get-tables \
  --database-name "$GLUE_DATABASE_NAME" \
  --query "TableList[].Name"
```

---

## 12. Documentación en el reporte



> Se implementó un flujo de ModelOps en AWS usando SageMaker Processing para ejecutar feature engineering, entrenamiento segmentado, evaluación contra baseline naive y batch scoring. El entrenamiento se realiza fuera de Streamlit para no degradar la experiencia de usuario. Los outputs se guardan en S3 y se catalogan con Glue Data Catalog para que el dashboard consuma pronósticos y métricas como tablas precomputadas.

segmentación:

> El dataset corresponde a ventas retail por tienda-producto, no a subsegmentos de población. Por ello, la segmentación del modelo se hace por categorías de producto usando `items_en.csv` e `item_categories_en.csv`. El flujo entrena un modelo global y modelos segmentados por categoría cuando existe suficiente volumen de entrenamiento y validación. Cada modelo segmentado se conserva únicamente si mejora contra el modelo global o contra un baseline naive.

automatización:

> Para automatizar el reentrenamiento, el sistema puede recibir nuevos CSV en `raw/current/` o `raw/runs/<run_id>/`, ejecutar los tres jobs de SageMaker Processing, publicar los outputs en `modelops/runs/<run_id>/` y sincronizar los resultados vigentes a `modelops/latest/`. Esta ruta puede ejecutarse manualmente o programarse con CodeBuild + EventBridge.

---

## 13. Evidencias que faltan o que deben guardarse

Tomar screenshots de:

```text
1. CloudFormation stack pfs-modelops en CREATE_COMPLETE
2. SageMaker Processing jobs en Completed
3. S3 raw/current/
4. S3 modelops/features/
5. S3 modelops/evaluation/
6. S3 modelops/predictions/
7. model_metrics.json abierto
8. Glue Crawler ejecutado
9. Glue tables creadas
10. CloudWatch logs de un job
```

---

## 14. Próximos pasos técnicos

### Paso 1: Pasar rutas al dashboard

Entregar a la compañera las rutas de la sección 8.

### Paso 2: Dar permisos al Task Role de ECS

La app de Streamlit en ECS debe poder leer:

```text
s3://pfs-modelops-494321812137-us-east-1/modelops/predictions/
s3://pfs-modelops-494321812137-us-east-1/modelops/evaluation/
```

Permisos mínimos:

```text
s3:GetObject
s3:ListBucket
```

### Paso 3: Conectar Streamlit a outputs

En el dashboard, crear funciones como:

```python
@st.cache_data(ttl=600)
def load_forecast_detail():
    return pd.read_parquet(FORECAST_DETAIL_S3_URI)
```

### Paso 4: Actualizar README y reporte

Agregar:

- flujo de modelos;
- rutas S3;
- métricas;
- justificación de precomputar outputs;
- screenshots AWS.

### Paso 5: Automatización opcional

Copiar el overlay de automatización:

```text
scripts/prepare_new_model_run.py
scripts/run_manual_retraining_cycle.sh
buildspec-modelops.yml
infra/modelops-automation-stack.yaml
```

Probar primero ejecución manual:

```bash
PYTHONPATH=. ./scripts/run_manual_retraining_cycle.sh
```

Luego, si se quiere programar:

```bash
aws cloudformation deploy \
  --template-file infra/modelops-automation-stack.yaml \
  --stack-name pfs-modelops-automation \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    ModelBucket="$MODEL_BUCKET" \
    SageMakerRoleArn="$SAGEMAKER_ROLE_ARN" \
    GlueCrawlerName="$GLUE_CRAWLER_NAME" \
    InstanceType=ml.m5.xlarge \
    ScheduleExpression="rate(7 days)"
```

---

## 15. Qué servicios AWS quedan cubiertos por la parte de modelos

| Servicio | Uso en esta parte |
|---|---|
| Amazon S3 | Datos crudos, features, modelos, evaluación, predicciones. |
| SageMaker Processing | Feature engineering, entrenamiento, evaluación y scoring batch. |
| AWS Glue Data Catalog | Catálogo de outputs analíticos. |
| AWS CloudFormation | Stack `pfs-modelops`. |
| IAM Role | Rol de ejecución para SageMaker. |
| CloudWatch Logs | Logs de Processing Jobs. |

Servicios que se cubren principalmente en la parte del dashboard:

| Servicio | Uso esperado |
|---|---|
| Amazon ECR | Imagen Docker de Streamlit. |
| ECS Fargate | Hosting de Streamlit. |
| RDS | Feedback, export jobs, model metadata si deciden guardarla. |
| Secrets Manager | Credenciales de RDS. |

EC2 no es necesario para este MVP.

---

## 16. Estado actual resumido

Estado de la parte de modelos:

```text
SageMaker Processing jobs: Completed
CSV reales subidos a S3: sí
Outputs de modelos generados: sí, validar rutas S3
Glue Crawler: pendiente si no se ha corrido
Rutas para dashboard: disponibles
Automatización programada: preparada como siguiente paso
```

---

## 17. Checklist final de tu parte

```text
[ ] Confirmar outputs en S3.
[ ] Descargar y revisar model_metrics.json.
[ ] Correr Glue Crawler.
[ ] Confirmar Glue tables.
[ ] Pasar rutas S3 a la compañera.
[ ] Agregar README_MODELOPS_HANDOFF.md al repo.
[ ] Tomar screenshots de evidencia.
[ ] Documentar flujo en reporte.
[ ] Hacer commit y push.
[ ] Abrir PR hacia develop.
```

---

## 18. Commit sugerido

```bash
git status

git add modelops/
git add scripts/
git add infra/modelops-stack.yaml
git add docs/
git add tests/
git add README_MODELING_BRANCH.md
git add README_MODELOPS_HANDOFF.md
git add requirements.txt
git add pyproject.toml uv.lock

git commit -m "Add ModelOps handoff and segmented retraining flow"
git push
```

---

## 19. Mensaje corto para la compañera

```text
Ya terminó el flujo de modelos en SageMaker Processing.

Los outputs principales para conectar al dashboard son:

Forecast:
s3://pfs-modelops-494321812137-us-east-1/modelops/predictions/forecast_detail.parquet
s3://pfs-modelops-494321812137-us-east-1/modelops/predictions/forecast_summary_by_category.parquet
s3://pfs-modelops-494321812137-us-east-1/modelops/predictions/forecast_summary_by_shop_segment.parquet

Evaluación:
s3://pfs-modelops-494321812137-us-east-1/modelops/evaluation/evaluation_by_segment.parquet
s3://pfs-modelops-494321812137-us-east-1/modelops/evaluation/evaluation_by_item.parquet
s3://pfs-modelops-494321812137-us-east-1/modelops/evaluation/model_metrics.json

Submission:
s3://pfs-modelops-494321812137-us-east-1/modelops/predictions/submission.csv

Falta que el dashboard lea estas rutas desde S3 y que el Task Role de ECS tenga permisos s3:GetObject/ListBucket sobre el bucket.
```
