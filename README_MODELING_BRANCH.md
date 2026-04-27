# Rama de modelos: `feature/modelops-segmented-retraining`

Esta rama agrega el flujo de modelos para el MVP del examen usando los datos traducidos de Kaggle.

## Datos esperados

Coloca localmente los archivos en `data/raw/`:

```text
sales_train.csv
test.csv
items_en.csv
item_categories_en.csv
shops_en.csv
sample_submission.csv
```

No subas esos archivos al repo.

## Comandos locales

```bash
uv sync
uv add sagemaker boto3 s3fs awswrangler pyyaml
uv add --dev pytest
```

```bash
uv run python scripts/run_model_flow_local.py \
  --sales-path data/raw/sales_train.csv \
  --test-path data/raw/test.csv \
  --items-path data/raw/items_en.csv \
  --categories-path data/raw/item_categories_en.csv \
  --shops-path data/raw/shops_en.csv \
  --work-dir data/modelops
```

## Comandos AWS

Crear bucket, Glue Data Catalog, Glue Crawler y SageMaker role:

```bash
aws cloudformation deploy \
  --template-file infra/modelops-stack.yaml \
  --stack-name pfs-modelops \
  --capabilities CAPABILITY_NAMED_IAM
```

Obtener outputs:

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
```

Subir CSVs:

```bash
uv run python scripts/upload_model_inputs_to_s3.py \
  --bucket "$MODEL_BUCKET" \
  --sales-path data/raw/sales_train.csv \
  --test-path data/raw/test.csv \
  --items-path data/raw/items_en.csv \
  --categories-path data/raw/item_categories_en.csv \
  --shops-path data/raw/shops_en.csv \
  --sample-submission-path data/raw/sample_submission.csv
```

Ejecutar jobs secuenciales de SageMaker Processing:

```bash
uv run python scripts/run_model_flow_aws_jobs.py \
  --role-arn "$SAGEMAKER_ROLE_ARN" \
  --bucket "$MODEL_BUCKET" \
  --region us-east-1 \
  --instance-type ml.m5.large
```

Catalogar outputs en Glue:

```bash
aws glue start-crawler --name "$GLUE_CRAWLER_NAME"
```

## Outputs esperados

```text
s3://$MODEL_BUCKET/modelops/features/
s3://$MODEL_BUCKET/modelops/model/model_bundle.joblib
s3://$MODEL_BUCKET/modelops/evaluation/model_metrics.json
s3://$MODEL_BUCKET/modelops/evaluation/evaluation_by_segment.parquet
s3://$MODEL_BUCKET/modelops/evaluation/evaluation_by_item.parquet
s3://$MODEL_BUCKET/modelops/predictions/forecast_detail.parquet
s3://$MODEL_BUCKET/modelops/predictions/submission.csv
```

## Decisión de modelado

- Modelo global para todos los pares tienda-producto.
- Modelos por `item_category_id` solo cuando hay suficiente volumen.
- Baseline naive: `cnt_lag_1`.
- Se conserva el modelo de segmento solo si mejora contra global y naive.
- El dashboard no entrena: solo consume outputs precomputados desde S3/Glue.
