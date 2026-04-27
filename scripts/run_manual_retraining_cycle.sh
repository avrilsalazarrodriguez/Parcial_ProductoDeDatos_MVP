#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_BUCKET:?MODEL_BUCKET is required}"
: "${SAGEMAKER_ROLE_ARN:?SAGEMAKER_ROLE_ARN is required}"
: "${GLUE_CRAWLER_NAME:?GLUE_CRAWLER_NAME is required}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-ml.m5.xlarge}"

SALES_PATH="${SALES_PATH:-data/raw/sales_train.csv}"
TEST_PATH="${TEST_PATH:-data/raw/test.csv}"
ITEMS_PATH="${ITEMS_PATH:-data/raw/items_en.csv}"
CATEGORIES_PATH="${CATEGORIES_PATH:-data/raw/item_categories_en.csv}"
SHOPS_PATH="${SHOPS_PATH:-data/raw/shops_en.csv}"
SAMPLE_SUBMISSION_PATH="${SAMPLE_SUBMISSION_PATH:-data/raw/sample_submission.csv}"

echo "Starting ModelOps run: ${RUN_ID}"

echo "1/5 Staging CSV inputs to S3"
PYTHONPATH=. uv run python scripts/prepare_new_model_run.py \
  --bucket "$MODEL_BUCKET" \
  --run-id "$RUN_ID" \
  --sales-path "$SALES_PATH" \
  --test-path "$TEST_PATH" \
  --items-path "$ITEMS_PATH" \
  --categories-path "$CATEGORIES_PATH" \
  --shops-path "$SHOPS_PATH" \
  --sample-submission-path "$SAMPLE_SUBMISSION_PATH" \
  --publish-current

echo "2/5 Running SageMaker Processing model flow"
PYTHONPATH=. uv run python scripts/run_model_flow_aws_jobs.py \
  --role-arn "$SAGEMAKER_ROLE_ARN" \
  --bucket "$MODEL_BUCKET" \
  --region "$REGION" \
  --instance-type "$INSTANCE_TYPE" \
  --base-prefix "modelops/runs/$RUN_ID" \
  --sales-key "raw/runs/$RUN_ID/sales_train.csv" \
  --test-key "raw/runs/$RUN_ID/test.csv" \
  --items-key "raw/runs/$RUN_ID/items_en.csv" \
  --categories-key "raw/runs/$RUN_ID/item_categories_en.csv" \
  --shops-key "raw/runs/$RUN_ID/shops_en.csv"

echo "3/5 Publishing predictions/evaluation/model to modelops/latest"
aws s3 sync \
  "s3://$MODEL_BUCKET/modelops/runs/$RUN_ID/predictions/" \
  "s3://$MODEL_BUCKET/modelops/latest/predictions/"

aws s3 sync \
  "s3://$MODEL_BUCKET/modelops/runs/$RUN_ID/evaluation/" \
  "s3://$MODEL_BUCKET/modelops/latest/evaluation/"

aws s3 sync \
  "s3://$MODEL_BUCKET/modelops/runs/$RUN_ID/model/" \
  "s3://$MODEL_BUCKET/modelops/latest/model/"

echo "4/5 Starting Glue crawler"
aws glue start-crawler --name "$GLUE_CRAWLER_NAME" || true

echo "5/5 Done"
echo "RUN_ID=${RUN_ID}"
echo "Latest forecast: s3://$MODEL_BUCKET/modelops/latest/predictions/forecast_detail.parquet"
echo "Latest metrics: s3://$MODEL_BUCKET/modelops/latest/evaluation/model_metrics.json"
