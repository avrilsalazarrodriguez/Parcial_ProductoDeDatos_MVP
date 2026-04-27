#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_BUCKET:?MODEL_BUCKET is required}"
: "${GLUE_DATABASE_NAME:?GLUE_DATABASE_NAME is required}"
: "${GLUE_CRAWLER_NAME:?GLUE_CRAWLER_NAME is required}"

cat <<EOF
# ModelOps outputs

MODEL_BUCKET=s3://$MODEL_BUCKET
GLUE_DATABASE_NAME=$GLUE_DATABASE_NAME
GLUE_CRAWLER_NAME=$GLUE_CRAWLER_NAME

FORECAST_DETAIL=s3://$MODEL_BUCKET/modelops/latest/predictions/forecast_detail.parquet
FORECAST_SUMMARY_CATEGORY=s3://$MODEL_BUCKET/modelops/latest/predictions/forecast_summary_by_category.parquet
FORECAST_SUMMARY_SHOP_SEGMENT=s3://$MODEL_BUCKET/modelops/latest/predictions/forecast_summary_by_shop_segment.parquet
SUBMISSION=s3://$MODEL_BUCKET/modelops/latest/predictions/submission.csv

EVALUATION_DETAIL=s3://$MODEL_BUCKET/modelops/latest/evaluation/evaluation_detail.parquet
EVALUATION_BY_SEGMENT=s3://$MODEL_BUCKET/modelops/latest/evaluation/evaluation_by_segment.parquet
EVALUATION_BY_ITEM=s3://$MODEL_BUCKET/modelops/latest/evaluation/evaluation_by_item.parquet
MODEL_METRICS=s3://$MODEL_BUCKET/modelops/latest/evaluation/model_metrics.json

MODEL_BUNDLE=s3://$MODEL_BUCKET/modelops/latest/model/model_bundle.joblib
EOF
