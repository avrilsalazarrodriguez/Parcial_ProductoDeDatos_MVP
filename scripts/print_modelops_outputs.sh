#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_BUCKET:?MODEL_BUCKET is required}"
GLUE_DATABASE_NAME="${GLUE_DATABASE_NAME:-UNKNOWN}"
GLUE_CRAWLER_NAME="${GLUE_CRAWLER_NAME:-UNKNOWN}"
MODELOPS_PREFIX="${MODELOPS_PREFIX:-modelops/latest}"
APP_EXPORT_PREFIX="${APP_EXPORT_PREFIX:-app/batch_exports}"

cat <<EOF
# ModelOps + App routes

MODEL_BUCKET=s3://$MODEL_BUCKET
MODELOPS_PREFIX=$MODELOPS_PREFIX
GLUE_DATABASE_NAME=$GLUE_DATABASE_NAME
GLUE_CRAWLER_NAME=$GLUE_CRAWLER_NAME

# Forecast consumed by Streamlit
FORECAST_DETAIL=s3://$MODEL_BUCKET/$MODELOPS_PREFIX/predictions/forecast_detail.parquet
FORECAST_SUMMARY_CATEGORY=s3://$MODEL_BUCKET/$MODELOPS_PREFIX/predictions/forecast_summary_by_category.parquet
FORECAST_SUMMARY_SHOP_SEGMENT=s3://$MODEL_BUCKET/$MODELOPS_PREFIX/predictions/forecast_summary_by_shop_segment.parquet
SUBMISSION=s3://$MODEL_BUCKET/$MODELOPS_PREFIX/predictions/submission.csv

# Evaluation consumed by Streamlit
EVALUATION_DETAIL=s3://$MODEL_BUCKET/$MODELOPS_PREFIX/evaluation/evaluation_detail.parquet
EVALUATION_BY_SEGMENT=s3://$MODEL_BUCKET/$MODELOPS_PREFIX/evaluation/evaluation_by_segment.parquet
EVALUATION_BY_ITEM=s3://$MODEL_BUCKET/$MODELOPS_PREFIX/evaluation/evaluation_by_item.parquet
MODEL_METRICS=s3://$MODEL_BUCKET/$MODELOPS_PREFIX/evaluation/model_metrics.json

# Optional registry
MODEL_RUNS=s3://$MODEL_BUCKET/modelops/registry/model_runs.csv
CHAMPION=s3://$MODEL_BUCKET/modelops/registry/champion.json

# CFO exports written by app
APP_BATCH_EXPORTS=s3://$MODEL_BUCKET/$APP_EXPORT_PREFIX/
EOF
