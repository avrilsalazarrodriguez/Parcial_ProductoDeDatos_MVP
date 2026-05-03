#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_BUCKET:?MODEL_BUCKET is required. Run source config/generated.env or export MODEL_BUCKET.}"
: "${AWS_REGION:=us-east-1}"

echo "Publishing existing ModelOps outputs to modelops/latest in s3://$MODEL_BUCKET"

aws s3 sync "s3://$MODEL_BUCKET/modelops/predictions/" "s3://$MODEL_BUCKET/modelops/latest/predictions/" --region "$AWS_REGION" || true
aws s3 sync "s3://$MODEL_BUCKET/modelops/evaluation/" "s3://$MODEL_BUCKET/modelops/latest/evaluation/" --region "$AWS_REGION" || true
aws s3 sync "s3://$MODEL_BUCKET/modelops/model/" "s3://$MODEL_BUCKET/modelops/latest/model/" --region "$AWS_REGION" || true

PYTHONPATH=. uv run python "$PWD/scripts/create_modelops_registry.py" --bucket "$MODEL_BUCKET" --region "$AWS_REGION"

echo "Validating latest outputs..."
aws s3 ls "s3://$MODEL_BUCKET/modelops/latest/predictions/" --region "$AWS_REGION"
aws s3 ls "s3://$MODEL_BUCKET/modelops/latest/evaluation/" --region "$AWS_REGION"
aws s3 ls "s3://$MODEL_BUCKET/modelops/registry/" --region "$AWS_REGION"
