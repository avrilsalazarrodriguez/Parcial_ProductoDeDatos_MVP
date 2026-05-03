#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_BUCKET:?MODEL_BUCKET is required}"
: "${AWS_REGION:=us-east-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-modelops_outputs}"

./scripts/30_train_compare_models_local.sh
PYTHONPATH=. uv run python scripts/32_validate_modelops_outputs.py --local-root "$OUTPUT_ROOT"
PYTHONPATH=. uv run python scripts/31_upload_modelops_outputs_s3.py --bucket "$MODEL_BUCKET" --local-root "$OUTPUT_ROOT" --latest-prefix "${MODELOPS_PREFIX:-modelops/latest}" --registry-prefix "${MODELOPS_REGISTRY_PREFIX:-modelops/registry}" --region "$AWS_REGION"
PYTHONPATH=. uv run python scripts/32_validate_modelops_outputs.py --bucket "$MODEL_BUCKET" --latest-prefix "${MODELOPS_PREFIX:-modelops/latest}" --registry-prefix "${MODELOPS_REGISTRY_PREFIX:-modelops/registry}" --region "$AWS_REGION"
