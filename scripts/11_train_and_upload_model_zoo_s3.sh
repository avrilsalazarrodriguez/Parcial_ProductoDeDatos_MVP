#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_BUCKET:?Falta export MODEL_BUCKET=<bucket>}"
export PYTHONPATH=.
export AWS_REGION="${AWS_REGION:-us-east-1}"
export MODELOPS_PREFIX="${MODELOPS_PREFIX:-modelops/latest}"
export MODELOPS_REGISTRY_PREFIX="${MODELOPS_REGISTRY_PREFIX:-modelops/registry}"

uv run python src/modelops_v2/model_zoo_train.py \
  --prep-dir data/prep \
  --data-dir data \
  --output-dir modelops_outputs \
  --s3-bucket "$MODEL_BUCKET" \
  --s3-prefix "$MODELOPS_PREFIX" \
  --registry-prefix "$MODELOPS_REGISTRY_PREFIX"
