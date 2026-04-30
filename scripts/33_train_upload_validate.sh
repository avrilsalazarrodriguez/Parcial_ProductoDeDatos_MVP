#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_BUCKET:?MODEL_BUCKET is required}"

./scripts/30_train_compare_models_local.sh
PYTHONPATH=. uv run python scripts/31_upload_modelops_outputs_s3.py --bucket "$MODEL_BUCKET"
PYTHONPATH=. uv run python scripts/32_validate_modelops_outputs.py --bucket "$MODEL_BUCKET"
