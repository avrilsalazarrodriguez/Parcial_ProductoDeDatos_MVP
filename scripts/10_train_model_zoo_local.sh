#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.

uv run python src/modelops_v2/model_zoo_train.py \
  --prep-dir data/prep \
  --data-dir data \
  --output-dir modelops_outputs \
  --s3-bucket "" \
  --s3-prefix modelops/latest \
  --registry-prefix modelops/registry
