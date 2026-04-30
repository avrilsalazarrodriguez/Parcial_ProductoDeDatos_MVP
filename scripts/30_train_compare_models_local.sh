#!/usr/bin/env bash
set -euo pipefail

MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-300000}"

PYTHONPATH=. uv run python -m src.modelops_v3.model_compare \
  --data-dir data \
  --artifacts-dir artifacts \
  --output-dir modelops_outputs \
  --max-train-rows "$MAX_TRAIN_ROWS"
