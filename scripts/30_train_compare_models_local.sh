#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-modelops_outputs}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-}"
MIN_IMPROVEMENT="${MIN_IMPROVEMENT:-0.00}"

CMD=(uv run python -m src.modelops_v3.model_compare --train-path data/prep/train.parquet --valid-path data/prep/valid.parquet --test-features-path data/prep/test_features.parquet --test-pairs-path data/prep/test_pairs.parquet --model-path artifacts/model.joblib --output-root "$OUTPUT_ROOT" --min-improvement-over-incumbent "$MIN_IMPROVEMENT")
if [[ -n "$MAX_TRAIN_ROWS" ]]; then
  CMD+=(--max-train-rows "$MAX_TRAIN_ROWS")
fi
PYTHONPATH=. "${CMD[@]}"
