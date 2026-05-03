#!/usr/bin/env bash
set -euo pipefail
: "${MODEL_BUCKET:?MODEL_BUCKET is required}"

ZIP_PATH="modelops_automation_source.zip"
rm -f "$ZIP_PATH"

zip -r "$ZIP_PATH" \
  modelops \
  scripts \
  requirements.txt \
  pyproject.toml \
  uv.lock \
  buildspec-modelops-automation.yml \
  -x "*.DS_Store" ".venv/*" "data/*" "*.csv" "*.parquet" "*.joblib"

aws s3 cp "$ZIP_PATH" "s3://$MODEL_BUCKET/code/modelops_automation_source.zip"
echo "Uploaded s3://$MODEL_BUCKET/code/modelops_automation_source.zip"
