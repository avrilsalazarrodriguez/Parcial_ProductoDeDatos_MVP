#!/usr/bin/env bash
set -euo pipefail
: "${MODEL_BUCKET:?MODEL_BUCKET is required}"
: "${AWS_REGION:=us-east-1}"

aws s3api put-bucket-notification-configuration \
  --bucket "$MODEL_BUCKET" \
  --notification-configuration '{"EventBridgeConfiguration": {}}' \
  --region "$AWS_REGION"

echo "Enabled S3 -> EventBridge notifications for bucket: $MODEL_BUCKET"
