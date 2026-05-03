#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_BUCKET:?MODEL_BUCKET is required}"

SOURCE_PREFIX="${SOURCE_PREFIX:-modelops}"
LATEST_PREFIX="${LATEST_PREFIX:-modelops/latest}"

printf 'Publishing existing ModelOps outputs to latest...\n'
printf 'Bucket: s3://%s\n' "$MODEL_BUCKET"
printf 'Source prefix: %s\n' "$SOURCE_PREFIX"
printf 'Latest prefix: %s\n' "$LATEST_PREFIX"

aws s3 sync "s3://$MODEL_BUCKET/$SOURCE_PREFIX/predictions/" \
  "s3://$MODEL_BUCKET/$LATEST_PREFIX/predictions/"

aws s3 sync "s3://$MODEL_BUCKET/$SOURCE_PREFIX/evaluation/" \
  "s3://$MODEL_BUCKET/$LATEST_PREFIX/evaluation/"

aws s3 sync "s3://$MODEL_BUCKET/$SOURCE_PREFIX/model/" \
  "s3://$MODEL_BUCKET/$LATEST_PREFIX/model/"

printf '\nDone. Validate with:\n'
printf 'aws s3 ls s3://%s/%s/ --recursive\n' "$MODEL_BUCKET" "$LATEST_PREFIX"
