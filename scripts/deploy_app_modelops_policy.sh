#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_BUCKET:?MODEL_BUCKET is required}"
: "${ECS_TASK_ROLE_NAME:?ECS_TASK_ROLE_NAME is required}"

aws cloudformation deploy \
  --template-file infra/app-task-modelops-policy.yaml \
  --stack-name pfs-app-modelops-s3-policy \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    ModelBucket="$MODEL_BUCKET" \
    EcsTaskRoleName="$ECS_TASK_ROLE_NAME"
