#!/usr/bin/env bash
set -euo pipefail

: "${ECS_CLUSTER:?ECS_CLUSTER is required}"
: "${ECS_SERVICE:?ECS_SERVICE is required}"

TASK_DEF_ARN=$(aws ecs describe-services \
  --cluster "$ECS_CLUSTER" \
  --services "$ECS_SERVICE" \
  --query "services[0].taskDefinition" \
  --output text)

TASK_ROLE_ARN=$(aws ecs describe-task-definition \
  --task-definition "$TASK_DEF_ARN" \
  --query "taskDefinition.taskRoleArn" \
  --output text)

EXECUTION_ROLE_ARN=$(aws ecs describe-task-definition \
  --task-definition "$TASK_DEF_ARN" \
  --query "taskDefinition.executionRoleArn" \
  --output text)

cat <<EOF
TASK_DEFINITION_ARN=$TASK_DEF_ARN
TASK_ROLE_ARN=$TASK_ROLE_ARN
EXECUTION_ROLE_ARN=$EXECUTION_ROLE_ARN

# If you need only the role name:
ECS_TASK_ROLE_NAME=$(basename "$TASK_ROLE_ARN")
EOF
