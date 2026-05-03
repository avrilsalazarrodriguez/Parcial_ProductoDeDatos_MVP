#!/usr/bin/env bash
set -euo pipefail

: "${AWS_REGION:=us-east-1}"
: "${APP_STACK_NAME:=pfs-mvp-app}"
: "${RDS_STACK_NAME:=pfs-mvp-rds}"

ECS_CLUSTER=$(aws cloudformation describe-stacks \
  --stack-name "$APP_STACK_NAME" \
  --query "Stacks[0].Outputs[?OutputKey=='ECSClusterName'].OutputValue" \
  --output text \
  --region "$AWS_REGION" 2>/dev/null || true)

ECS_SERVICE=$(aws cloudformation describe-stacks \
  --stack-name "$APP_STACK_NAME" \
  --query "Stacks[0].Outputs[?OutputKey=='ECSServiceName'].OutputValue" \
  --output text \
  --region "$AWS_REGION" 2>/dev/null || true)

if [[ -n "$ECS_CLUSTER" && "$ECS_CLUSTER" != "None" && -n "$ECS_SERVICE" && "$ECS_SERVICE" != "None" ]]; then
  aws ecs update-service \
    --cluster "$ECS_CLUSTER" \
    --service "$ECS_SERVICE" \
    --desired-count 0 \
    --region "$AWS_REGION"
  echo "ECS service desired-count set to 0."
else
  echo "ECS service not found."
fi

RDS_ID=$(aws cloudformation describe-stacks \
  --stack-name "$RDS_STACK_NAME" \
  --query "Stacks[0].Outputs[?OutputKey=='RdsInstanceId'].OutputValue" \
  --output text \
  --region "$AWS_REGION" 2>/dev/null || true)

if [[ -n "$RDS_ID" && "$RDS_ID" != "None" ]]; then
  STATUS=$(aws rds describe-db-instances \
    --db-instance-identifier "$RDS_ID" \
    --query "DBInstances[0].DBInstanceStatus" \
    --output text \
    --region "$AWS_REGION")
  if [[ "$STATUS" == "available" ]]; then
    aws rds stop-db-instance --db-instance-identifier "$RDS_ID" --region "$AWS_REGION"
    echo "RDS stop requested: $RDS_ID"
  else
    echo "RDS status is $STATUS; not stopping."
  fi
else
  echo "RDS instance not found."
fi
