#!/usr/bin/env bash
set -euo pipefail

: "${AWS_REGION:=us-east-1}"
: "${APP_STACK_NAME:=pfs-mvp-app}"
: "${MODEL_BUCKET:?MODEL_BUCKET is required.}"

ECS_CLUSTER=$(aws cloudformation describe-stacks \
  --stack-name "$APP_STACK_NAME" \
  --query "Stacks[0].Outputs[?OutputKey=='ECSClusterName'].OutputValue" \
  --output text \
  --region "$AWS_REGION")

ECS_SERVICE=$(aws cloudformation describe-stacks \
  --stack-name "$APP_STACK_NAME" \
  --query "Stacks[0].Outputs[?OutputKey=='ECSServiceName'].OutputValue" \
  --output text \
  --region "$AWS_REGION")

LOG_GROUP=$(aws cloudformation describe-stacks \
  --stack-name "$APP_STACK_NAME" \
  --query "Stacks[0].Outputs[?OutputKey=='LogGroupName'].OutputValue" \
  --output text \
  --region "$AWS_REGION")

APP_URL=$(aws cloudformation describe-stacks \
  --stack-name "$APP_STACK_NAME" \
  --query "Stacks[0].Outputs[?OutputKey=='AppURL'].OutputValue" \
  --output text \
  --region "$AWS_REGION")

echo "Cluster: $ECS_CLUSTER"
echo "Service: $ECS_SERVICE"
echo "Log group: $LOG_GROUP"
echo "URL: $APP_URL"

echo
echo "ECS service status:"
aws ecs describe-services \
  --cluster "$ECS_CLUSTER" \
  --services "$ECS_SERVICE" \
  --query "services[0].{status:status,running:runningCount,desired:desiredCount,deployments:deployments[*].rolloutState}" \
  --output table \
  --region "$AWS_REGION"

echo
echo "ModelOps outputs:"
aws s3 ls "s3://$MODEL_BUCKET/modelops/latest/predictions/" --region "$AWS_REGION"
aws s3 ls "s3://$MODEL_BUCKET/modelops/latest/evaluation/" --region "$AWS_REGION"
aws s3 ls "s3://$MODEL_BUCKET/modelops/registry/" --region "$AWS_REGION"

echo
echo "Recent logs:"
aws logs tail "$LOG_GROUP" --since 15m --region "$AWS_REGION" || true
