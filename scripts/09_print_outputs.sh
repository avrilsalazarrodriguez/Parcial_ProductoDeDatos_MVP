#!/usr/bin/env bash
set -euo pipefail

: "${AWS_REGION:=us-east-1}"
: "${APP_STACK_NAME:=pfs-mvp-app}"
: "${RDS_STACK_NAME:=pfs-mvp-rds}"
: "${ECR_STACK_NAME:=pfs-mvp-ecr}"
: "${MODEL_BUCKET:?MODEL_BUCKET is required.}"

APP_URL=$(aws cloudformation describe-stacks --stack-name "$APP_STACK_NAME" --query "Stacks[0].Outputs[?OutputKey=='AppURL'].OutputValue" --output text --region "$AWS_REGION" 2>/dev/null || true)
ECS_CLUSTER=$(aws cloudformation describe-stacks --stack-name "$APP_STACK_NAME" --query "Stacks[0].Outputs[?OutputKey=='ECSClusterName'].OutputValue" --output text --region "$AWS_REGION" 2>/dev/null || true)
ECS_SERVICE=$(aws cloudformation describe-stacks --stack-name "$APP_STACK_NAME" --query "Stacks[0].Outputs[?OutputKey=='ECSServiceName'].OutputValue" --output text --region "$AWS_REGION" 2>/dev/null || true)
LOG_GROUP=$(aws cloudformation describe-stacks --stack-name "$APP_STACK_NAME" --query "Stacks[0].Outputs[?OutputKey=='LogGroupName'].OutputValue" --output text --region "$AWS_REGION" 2>/dev/null || true)
ECR_URI=$(aws cloudformation describe-stacks --stack-name "$ECR_STACK_NAME" --query "Stacks[0].Outputs[?OutputKey=='RepositoryUri'].OutputValue" --output text --region "$AWS_REGION" 2>/dev/null || true)
RDS_ENDPOINT=$(aws cloudformation describe-stacks --stack-name "$RDS_STACK_NAME" --query "Stacks[0].Outputs[?OutputKey=='RdsEndpoint'].OutputValue" --output text --region "$AWS_REGION" 2>/dev/null || true)

cat <<EOF
# MVP AWS outputs
APP_URL=$APP_URL
ECS_CLUSTER=$ECS_CLUSTER
ECS_SERVICE=$ECS_SERVICE
LOG_GROUP=$LOG_GROUP
ECR_URI=$ECR_URI
RDS_ENDPOINT=$RDS_ENDPOINT
MODEL_BUCKET=s3://$MODEL_BUCKET
FORECAST_DETAIL=s3://$MODEL_BUCKET/modelops/latest/predictions/forecast_detail.parquet
EVALUATION_BY_SEGMENT=s3://$MODEL_BUCKET/modelops/latest/evaluation/evaluation_by_segment.parquet
EVALUATION_BY_ITEM=s3://$MODEL_BUCKET/modelops/latest/evaluation/evaluation_by_item.parquet
MODEL_METRICS=s3://$MODEL_BUCKET/modelops/latest/evaluation/model_metrics.json
APP_BATCH_EXPORTS=s3://$MODEL_BUCKET/app/batch_exports/
EOF
