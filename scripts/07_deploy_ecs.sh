#!/usr/bin/env bash
set -euo pipefail

: "${AWS_REGION:=us-east-1}"
: "${VPC_ID:?VPC_ID is required.}"
: "${SUBNET_IDS:?SUBNET_IDS is required.}"
: "${MODEL_BUCKET:?MODEL_BUCKET is required.}"
: "${ECR_URI:?ECR_URI is required.}"
: "${APP_STACK_NAME:=pfs-mvp-app}"
: "${ECS_SERVICE_NAME:=pfs-mvp}"
: "${RDS_STACK_NAME:=pfs-mvp-rds}"
: "${RDS_SECRET_NAME:=itam/rds/pfs-mvp/credentials}"

aws cloudformation deploy \
  --template-file infra/03-ecs-fargate-streamlit-modelops.yaml \
  --stack-name "$APP_STACK_NAME" \
  --capabilities CAPABILITY_NAMED_IAM \
  --region "$AWS_REGION" \
  --parameter-overrides \
    VpcId="$VPC_ID" \
    SubnetIds="$SUBNET_IDS" \
    ImageUri="$ECR_URI:latest" \
    ServiceName="$ECS_SERVICE_NAME" \
    DesiredCount=1 \
    Cpu=1024 \
    Memory=2048 \
    LogRetentionDays=7 \
    ModelBucket="$MODEL_BUCKET" \
    ModelOpsPrefix=modelops/latest \
    ModelOpsRegistryPrefix=modelops/registry \
    AppExportPrefix=app/batch_exports \
    RdsStackName="$RDS_STACK_NAME" \
    RdsSecretName="$RDS_SECRET_NAME"

aws cloudformation describe-stacks \
  --stack-name "$APP_STACK_NAME" \
  --query "Stacks[0].Outputs" \
  --output table \
  --region "$AWS_REGION"

APP_URL=$(aws cloudformation describe-stacks \
  --stack-name "$APP_STACK_NAME" \
  --query "Stacks[0].Outputs[?OutputKey=='AppURL'].OutputValue" \
  --output text \
  --region "$AWS_REGION")

echo "export APP_URL=$APP_URL" | tee -a config/generated.env
echo "Open: $APP_URL"
