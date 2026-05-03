#!/usr/bin/env bash
set -euo pipefail

: "${AWS_REGION:=us-east-1}"
: "${APP_STACK_NAME:=pfs-mvp-app}"
: "${ECR_STACK_NAME:=pfs-mvp-ecr}"
: "${RDS_STACK_NAME:=pfs-mvp-rds}"

cat <<EOF
This will delete CloudFormation stacks:
- $APP_STACK_NAME
- $ECR_STACK_NAME
- $RDS_STACK_NAME

S3 ModelOps bucket is NOT deleted by this script.
EOF

read -r -p "Type DELETE to continue: " confirm
if [[ "$confirm" != "DELETE" ]]; then
  echo "Cancelled."
  exit 0
fi

aws cloudformation delete-stack --stack-name "$APP_STACK_NAME" --region "$AWS_REGION" || true
aws cloudformation wait stack-delete-complete --stack-name "$APP_STACK_NAME" --region "$AWS_REGION" || true
aws cloudformation delete-stack --stack-name "$ECR_STACK_NAME" --region "$AWS_REGION" || true
aws cloudformation wait stack-delete-complete --stack-name "$ECR_STACK_NAME" --region "$AWS_REGION" || true
aws cloudformation delete-stack --stack-name "$RDS_STACK_NAME" --region "$AWS_REGION" || true
aws cloudformation wait stack-delete-complete --stack-name "$RDS_STACK_NAME" --region "$AWS_REGION" || true

echo "Deletion complete."
