#!/usr/bin/env bash
set -euo pipefail

: "${AWS_REGION:=us-east-1}"
: "${ECR_STACK_NAME:=pfs-mvp-ecr}"
: "${ECR_REPO_NAME:=pfs-mvp-app}"

aws cloudformation deploy \
  --template-file infra/02-ecr-repository.yaml \
  --stack-name "$ECR_STACK_NAME" \
  --region "$AWS_REGION" \
  --parameter-overrides RepositoryName="$ECR_REPO_NAME"

export ECR_URI=$(aws cloudformation describe-stacks \
  --stack-name "$ECR_STACK_NAME" \
  --query "Stacks[0].Outputs[?OutputKey=='RepositoryUri'].OutputValue" \
  --output text \
  --region "$AWS_REGION")

echo "export ECR_URI=$ECR_URI" | tee -a config/generated.env
echo "ECR_URI=$ECR_URI"
