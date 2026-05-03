#!/usr/bin/env bash
set -euo pipefail

: "${AWS_REGION:?AWS_REGION is required}"
: "${ECR_REPO_NAME:?ECR_REPO_NAME is required}"

ECR_URI=$(aws ecr describe-repositories \
  --repository-names "$ECR_REPO_NAME" \
  --query "repositories[0].repositoryUri" \
  --output text \
  --region "$AWS_REGION")

echo "ECR_REPO_NAME=${ECR_REPO_NAME}"
echo "ECR_URI=${ECR_URI}"

aws ecr get-login-password --region "$AWS_REGION" | \
  docker login --username AWS --password-stdin "$(echo "$ECR_URI" | cut -d/ -f1)"

docker buildx build \
  --platform linux/amd64 \
  --no-cache \
  -t "${ECR_URI}:modelops-s3" \
  -t "${ECR_URI}:latest" \
  --push .

echo "Image pushed:"
echo "${ECR_URI}:modelops-s3"
echo "${ECR_URI}:latest"
