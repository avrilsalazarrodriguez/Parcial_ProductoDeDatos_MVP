#!/usr/bin/env bash
set -euo pipefail
: "${AWS_REGION:=us-east-1}"

PROJECT_NAME=$(aws cloudformation describe-stacks \
  --stack-name pfs-modelops-automation-complete \
  --query "Stacks[0].Outputs[?OutputKey=='CodeBuildProjectName'].OutputValue" \
  --output text \
  --region "$AWS_REGION")

aws codebuild start-build \
  --project-name "$PROJECT_NAME" \
  --region "$AWS_REGION"

echo "Started CodeBuild project: $PROJECT_NAME"
