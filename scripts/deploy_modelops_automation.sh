#!/usr/bin/env bash
set -euo pipefail
: "${AWS_REGION:=us-east-1}"
: "${MODEL_BUCKET:?MODEL_BUCKET is required}"
: "${SAGEMAKER_ROLE_ARN:?SAGEMAKER_ROLE_ARN is required}"
: "${GLUE_CRAWLER_NAME:?GLUE_CRAWLER_NAME is required}"
: "${INSTANCE_TYPE:=ml.m5.xlarge}"

aws cloudformation deploy \
  --template-file infra/modelops-automation-codebuild-eventbridge.yaml \
  --stack-name pfs-modelops-automation-complete \
  --capabilities CAPABILITY_NAMED_IAM \
  --region "$AWS_REGION" \
  --parameter-overrides \
    ModelBucket="$MODEL_BUCKET" \
    SageMakerRoleArn="$SAGEMAKER_ROLE_ARN" \
    GlueCrawlerName="$GLUE_CRAWLER_NAME" \
    InstanceType="$INSTANCE_TYPE" \
    AutoMergeIncoming=true \
    MinImprovement=0.0

aws cloudformation describe-stacks \
  --stack-name pfs-modelops-automation-complete \
  --query "Stacks[0].Outputs" \
  --output table \
  --region "$AWS_REGION"
