#!/usr/bin/env bash
set -euo pipefail

: "${AWS_REGION:=us-east-1}"
: "${MODEL_BUCKET:?MODEL_BUCKET is required}"

printf '\n== AWS identity ==\n'
aws sts get-caller-identity --region "$AWS_REGION"

printf '\n== CloudFormation stacks ==\n'
for stack in pfs-modelops pfs-mvp-rds pfs-mvp-ecr pfs-mvp-app; do
  status=$(aws cloudformation describe-stacks \
    --stack-name "$stack" \
    --query "Stacks[0].StackStatus" \
    --output text \
    --region "$AWS_REGION" 2>/dev/null || true)
  printf '%-24s %s\n' "$stack" "${status:-MISSING}"
done

printf '\n== ModelOps S3 latest ==\n'
aws s3 ls "s3://$MODEL_BUCKET/modelops/latest/predictions/" || true
aws s3 ls "s3://$MODEL_BUCKET/modelops/latest/evaluation/" || true
aws s3 ls "s3://$MODEL_BUCKET/modelops/registry/" || true

printf '\n== ECS service ==\n'
aws ecs describe-services \
  --cluster pfs-mvp-cluster \
  --services pfs-mvp \
  --region "$AWS_REGION" \
  --query "services[0].{status:status,running:runningCount,desired:desiredCount,rollout:deployments[0].rolloutState}" \
  --output table || true

printf '\n== ECR images ==\n'
aws ecr describe-images \
  --repository-name pfs-mvp-app \
  --region "$AWS_REGION" \
  --query "imageDetails[].imageTags" \
  --output json || true

printf '\n== RDS instances ==\n'
aws rds describe-db-instances \
  --region "$AWS_REGION" \
  --query "DBInstances[].{DB:DBInstanceIdentifier,Status:DBInstanceStatus,Endpoint:Endpoint.Address}" \
  --output table || true

printf '\n== Done ==\n'
