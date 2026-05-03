#!/usr/bin/env bash
set -euo pipefail

: "${AWS_REGION:=us-east-1}"
: "${VPC_ID:?VPC_ID is required. Run source config/generated.env.}"
: "${SUBNET_IDS:?SUBNET_IDS is required. Run source config/generated.env.}"
: "${VPC_CIDR:?VPC_CIDR is required. Run source config/generated.env.}"
: "${DEVELOPER_IP_CIDR:=0.0.0.0/0}"
: "${RDS_STACK_NAME:=pfs-mvp-rds}"
: "${RDS_SECRET_NAME:=itam/rds/pfs-mvp/credentials}"
: "${DB_PASSWORD:=CambiaEstePassword123!}"

aws cloudformation deploy \
  --template-file infra/01-rds-postgres-app.yaml \
  --stack-name "$RDS_STACK_NAME" \
  --capabilities CAPABILITY_NAMED_IAM \
  --region "$AWS_REGION" \
  --parameter-overrides \
    VpcId="$VPC_ID" \
    SubnetIds="$SUBNET_IDS" \
    VpcCidrBlock="$VPC_CIDR" \
    DeveloperIpCidr="$DEVELOPER_IP_CIDR" \
    DBName=pfs_mvp \
    DBUsername=itam \
    DBPassword="$DB_PASSWORD" \
    SecretName="$RDS_SECRET_NAME"

aws cloudformation describe-stacks \
  --stack-name "$RDS_STACK_NAME" \
  --query "Stacks[0].Outputs" \
  --output table \
  --region "$AWS_REGION"
