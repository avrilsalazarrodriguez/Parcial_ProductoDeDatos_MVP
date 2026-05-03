#!/usr/bin/env bash
set -euo pipefail

export AWS_REGION="${AWS_REGION:-us-east-1}"
mkdir -p config

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region "$AWS_REGION")
VPC_ID=$(aws ec2 describe-vpcs \
  --filters "Name=isDefault,Values=true" \
  --query "Vpcs[0].VpcId" \
  --output text \
  --region "$AWS_REGION")

if [[ -z "$VPC_ID" || "$VPC_ID" == "None" ]]; then
  echo "No default VPC found in $AWS_REGION. Create/select a VPC and subnets manually." >&2
  exit 1
fi

VPC_CIDR=$(aws ec2 describe-vpcs \
  --vpc-ids "$VPC_ID" \
  --query "Vpcs[0].CidrBlock" \
  --output text \
  --region "$AWS_REGION")

SUBNET_IDS=$(aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=$VPC_ID" \
  --query "Subnets[?MapPublicIpOnLaunch==\`true\`].SubnetId" \
  --output text \
  --region "$AWS_REGION" | tr '\t' ',')

if [[ -z "$SUBNET_IDS" ]]; then
  SUBNET_IDS=$(aws ec2 describe-subnets \
    --filters "Name=vpc-id,Values=$VPC_ID" \
    --query "Subnets[].SubnetId" \
    --output text \
    --region "$AWS_REGION" | tr '\t' ',')
fi

DEVELOPER_IP=$(curl -s https://checkip.amazonaws.com || true)
if [[ -n "$DEVELOPER_IP" ]]; then
  DEVELOPER_IP_CIDR="${DEVELOPER_IP}/32"
else
  DEVELOPER_IP_CIDR="0.0.0.0/0"
fi

cat > config/generated.env <<EOF
export AWS_REGION=$AWS_REGION
export ACCOUNT_ID=$ACCOUNT_ID
export VPC_ID=$VPC_ID
export VPC_CIDR=$VPC_CIDR
export SUBNET_IDS=$SUBNET_IDS
export DEVELOPER_IP_CIDR=$DEVELOPER_IP_CIDR
export MODEL_BUCKET=${MODEL_BUCKET:-pfs-modelops-${ACCOUNT_ID}-${AWS_REGION}}
export RDS_STACK_NAME=${RDS_STACK_NAME:-pfs-mvp-rds}
export RDS_SECRET_NAME=${RDS_SECRET_NAME:-itam/rds/pfs-mvp/credentials}
export ECR_STACK_NAME=${ECR_STACK_NAME:-pfs-mvp-ecr}
export ECR_REPO_NAME=${ECR_REPO_NAME:-pfs-mvp-app}
export APP_STACK_NAME=${APP_STACK_NAME:-pfs-mvp-app}
export ECS_SERVICE_NAME=${ECS_SERVICE_NAME:-pfs-mvp}
EOF

cat config/generated.env

echo
echo "Run: source config/generated.env"
