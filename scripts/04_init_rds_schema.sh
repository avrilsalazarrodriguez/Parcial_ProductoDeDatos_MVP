#!/usr/bin/env bash
set -euo pipefail

: "${AWS_REGION:=us-east-1}"
: "${RDS_STACK_NAME:=pfs-mvp-rds}"
: "${RDS_SECRET_NAME:=itam/rds/pfs-mvp/credentials}"

PYTHONPATH=. uv run python scripts/04_init_rds_schema.py \
  --stack-name "$RDS_STACK_NAME" \
  --secret-name "$RDS_SECRET_NAME" \
  --region "$AWS_REGION" \
  --sql-path sql/01_schema_app.sql
