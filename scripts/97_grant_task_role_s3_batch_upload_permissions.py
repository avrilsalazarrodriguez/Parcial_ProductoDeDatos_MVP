#!/usr/bin/env python3
from __future__ import annotations

"""Grant ECS task role access to uploaded-batch prediction S3 prefixes.

This fixes errors like:

AccessDenied: not authorized to perform s3:PutObject on
arn:aws:s3:::<bucket>/app/batch_uploads/...

It attaches an inline least-scope policy to the ECS task role. It does not change
Streamlit code or redeploy ECS.
"""

import argparse
import json
import os
from typing import Any

import boto3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role-name", default=os.getenv("ECS_TASK_ROLE_NAME", "pfs-mvp-task-role"))
    parser.add_argument("--bucket", default=os.getenv("MODEL_BUCKET") or os.getenv("MODELOPS_BUCKET"))
    parser.add_argument("--region", default=os.getenv("AWS_REGION", "us-east-1"))
    parser.add_argument("--policy-name", default="pfs-mvp-uploaded-batch-s3-access")
    parser.add_argument("--include-batch-exports", action="store_true", help="Also allow app/batch_exports/*")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def role_name_from_input(value: str) -> str:
    # Accept either role name or ARN.
    return value.rsplit("/", 1)[-1]


def build_policy(bucket: str, include_batch_exports: bool = False) -> dict[str, Any]:
    object_resources = [f"arn:aws:s3:::{bucket}/app/batch_uploads/*"]
    prefixes = ["app/batch_uploads/*", "app/batch_uploads/"]
    if include_batch_exports:
        object_resources.append(f"arn:aws:s3:::{bucket}/app/batch_exports/*")
        prefixes.extend(["app/batch_exports/*", "app/batch_exports/"])

    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "ListModelOpsAppUploadPrefixes",
                "Effect": "Allow",
                "Action": ["s3:ListBucket"],
                "Resource": f"arn:aws:s3:::{bucket}",
                "Condition": {"StringLike": {"s3:prefix": prefixes}},
            },
            {
                "Sid": "ReadWriteUploadedBatchPredictions",
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
                "Resource": object_resources,
            },
        ],
    }


def main() -> None:
    args = parse_args()
    if not args.bucket:
        raise SystemExit("Falta --bucket o variable MODEL_BUCKET/MODELOPS_BUCKET")

    role_name = role_name_from_input(args.role_name)
    policy = build_policy(args.bucket, include_batch_exports=args.include_batch_exports)

    print("role:", role_name)
    print("bucket:", args.bucket)
    print("policy_name:", args.policy_name)
    print(json.dumps(policy, indent=2))

    if args.dry_run:
        print("dry-run: no apliqué cambios")
        return

    iam = boto3.client("iam", region_name=args.region)
    iam.put_role_policy(
        RoleName=role_name,
        PolicyName=args.policy_name,
        PolicyDocument=json.dumps(policy),
    )
    print("OK: inline policy attached/updated")
    print("Puede tardar algunos segundos/minutos en reflejarse en la task ECS.")


if __name__ == "__main__":
    main()
