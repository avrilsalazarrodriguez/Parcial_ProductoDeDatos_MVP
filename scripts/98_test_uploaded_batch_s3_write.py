#!/usr/bin/env python3
from __future__ import annotations

import os
from datetime import datetime, timezone

import boto3

bucket = os.getenv("MODEL_BUCKET") or os.getenv("MODELOPS_BUCKET")
region = os.getenv("AWS_REGION", "us-east-1")
if not bucket:
    raise SystemExit("Falta MODEL_BUCKET/MODELOPS_BUCKET")

s3 = boto3.client("s3", region_name=region)
key = "app/batch_uploads/_diagnostics/put_test_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + ".txt"
body = b"uploaded batch s3 write test\n"
print("put:", f"s3://{bucket}/{key}")
s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="text/plain")
obj = s3.get_object(Bucket=bucket, Key=key)
print("read bytes:", len(obj["Body"].read()))
print("OK")
