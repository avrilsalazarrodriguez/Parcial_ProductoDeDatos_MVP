#!/usr/bin/env python3
from __future__ import annotations

import os
import pandas as pd
import boto3

bucket = os.getenv("MODEL_BUCKET") or os.getenv("MODELOPS_BUCKET")
region = os.getenv("AWS_REGION", "us-east-1")
if not bucket:
    raise SystemExit("Falta MODEL_BUCKET/MODELOPS_BUCKET")

s3 = boto3.client("s3", region_name=region)
history_key = "app/batch_uploads/history/uploaded_predictions_history.csv"
print("predictions:", f"s3://{bucket}/app/batch_uploads/predictions/")
print("history:", f"s3://{bucket}/{history_key}")

try:
    obj = s3.get_object(Bucket=bucket, Key=history_key)
    history = pd.read_csv(obj["Body"])
    print("\nHistorial:")
    print(history.head(30).to_string(index=False))
except Exception as exc:
    print("\nNo pude leer historial:", exc)

print("\nObjetos:")
resp = s3.list_objects_v2(Bucket=bucket, Prefix="app/batch_uploads/predictions/")
for obj in resp.get("Contents", [])[-50:]:
    print(obj["Key"], obj.get("Size"))
