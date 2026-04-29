"""Validate that active ModelOps S3 outputs exist and have expected schemas."""

from __future__ import annotations

import argparse
import json
import os
from io import BytesIO
from typing import Any

import boto3
import pandas as pd

EXPECTED_OBJECTS = {
    "forecast_detail": "predictions/forecast_detail.parquet",
    "forecast_summary_by_category": "predictions/forecast_summary_by_category.parquet",
    "forecast_summary_by_shop_segment": "predictions/forecast_summary_by_shop_segment.parquet",
    "submission": "predictions/submission.csv",
    "evaluation_detail": "evaluation/evaluation_detail.parquet",
    "evaluation_by_segment": "evaluation/evaluation_by_segment.parquet",
    "evaluation_by_item": "evaluation/evaluation_by_item.parquet",
    "model_metrics": "evaluation/model_metrics.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=os.getenv("MODEL_BUCKET"), required=False)
    parser.add_argument("--prefix", default=os.getenv("MODELOPS_PREFIX", "modelops/latest"))
    parser.add_argument("--json-output", default=None)
    return parser.parse_args()


def object_exists(s3_client, bucket: str, key: str) -> bool:
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def read_preview(s3_client, bucket: str, key: str) -> dict[str, Any]:
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    if key.endswith(".parquet"):
        df = pd.read_parquet(BytesIO(body))
        return {"rows": len(df), "columns": list(df.columns)[:25]}
    if key.endswith(".csv"):
        df = pd.read_csv(BytesIO(body), nrows=5)
        return {"rows_preview": len(df), "columns": list(df.columns)}
    if key.endswith(".json"):
        payload = json.loads(body.decode("utf-8"))
        return {"keys": list(payload.keys())[:25]}
    return {"bytes": len(body)}


def main() -> None:
    args = parse_args()
    if not args.bucket:
        raise SystemExit("MODEL_BUCKET or --bucket is required")

    prefix = args.prefix.strip("/")
    s3_client = boto3.client("s3")

    report: dict[str, Any] = {
        "bucket": args.bucket,
        "prefix": prefix,
        "objects": {},
        "all_required_exist": True,
    }

    for name, rel_key in EXPECTED_OBJECTS.items():
        key = f"{prefix}/{rel_key}"
        exists = object_exists(s3_client, args.bucket, key)
        item: dict[str, Any] = {"key": key, "exists": exists}
        if exists:
            try:
                item["preview"] = read_preview(s3_client, args.bucket, key)
            except Exception as exc:  # keep validation informative
                item["preview_error"] = str(exc)
        else:
            report["all_required_exist"] = False
        report["objects"][name] = item

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False)

    if not report["all_required_exist"]:
        raise SystemExit("Some required ModelOps outputs are missing")


if __name__ == "__main__":
    main()
