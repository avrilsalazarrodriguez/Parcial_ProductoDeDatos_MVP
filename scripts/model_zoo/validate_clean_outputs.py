"""Validate that latest forecast outputs are friendly and model catalog exists."""

from __future__ import annotations

import argparse
from io import BytesIO

import boto3
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--latest-prefix", default="modelops/latest")
    parser.add_argument("--registry-prefix", default="modelops/registry")
    parser.add_argument("--region", default="us-east-1")
    return parser.parse_args()


def read_parquet(s3, bucket: str, key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(BytesIO(obj["Body"].read()))


def object_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    s3 = boto3.client("s3", region_name=args.region)
    bucket = args.bucket
    latest = args.latest_prefix.strip("/")
    registry = args.registry_prefix.strip("/")
    errors = []

    forecast_key = f"{latest}/predictions/forecast_detail.parquet"
    forecast = read_parquet(s3, bucket, forecast_key)
    print("forecast_detail", forecast.shape)
    required_forecast_cols = {"shop_id", "shop_name", "item_id", "prediction", "item_category_name"}
    missing = required_forecast_cols - set(forecast.columns)
    if missing:
        errors.append(f"forecast_detail missing columns: {sorted(missing)}")

    if "recency" in forecast.columns and "recency_label" not in forecast.columns:
        errors.append("forecast_detail has recency but no recency_label")

    for key in [f"{registry}/champion.json", f"{registry}/model_runs.csv"]:
        if not object_exists(s3, bucket, key):
            errors.append(f"missing s3://{bucket}/{key}")

    if errors:
        print("VALIDATION FAILED")
        for error in errors:
            print("-", error)
        raise SystemExit(1)

    print("VALIDATION OK")
    print(forecast[[c for c in ["shop_id", "shop_name", "item_id", "prediction", "item_category_name", "recency_label"] if c in forecast.columns]].head())


if __name__ == "__main__":
    main()
