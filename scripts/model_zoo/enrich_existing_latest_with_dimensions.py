"""Enrich existing `modelops/latest` outputs with shop/item/category names.

Use this as the fast fix before retraining the model zoo. It reads the current
outputs and dimensions from S3, fixes missing display values and recency labels,
and writes the cleaned tables back to `modelops/latest`.
"""

from __future__ import annotations

import argparse
from io import BytesIO

import boto3
import pandas as pd

from modelops.data_quality_dimensions import (
    build_item_dimension,
    build_shop_dimension,
    enrich_with_dimensions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--latest-prefix", default="modelops/latest")
    parser.add_argument("--raw-prefix", default="raw/current")
    parser.add_argument("--region", default="us-east-1")
    return parser.parse_args()


def read_csv(s3, bucket: str, key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(obj["Body"])


def read_parquet(s3, bucket: str, key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(BytesIO(obj["Body"].read()))


def write_parquet(s3, bucket: str, key: str, df: pd.DataFrame) -> None:
    buffer = BytesIO()
    df.to_parquet(buffer, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
    print(f"updated s3://{bucket}/{key} rows={len(df):,}")


def main() -> None:
    args = parse_args()
    s3 = boto3.client("s3", region_name=args.region)
    bucket = args.bucket
    latest = args.latest_prefix.strip("/")
    raw = args.raw_prefix.strip("/")

    shops = read_csv(s3, bucket, f"{raw}/shops_en.csv")
    items = read_csv(s3, bucket, f"{raw}/items_en.csv")
    categories = read_csv(s3, bucket, f"{raw}/item_categories_en.csv")
    shop_dim = build_shop_dimension(shops)
    item_dim = build_item_dimension(items, categories)

    keys = [
        f"{latest}/predictions/forecast_detail.parquet",
        f"{latest}/predictions/forecast_summary_by_category.parquet",
        f"{latest}/predictions/forecast_summary_by_shop_segment.parquet",
        f"{latest}/evaluation/evaluation_detail.parquet",
        f"{latest}/evaluation/evaluation_by_segment.parquet",
        f"{latest}/evaluation/evaluation_by_item.parquet",
    ]
    for key in keys:
        try:
            df = read_parquet(s3, bucket, key)
        except Exception as exc:  # noqa: BLE001
            print(f"skip s3://{bucket}/{key}: {exc}")
            continue
        enriched = enrich_with_dimensions(df, shop_dimension=shop_dim, item_dimension=item_dim)
        write_parquet(s3, bucket, key, enriched)


if __name__ == "__main__":
    main()
