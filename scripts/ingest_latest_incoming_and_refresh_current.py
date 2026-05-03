"""Merge the latest S3 incoming sales batch into raw/current/sales_train.csv.

This script is designed for CodeBuild/EventBridge automation. It finds the most
recent object under raw/incoming/, appends it to raw/current/sales_train.csv,
archives the previous current file, and writes the updated current file back to S3.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from io import BytesIO

import boto3
import pandas as pd

EXPECTED_COLUMNS = [
    "date",
    "date_block_num",
    "shop_id",
    "item_id",
    "item_price",
    "item_cnt_day",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--incoming-prefix", default="raw/incoming/")
    parser.add_argument("--current-key", default="raw/current/sales_train.csv")
    parser.add_argument("--enabled", default="true")
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def read_csv_from_s3(s3_client, bucket: str, key: str) -> pd.DataFrame:
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(obj["Body"])


def write_csv_to_s3(s3_client, df: pd.DataFrame, bucket: str, key: str) -> None:
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    s3_client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue(), ContentType="text/csv")


def list_latest_incoming(s3_client, bucket: str, prefix: str) -> str | None:
    paginator = s3_client.get_paginator("list_objects_v2")
    objects = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        objects.extend(page.get("Contents", []))
    csv_objects = [obj for obj in objects if obj["Key"].endswith(".csv")]
    if not csv_objects:
        return None
    csv_objects.sort(key=lambda item: item["LastModified"], reverse=True)
    return csv_objects[0]["Key"]


def validate_columns(df: pd.DataFrame, name: str) -> None:
    missing = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def main() -> None:
    args = parse_args()
    if args.enabled.lower() != "true":
        print("Incoming merge disabled; leaving raw/current/sales_train.csv unchanged.")
        return

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    s3_client = boto3.client("s3")
    latest_key = list_latest_incoming(s3_client, args.bucket, args.incoming_prefix)
    if latest_key is None:
        print("No incoming CSV found; leaving raw/current/sales_train.csv unchanged.")
        return

    print(f"Latest incoming key: s3://{args.bucket}/{latest_key}")
    current = read_csv_from_s3(s3_client, args.bucket, args.current_key)
    incoming = read_csv_from_s3(s3_client, args.bucket, latest_key)
    validate_columns(current, "current")
    validate_columns(incoming, "incoming")

    archive_key = f"raw/archive/{run_id}/sales_train_before_merge.csv"
    write_csv_to_s3(s3_client, current, args.bucket, archive_key)

    merged = pd.concat([current[EXPECTED_COLUMNS], incoming[EXPECTED_COLUMNS]], ignore_index=True)
    merged = merged.drop_duplicates()
    write_csv_to_s3(s3_client, merged, args.bucket, args.current_key)

    print(f"Archived previous current: s3://{args.bucket}/{archive_key}")
    print(f"Updated current: s3://{args.bucket}/{args.current_key}")
    print(f"current_rows={len(current):,} incoming_rows={len(incoming):,} merged_rows={len(merged):,}")


if __name__ == "__main__":
    main()
