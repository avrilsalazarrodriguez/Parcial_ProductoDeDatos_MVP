"""Helpers for saving CFO batch exports to Amazon S3."""

from __future__ import annotations

import os
from io import StringIO
from time import gmtime, strftime

import boto3
import pandas as pd


def get_export_bucket() -> str:
    """Return the bucket used by Streamlit to persist generated exports."""
    bucket = os.getenv("APP_EXPORT_BUCKET") or os.getenv("MODEL_BUCKET")
    if bucket:
        return bucket

    account_id = boto3.client("sts").get_caller_identity()["Account"]
    region = boto3.session.Session().region_name or os.getenv("AWS_REGION", "us-east-1")
    return f"sagemaker-{region}-{account_id}"


def get_export_prefix() -> str:
    """Return the S3 prefix used for CFO exports."""
    return os.getenv("APP_EXPORT_PREFIX", "app/batch_exports").strip("/")


def upload_batch_dataframe_to_s3(
    df: pd.DataFrame,
    scope: str,
    shop_id: int | None = None,
) -> str:
    """Upload a batch forecast dataframe as CSV and return its S3 URI."""
    bucket = get_export_bucket()
    prefix = get_export_prefix()
    timestamp = strftime("%Y%m%d-%H%M%S", gmtime())

    safe_scope = scope.lower().replace(" ", "_").replace("/", "_")
    shop_part = f"shop_{shop_id}" if shop_id is not None else "all"
    key = f"{prefix}/{safe_scope}/{shop_part}/predictions_{timestamp}.csv"

    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)

    boto3.client("s3").put_object(
        Bucket=bucket,
        Key=key,
        Body=csv_buffer.getvalue().encode("utf-8"),
        ContentType="text/csv",
    )
    return f"s3://{bucket}/{key}"
