"""I/O utilities that support local files and S3 URIs."""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

import boto3
import pandas as pd


def is_s3_uri(path: str) -> bool:
    """Return True if a path points to S3."""
    return str(path).startswith("s3://")


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split an S3 URI into bucket and key."""
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Not an S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def read_csv(path: str, **kwargs) -> pd.DataFrame:
    """Read CSV from local path or S3."""
    return pd.read_csv(path, **kwargs)


def read_parquet(path: str, **kwargs) -> pd.DataFrame:
    """Read Parquet from local path or S3."""
    return pd.read_parquet(path, **kwargs)


def write_parquet(df: pd.DataFrame, path: str) -> None:
    """Write Parquet to local path or S3."""
    if not is_s3_uri(path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def write_csv(df: pd.DataFrame, path: str) -> None:
    """Write CSV to local path or S3."""
    if not is_s3_uri(path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_json(payload: dict, path: str) -> None:
    """Write JSON to local path or S3."""
    if is_s3_uri(path):
        bucket, key = parse_s3_uri(path)
        boto3.client("s3").put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8"),
            ContentType="application/json",
        )
    else:
        local_path = Path(path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def copy_local_to_s3(local_path: str, s3_uri: str) -> None:
    """Upload a local file to S3."""
    bucket, key = parse_s3_uri(s3_uri)
    boto3.client("s3").upload_file(local_path, bucket, key)
