"""Prepare a new ModelOps run by uploading CSV inputs to S3.

Manual ingestion path:
    local CSV files -> s3://<bucket>/raw/runs/<run_id>/
                    -> s3://<bucket>/raw/current/  (optional)

This script does not train a model. It only stages data so the existing
SageMaker Processing flow can consume it.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import boto3


INPUTS = {
    "sales_train.csv": "sales_path",
    "test.csv": "test_path",
    "items_en.csv": "items_path",
    "item_categories_en.csv": "categories_path",
    "shops_en.csv": "shops_path",
    "sample_submission.csv": "sample_submission_path",
}


def make_run_id() -> str:
    """Return a UTC timestamp that is safe for S3 prefixes."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def upload_file(s3_client, bucket: str, local_path: Path, key: str) -> None:
    """Upload a local file to S3 and fail fast if the file is missing."""
    if not local_path.exists():
        raise FileNotFoundError(f"Missing local file: {local_path}")
    print(f"Uploading {local_path} -> s3://{bucket}/{key}")
    s3_client.upload_file(str(local_path), bucket, key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True, help="ModelOps S3 bucket name")
    parser.add_argument("--run-id", default=None, help="Run id. Defaults to UTC timestamp")

    parser.add_argument("--sales-path", required=True)
    parser.add_argument("--test-path", required=True)
    parser.add_argument("--items-path", required=True)
    parser.add_argument("--categories-path", required=True)
    parser.add_argument("--shops-path", required=True)
    parser.add_argument("--sample-submission-path", required=True)

    parser.add_argument(
        "--publish-current",
        action="store_true",
        help="Also publish files to raw/current/ for stable scheduled runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id or make_run_id()
    s3_client = boto3.client("s3")

    for filename, attr in INPUTS.items():
        local_path = Path(getattr(args, attr))
        upload_file(
            s3_client=s3_client,
            bucket=args.bucket,
            local_path=local_path,
            key=f"raw/runs/{run_id}/{filename}",
        )

        if args.publish_current:
            upload_file(
                s3_client=s3_client,
                bucket=args.bucket,
                local_path=local_path,
                key=f"raw/current/{filename}",
            )

    print("Done.")
    print(f"RUN_ID={run_id}")
    print(f"RAW_RUN_PREFIX=s3://{args.bucket}/raw/runs/{run_id}/")
    if args.publish_current:
        print(f"RAW_CURRENT_PREFIX=s3://{args.bucket}/raw/current/")


if __name__ == "__main__":
    main()
