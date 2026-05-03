"""Upload a new incoming CSV to S3 to trigger or test retraining automation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import boto3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--incoming-path", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--prefix", default="raw/incoming")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    local_path = Path(args.incoming_path)
    if not local_path.exists():
        raise FileNotFoundError(local_path)

    key = f"{args.prefix.strip('/')}/{run_id}/{local_path.name}"
    boto3.client("s3").upload_file(str(local_path), args.bucket, key)
    print(f"uploaded=s3://{args.bucket}/{key}")
    print(f"run_id={run_id}")


if __name__ == "__main__":
    main()
