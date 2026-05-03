"""Create initial ModelOps registry files in S3.

Creates:
- modelops/registry/champion.json
- modelops/registry/model_runs.csv
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from io import StringIO

import boto3
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--run-id", default="initial-manual-run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    now = datetime.now(timezone.utc).isoformat()

    champion = {
        "model_run_id": args.run_id,
        "created_at": now,
        "status": "promoted",
        "is_champion": True,
        "source_prefix": f"s3://{args.bucket}/modelops/latest",
        "model_uri": f"s3://{args.bucket}/modelops/latest/model/model_bundle.joblib",
        "forecast_uri": f"s3://{args.bucket}/modelops/latest/predictions/forecast_detail.parquet",
        "metrics_uri": f"s3://{args.bucket}/modelops/latest/evaluation/model_metrics.json",
        "promotion_reason": "initial_registry_created_from_existing_successful_modelops_outputs",
    }

    s3 = boto3.client("s3", region_name=args.region)
    s3.put_object(
        Bucket=args.bucket,
        Key="modelops/registry/champion.json",
        Body=json.dumps(champion, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

    buffer = StringIO()
    pd.DataFrame([champion]).to_csv(buffer, index=False)
    s3.put_object(
        Bucket=args.bucket,
        Key="modelops/registry/model_runs.csv",
        Body=buffer.getvalue().encode("utf-8"),
        ContentType="text/csv",
    )

    print(f"Created s3://{args.bucket}/modelops/registry/champion.json")
    print(f"Created s3://{args.bucket}/modelops/registry/model_runs.csv")


if __name__ == "__main__":
    main()
