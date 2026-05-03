"""Initialize a simple S3 model registry from existing modelops/latest outputs.

This script is useful when the first successful ModelOps run was executed before
model registry/champion-challenger logic existed. It creates:

- modelops/registry/champion.json
- modelops/registry/model_runs.csv

The app can then show a model_id and the current champion without requiring RDS.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from io import BytesIO, StringIO
from typing import Any

import boto3
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--run-id", default="initial-manual-run")
    parser.add_argument("--prefix", default="modelops/latest")
    return parser.parse_args()


def s3_key_exists(s3_client: Any, bucket: str, key: str) -> bool:
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def read_segment_metrics(s3_client: Any, bucket: str, prefix: str) -> dict[str, Any]:
    key = f"{prefix.strip('/')}/evaluation/evaluation_by_segment.parquet"
    if not s3_key_exists(s3_client, bucket, key):
        return {"weighted_mae": None, "weighted_naive_mae": None, "n_segments": None}

    obj = s3_client.get_object(Bucket=bucket, Key=key)
    df = pd.read_parquet(BytesIO(obj["Body"].read()))
    if df.empty:
        return {"weighted_mae": None, "weighted_naive_mae": None, "n_segments": 0}

    weight_col = "n" if "n" in df.columns else None
    if "mae" in df.columns:
        if weight_col:
            weighted_mae = float((df["mae"] * df[weight_col]).sum() / df[weight_col].sum())
        else:
            weighted_mae = float(df["mae"].mean())
    else:
        weighted_mae = None

    if "naive_mae" in df.columns:
        if weight_col:
            weighted_naive_mae = float(
                (df["naive_mae"] * df[weight_col]).sum() / df[weight_col].sum()
            )
        else:
            weighted_naive_mae = float(df["naive_mae"].mean())
    else:
        weighted_naive_mae = None

    return {
        "weighted_mae": weighted_mae,
        "weighted_naive_mae": weighted_naive_mae,
        "n_segments": int(len(df)),
    }


def main() -> None:
    args = parse_args()
    bucket = args.bucket
    prefix = args.prefix.strip("/")
    run_id = args.run_id
    now = datetime.now(timezone.utc).isoformat()
    s3_client = boto3.client("s3")

    metrics = read_segment_metrics(s3_client, bucket, prefix)

    champion = {
        "model_run_id": run_id,
        "created_at": now,
        "status": "promoted",
        "is_champion": True,
        "source_prefix": f"s3://{bucket}/{prefix}",
        "model_uri": f"s3://{bucket}/{prefix}/model/model_bundle.joblib",
        "forecast_uri": f"s3://{bucket}/{prefix}/predictions/forecast_detail.parquet",
        "metrics_uri": f"s3://{bucket}/{prefix}/evaluation/model_metrics.json",
        "weighted_mae": metrics["weighted_mae"],
        "weighted_naive_mae": metrics["weighted_naive_mae"],
        "n_segments": metrics["n_segments"],
        "promotion_reason": "initialized_from_existing_modelops_latest_outputs",
    }

    s3_client.put_object(
        Bucket=bucket,
        Key="modelops/registry/champion.json",
        Body=json.dumps(champion, indent=2, default=str).encode("utf-8"),
        ContentType="application/json",
    )

    buffer = StringIO()
    pd.DataFrame([champion]).to_csv(buffer, index=False)
    s3_client.put_object(
        Bucket=bucket,
        Key="modelops/registry/model_runs.csv",
        Body=buffer.getvalue().encode("utf-8"),
        ContentType="text/csv",
    )

    print(json.dumps(champion, indent=2, default=str))


if __name__ == "__main__":
    main()
