"""Register a candidate model run and promote it only if it improves champion.

The registry lives in S3 and is designed to be consumed by Streamlit:

- modelops/registry/champion.json
- modelops/registry/model_runs.csv
- modelops/registry/segment_metrics/<run_id>.parquet
- modelops/registry/item_metrics/<run_id>.parquet

Promotion copies the candidate run outputs into modelops/latest/*.
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
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-prefix", required=True, help="Example: modelops/runs/<run_id>")
    parser.add_argument("--min-improvement", type=float, default=0.0)
    parser.add_argument("--force-promote", action="store_true")
    return parser.parse_args()


def read_json_or_none(s3_client: Any, bucket: str, key: str) -> dict[str, Any] | None:
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except Exception:
        return None


def write_json(s3_client: Any, bucket: str, key: str, payload: dict[str, Any]) -> None:
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, indent=2, default=str).encode("utf-8"),
        ContentType="application/json",
    )


def read_parquet(s3_client: Any, bucket: str, key: str) -> pd.DataFrame:
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(BytesIO(obj["Body"].read()))


def read_csv_or_empty(s3_client: Any, bucket: str, key: str) -> pd.DataFrame:
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(obj["Body"])
    except Exception:
        return pd.DataFrame()


def write_csv(s3_client: Any, bucket: str, key: str, df: pd.DataFrame) -> None:
    buffer = StringIO()
    df.to_csv(buffer, index=False)
    s3_client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue().encode("utf-8"))


def write_parquet(s3_client: Any, bucket: str, key: str, df: pd.DataFrame) -> None:
    buffer = BytesIO()
    df.to_parquet(buffer, index=False)
    s3_client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())


def copy_prefix(s3_client: Any, bucket: str, source_prefix: str, target_prefix: str) -> None:
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=source_prefix):
        for obj in page.get("Contents", []):
            source_key = obj["Key"]
            target_key = source_key.replace(source_prefix, target_prefix, 1)
            s3_client.copy_object(
                Bucket=bucket,
                CopySource={"Bucket": bucket, "Key": source_key},
                Key=target_key,
            )
            print(f"copied s3://{bucket}/{source_key} -> s3://{bucket}/{target_key}")


def weighted_mean(df: pd.DataFrame, value_col: str, weight_col: str = "n") -> float | None:
    if value_col not in df.columns or df.empty:
        return None
    if weight_col in df.columns and df[weight_col].sum() > 0:
        return float((df[value_col] * df[weight_col]).sum() / df[weight_col].sum())
    return float(df[value_col].mean())


def compute_candidate_metrics(segment_df: pd.DataFrame, item_df: pd.DataFrame) -> dict[str, Any]:
    weighted_mae = weighted_mean(segment_df, "mae")
    weighted_naive_mae = weighted_mean(segment_df, "naive_mae")
    item_weighted_mae = weighted_mean(item_df, "mae")
    beats_naive_rate = None
    if "beats_naive_rate" in segment_df.columns and not segment_df.empty:
        beats_naive_rate = float(segment_df["beats_naive_rate"].mean())
    elif {"mae", "naive_mae"}.issubset(segment_df.columns) and not segment_df.empty:
        beats_naive_rate = float((segment_df["mae"] <= segment_df["naive_mae"]).mean())

    return {
        "weighted_mae": weighted_mae,
        "weighted_naive_mae": weighted_naive_mae,
        "item_weighted_mae": item_weighted_mae,
        "beats_naive_rate": beats_naive_rate,
        "n_segments": int(len(segment_df)),
        "n_items_evaluated": int(len(item_df)),
    }


def decide_promotion(
    candidate: dict[str, Any],
    champion: dict[str, Any] | None,
    min_improvement: float,
    force: bool,
) -> tuple[bool, str]:
    if force:
        return True, "forced_promotion"
    if champion is None:
        return True, "no_existing_champion"

    candidate_mae = candidate.get("weighted_mae")
    champion_mae = champion.get("weighted_mae")
    if candidate_mae is None:
        return False, "candidate_missing_weighted_mae"
    if champion_mae is None:
        return True, "champion_missing_weighted_mae"

    threshold = float(champion_mae) * (1.0 - min_improvement)
    if float(candidate_mae) <= threshold:
        return True, f"candidate_mae_{candidate_mae:.6f}_beats_threshold_{threshold:.6f}"
    return False, f"candidate_mae_{candidate_mae:.6f}_does_not_beat_champion_{champion_mae:.6f}"


def main() -> None:
    args = parse_args()
    bucket = args.bucket
    run_id = args.run_id
    base_prefix = args.base_prefix.strip("/")
    s3_client = boto3.client("s3")

    segment_key = f"{base_prefix}/evaluation/evaluation_by_segment.parquet"
    item_key = f"{base_prefix}/evaluation/evaluation_by_item.parquet"
    segment_df = read_parquet(s3_client, bucket, segment_key)
    item_df = read_parquet(s3_client, bucket, item_key)
    metrics = compute_candidate_metrics(segment_df, item_df)

    champion_key = "modelops/registry/champion.json"
    runs_key = "modelops/registry/model_runs.csv"
    champion = read_json_or_none(s3_client, bucket, champion_key)

    candidate = {
        "model_run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "candidate",
        "is_champion": False,
        "source_prefix": f"s3://{bucket}/{base_prefix}",
        "model_uri": f"s3://{bucket}/{base_prefix}/model/model_bundle.joblib",
        "forecast_uri": f"s3://{bucket}/{base_prefix}/predictions/forecast_detail.parquet",
        "metrics_uri": f"s3://{bucket}/{base_prefix}/evaluation/model_metrics.json",
        **metrics,
    }

    promote, reason = decide_promotion(candidate, champion, args.min_improvement, args.force_promote)
    candidate["promotion_reason"] = reason
    candidate["status"] = "promoted" if promote else "rejected"
    candidate["is_champion"] = bool(promote)

    if promote:
        copy_prefix(s3_client, bucket, f"{base_prefix}/predictions/", "modelops/latest/predictions/")
        copy_prefix(s3_client, bucket, f"{base_prefix}/evaluation/", "modelops/latest/evaluation/")
        copy_prefix(s3_client, bucket, f"{base_prefix}/model/", "modelops/latest/model/")
        write_json(s3_client, bucket, champion_key, candidate)

    runs_df = read_csv_or_empty(s3_client, bucket, runs_key)
    runs_df = pd.concat([runs_df, pd.DataFrame([candidate])], ignore_index=True)
    write_csv(s3_client, bucket, runs_key, runs_df)
    write_parquet(s3_client, bucket, f"modelops/registry/segment_metrics/{run_id}.parquet", segment_df)
    write_parquet(s3_client, bucket, f"modelops/registry/item_metrics/{run_id}.parquet", item_df)

    print(json.dumps(candidate, indent=2, default=str))


if __name__ == "__main__":
    main()
