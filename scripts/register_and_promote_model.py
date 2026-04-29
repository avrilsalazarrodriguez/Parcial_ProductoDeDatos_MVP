"""Register a candidate ModelOps run and promote it to latest if it wins.

The dashboard should read stable paths under ``modelops/latest``. This script
implements a simple champion/challenger policy so the newest model does not
silently replace the current one unless it improves metrics.
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
    parser.add_argument("--run-prefix-template", default="modelops/runs/{run_id}")
    parser.add_argument("--registry-prefix", default="modelops/registry")
    parser.add_argument("--latest-prefix", default="modelops/latest")
    parser.add_argument("--min-improvement", type=float, default=0.01)
    parser.add_argument("--promote-even-if-not-better", action="store_true")
    return parser.parse_args()


def read_json_or_none(s3_client, bucket: str, key: str) -> dict[str, Any] | None:
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except Exception:
        return None


def write_json(s3_client, bucket: str, key: str, payload: dict[str, Any]) -> None:
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, indent=2, default=str).encode("utf-8"),
        ContentType="application/json",
    )


def read_csv_or_empty(s3_client, bucket: str, key: str) -> pd.DataFrame:
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(obj["Body"])
    except Exception:
        return pd.DataFrame()


def write_csv(s3_client, bucket: str, key: str, df: pd.DataFrame) -> None:
    buffer = StringIO()
    df.to_csv(buffer, index=False)
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=buffer.getvalue().encode("utf-8"),
        ContentType="text/csv",
    )


def read_parquet(s3_client, bucket: str, key: str) -> pd.DataFrame:
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(BytesIO(obj["Body"].read()))


def copy_prefix(s3_client, bucket: str, source_prefix: str, target_prefix: str) -> None:
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=source_prefix.rstrip("/") + "/"):
        for obj in page.get("Contents", []):
            source_key = obj["Key"]
            target_key = source_key.replace(source_prefix.rstrip("/") + "/", target_prefix.rstrip("/") + "/", 1)
            print(f"copy s3://{bucket}/{source_key} -> s3://{bucket}/{target_key}")
            s3_client.copy_object(
                Bucket=bucket,
                CopySource={"Bucket": bucket, "Key": source_key},
                Key=target_key,
            )


def extract_metric(metrics: dict[str, Any], candidates: list[str]) -> float | None:
    for key in candidates:
        if key in metrics and metrics[key] is not None:
            try:
                return float(metrics[key])
            except (TypeError, ValueError):
                pass
    return None


def summarize_segment_metrics(segment_df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {"n_segments": int(len(segment_df))}
    for metric in ["rmse_model", "rmse_naive", "wape_model", "wape_naive", "mae_model", "mae_naive"]:
        if metric in segment_df.columns:
            summary[metric] = float(segment_df[metric].mean())
    if {"rmse_model", "rmse_naive"}.issubset(segment_df.columns):
        summary["segments_beating_naive_rate"] = float(
            (segment_df["rmse_model"] <= segment_df["rmse_naive"]).mean()
        )
    return summary


def should_promote(candidate: dict[str, Any], champion: dict[str, Any] | None, min_improvement: float) -> tuple[bool, str]:
    if champion is None:
        return True, "no_existing_champion"

    candidate_rmse = extract_metric(candidate, ["rmse_model", "final_rmse", "rmse_final"])
    champion_rmse = extract_metric(champion, ["rmse_model", "final_rmse", "rmse_final"])

    if candidate_rmse is None or champion_rmse is None:
        return False, "missing_comparable_rmse"

    threshold = champion_rmse * (1 - min_improvement)
    if candidate_rmse <= threshold:
        return True, f"candidate_rmse={candidate_rmse:.5f}_beats_threshold={threshold:.5f}"
    return False, f"candidate_rmse={candidate_rmse:.5f}_not_better_than_champion={champion_rmse:.5f}"


def main() -> None:
    args = parse_args()
    s3_client = boto3.client("s3")
    run_prefix = args.run_prefix_template.format(run_id=args.run_id).strip("/")
    registry_prefix = args.registry_prefix.strip("/")
    latest_prefix = args.latest_prefix.strip("/")

    metrics_key = f"{run_prefix}/evaluation/model_metrics.json"
    segment_key = f"{run_prefix}/evaluation/evaluation_by_segment.parquet"
    champion_key = f"{registry_prefix}/champion.json"
    runs_key = f"{registry_prefix}/model_runs.csv"

    metrics = read_json_or_none(s3_client, args.bucket, metrics_key) or {}
    segment_df = read_parquet(s3_client, args.bucket, segment_key)
    segment_summary = summarize_segment_metrics(segment_df)

    candidate = {
        "model_run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_prefix": f"s3://{args.bucket}/{run_prefix}",
        "model_uri": f"s3://{args.bucket}/{run_prefix}/model/model_bundle.joblib",
        "forecast_uri": f"s3://{args.bucket}/{run_prefix}/predictions/forecast_detail.parquet",
        "metrics_uri": f"s3://{args.bucket}/{metrics_key}",
        "status": "candidate",
        **metrics,
        **segment_summary,
    }

    champion = read_json_or_none(s3_client, args.bucket, champion_key)
    promote, reason = should_promote(candidate, champion, args.min_improvement)
    if args.promote_even_if_not_better:
        promote = True
        reason = "forced_promotion"

    candidate["is_champion"] = bool(promote)
    candidate["status"] = "promoted" if promote else "rejected"
    candidate["promotion_reason"] = reason

    if promote:
        copy_prefix(s3_client, args.bucket, f"{run_prefix}/predictions", f"{latest_prefix}/predictions")
        copy_prefix(s3_client, args.bucket, f"{run_prefix}/evaluation", f"{latest_prefix}/evaluation")
        copy_prefix(s3_client, args.bucket, f"{run_prefix}/model", f"{latest_prefix}/model")
        write_json(s3_client, args.bucket, champion_key, candidate)

    runs_df = read_csv_or_empty(s3_client, args.bucket, runs_key)
    runs_df = pd.concat([runs_df, pd.DataFrame([candidate])], ignore_index=True)
    write_csv(s3_client, args.bucket, runs_key, runs_df)

    segment_buffer = BytesIO()
    segment_df.to_parquet(segment_buffer, index=False)
    s3_client.put_object(
        Bucket=args.bucket,
        Key=f"{registry_prefix}/segment_metrics/{args.run_id}.parquet",
        Body=segment_buffer.getvalue(),
    )

    print(json.dumps(candidate, indent=2, default=str))


if __name__ == "__main__":
    main()
