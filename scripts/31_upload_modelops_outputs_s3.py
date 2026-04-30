"""Upload local modelops_outputs to the existing ModelOps S3 layout."""

from __future__ import annotations

import argparse
from pathlib import Path

import boto3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--local-root", type=Path, default=Path("modelops_outputs"))
    parser.add_argument("--latest-prefix", default="modelops/latest")
    parser.add_argument("--registry-prefix", default="modelops/registry")
    return parser.parse_args()


def upload_file(s3, bucket: str, local_path: Path, key: str) -> None:
    s3.upload_file(str(local_path), bucket, key)
    print(f"uploaded s3://{bucket}/{key}")


def main() -> None:
    args = parse_args()
    root = args.local_root
    s3 = boto3.client("s3")
    mapping = {
        root / "predictions" / "forecast_detail.parquet": f"{args.latest_prefix}/predictions/forecast_detail.parquet",
        root / "predictions" / "forecast_detail.csv": f"{args.latest_prefix}/predictions/forecast_detail.csv",
        root / "evaluation" / "evaluation_detail.parquet": f"{args.latest_prefix}/evaluation/evaluation_detail.parquet",
        root / "evaluation" / "evaluation_by_segment.parquet": f"{args.latest_prefix}/evaluation/evaluation_by_segment.parquet",
        root / "evaluation" / "evaluation_by_item.parquet": f"{args.latest_prefix}/evaluation/evaluation_by_item.parquet",
        root / "evaluation" / "model_metrics.json": f"{args.latest_prefix}/evaluation/model_metrics.json",
        root / "evaluation" / "model_zoo_summary.csv": f"{args.latest_prefix}/evaluation/model_zoo_summary.csv",
        root / "registry" / "champion.json": f"{args.registry_prefix}/champion.json",
        root / "registry" / "model_runs.csv": f"{args.registry_prefix}/model_runs.csv",
        root / "review" / "review_suggestions.parquet": f"{args.latest_prefix}/review/review_suggestions.parquet",
        root / "review" / "review_suggestions.csv": f"{args.latest_prefix}/review/review_suggestions.csv",
    }
    for local_path, key in mapping.items():
        if local_path.exists():
            upload_file(s3, args.bucket, local_path, key)
        else:
            print(f"skip missing {local_path}")


if __name__ == "__main__":
    main()
