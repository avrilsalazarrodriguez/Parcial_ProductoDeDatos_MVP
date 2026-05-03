"""Smoke test for ModelOps S3 loaders used by the app."""

from __future__ import annotations

import argparse
import os

from backend.modelops_s3 import (
    load_evaluation_by_segment,
    load_forecast_detail,
    load_model_metrics,
    modelops_healthcheck,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=os.getenv("MODEL_BUCKET"))
    parser.add_argument("--prefix", default=os.getenv("MODELOPS_PREFIX", "modelops/latest"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bucket:
        os.environ["MODEL_BUCKET"] = args.bucket
    if args.prefix:
        os.environ["MODELOPS_PREFIX"] = args.prefix

    print("Healthcheck:")
    print(modelops_healthcheck())

    forecast = load_forecast_detail()
    print("forecast_detail", forecast.shape, forecast.head().to_dict(orient="records")[:2])

    segments = load_evaluation_by_segment()
    print("evaluation_by_segment", segments.shape, segments.head().to_dict(orient="records")[:2])

    metrics = load_model_metrics()
    print("model_metrics", metrics)


if __name__ == "__main__":
    main()
