"""Validación rápida de outputs ModelOps v2 locales o S3."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from backend.modelops_s3 import (
    load_champion,
    load_evaluation_by_item,
    load_evaluation_by_segment,
    load_forecast_detail,
    load_model_metrics,
    load_model_runs,
    load_review_suggestions,
)


def assert_not_empty(name: str, df: pd.DataFrame) -> None:
    if df.empty:
        raise AssertionError(f"{name} está vacío")
    print(f"OK {name}: {df.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Valida outputs ModelOps v2")
    parser.add_argument("--bucket", type=str, default="")
    parser.add_argument("--prefix", type=str, default="modelops/latest")
    parser.add_argument("--registry-prefix", type=str, default="modelops/registry")
    parser.add_argument("--local-root", type=Path, default=Path("modelops_outputs"))
    args = parser.parse_args()

    if args.bucket:
        os.environ["USE_MODELOPS_S3"] = "true"
        os.environ["MODEL_BUCKET"] = args.bucket
    else:
        os.environ["USE_MODELOPS_S3"] = "false"
    os.environ["MODELOPS_PREFIX"] = args.prefix
    os.environ["MODELOPS_REGISTRY_PREFIX"] = args.registry_prefix
    os.environ["MODELOPS_LOCAL_ROOT"] = str(args.local_root)

    forecast = load_forecast_detail()
    runs = load_model_runs()
    suggestions = load_review_suggestions()
    by_item = load_evaluation_by_item()
    by_segment = load_evaluation_by_segment()
    metrics = load_model_metrics()
    champion = load_champion()

    assert_not_empty("forecast_detail", forecast)
    assert_not_empty("model_runs", runs)
    assert_not_empty("review_suggestions", suggestions)
    assert_not_empty("evaluation_by_item", by_item)
    assert_not_empty("evaluation_by_segment", by_segment)

    required_forecast_cols = ["shop_name", "item_name", "prediction", "model_scope", "model_scope_explanation"]
    missing = [col for col in required_forecast_cols if col not in forecast.columns]
    if missing:
        raise AssertionError(f"forecast_detail no tiene columnas requeridas: {missing}")

    print("OK champion:", champion.get("model_name"), champion.get("metrics", {}))
    print("OK metrics keys:", list(metrics.keys()))


if __name__ == "__main__":
    main()
