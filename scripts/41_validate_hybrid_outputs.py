#!/usr/bin/env python3
"""Validate hybrid router outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="modelops_outputs_hybrid")
    args = parser.parse_args()

    root = Path(args.root)
    required = [
        root / "latest/predictions/forecast_detail.parquet",
        root / "latest/evaluation/model_metrics.json",
        root / "latest/evaluation/evaluation_curves_by_model.parquet",
        root / "latest/evaluation/evaluation_by_segment.parquet",
        root / "latest/evaluation/evaluation_by_item.parquet",
        root / "registry/champion.json",
        root / "registry/model_runs.csv",
    ]

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required outputs:\n" + "\n".join(missing))

    forecast = pd.read_parquet(root / "latest/predictions/forecast_detail.parquet")
    runs = pd.read_csv(root / "registry/model_runs.csv")
    metrics = json.loads((root / "latest/evaluation/model_metrics.json").read_text(encoding="utf-8"))

    print("forecast shape:", forecast.shape)
    print("model_scope distribution:")
    print(forecast["model_scope"].value_counts(dropna=False).to_string())
    print("\nmodel runs:")
    print(runs[["model_id", "model_name", "rmse", "mae", "is_champion"]].sort_values(["is_champion", "rmse"], ascending=[False, True]).to_string(index=False))
    print("\nchampion:", metrics.get("champion", {}).get("model_id"))


if __name__ == "__main__":
    main()
