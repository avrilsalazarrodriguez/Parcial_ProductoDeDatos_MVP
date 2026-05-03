#!/usr/bin/env python3
from __future__ import annotations

"""Generate valid CSV examples for the uploaded batch inference UI.

The best examples are generated from local data/prep/test_features.parquet so
columns match the model's feature_cols exactly.
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="artifacts/model.joblib")
    parser.add_argument("--test-features-path", default="data/prep/test_features.parquet")
    parser.add_argument("--test-pairs-path", default="data/prep/test_pairs.parquet")
    parser.add_argument("--output-dir", default="data/examples/batch_upload")
    parser.add_argument("--rows", type=int, default=25)
    return parser.parse_args()


def feature_cols_from_model(model_path: Path) -> list[str]:
    payload = joblib.load(model_path)
    bundle = payload.get("bundle", payload) if isinstance(payload, dict) else payload
    if isinstance(bundle, dict) and "feature_cols" in bundle:
        return list(bundle["feature_cols"])
    if hasattr(bundle, "feature_names_in_"):
        return [str(c) for c in bundle.feature_names_in_]
    raise ValueError("No pude leer feature_cols del modelo.")


def enrich_with_ids(features: pd.DataFrame, pairs_path: Path) -> pd.DataFrame:
    out = features.copy()
    if pairs_path.exists():
        pairs = pd.read_parquet(pairs_path)
        for col in ["shop_id", "item_id", "item_category_id"]:
            if col in pairs.columns and col not in out.columns and len(pairs) == len(out):
                out[col] = pairs[col].to_numpy()
    return out


def write_sample(df: pd.DataFrame, path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.head(rows).to_csv(path, index=False)
    print("wrote", path, "rows", min(rows, len(df)))


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    features_path = Path(args.test_features_path)
    pairs_path = Path(args.test_pairs_path)
    outdir = Path(args.output_dir)

    feature_cols = feature_cols_from_model(model_path)
    features = pd.read_parquet(features_path)
    features = enrich_with_ids(features, pairs_path)

    keep_cols = []
    for col in ["shop_id", "item_id", "item_category_id"] + feature_cols:
        if col in features.columns and col not in keep_cols:
            keep_cols.append(col)
    base = features[keep_cols].copy()

    write_sample(base, outdir / "batch_upload_mixed_catalog_sample.csv", args.rows)

    recent = base.copy()
    lag1_cols = [c for c in recent.columns if c in {"cnt_lag_1", "lag_1", "target_lag_1", "item_cnt_month_lag_1"}]
    if lag1_cols:
        recent = recent[pd.to_numeric(recent[lag1_cols[0]], errors="coerce").fillna(0) >= 1]
    write_sample(recent if not recent.empty else base, outdir / "batch_upload_recent_demand_sample.csv", args.rows)

    inactive = base.copy()
    if "recency" in inactive.columns:
        inactive = inactive[pd.to_numeric(inactive["recency"], errors="coerce").fillna(99) >= 99]
    write_sample(inactive if not inactive.empty else base.tail(args.rows), outdir / "batch_upload_inactive_sample.csv", args.rows)

    print("feature_cols:", len(feature_cols))
    print("output_dir:", outdir)


if __name__ == "__main__":
    main()
