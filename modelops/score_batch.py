"""Generate batch predictions for the app/dashboard and Kaggle-style submission."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from modelops.io import read_parquet, write_csv, write_json, write_parquet


def predict_with_bundle(bundle: dict, features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Predict using selected segment models when available."""
    feature_cols = bundle["feature_cols"]
    target_clip = bundle.get("target_clip", 20.0)
    global_model = bundle["global_model"]
    segment_models = bundle.get("segment_models", {})

    preds = np.clip(global_model.predict(features[feature_cols]), 0, target_clip)
    scopes = np.array(["global"] * len(features), dtype=object)

    if segment_models:
        for segment_key, model in segment_models.items():
            mask = features["segment_key"].astype(str).eq(str(segment_key)).values
            if mask.any():
                preds[mask] = np.clip(model.predict(features.loc[mask, feature_cols]), 0, target_clip)
                scopes[mask] = f"segment:{segment_key}"

    return preds, scopes.tolist()


def score_batch(
    model_path: str,
    inference_features_path: str,
    inference_pairs_path: str,
    output_dir: str,
) -> dict:
    """Score inference data and write prediction tables."""
    bundle = joblib.load(model_path)
    features = read_parquet(inference_features_path)
    pairs = read_parquet(inference_pairs_path)

    preds, scopes = predict_with_bundle(bundle, features)
    result = pairs.copy()
    result["item_cnt_month"] = preds
    result["prediction"] = preds
    result["model_scope"] = scopes

    fields_for_dashboard = [
        "segment_key",
        "segment_name",
        "segment_source",
        "item_category_id",
        "item_category_name",
        "category_group",
        "price_tier",
        "demand_tier",
        "months_active",
        "recency",
    ]
    for col in fields_for_dashboard:
        if col in features.columns:
            result[col] = features[col].values

    base = output_dir.rstrip("/")
    write_parquet(result, f"{base}/forecast_detail.parquet")
    write_csv(result[["ID", "item_cnt_month"]], f"{base}/submission.csv")

    summary = (
        result.groupby(["shop_id", "segment_key", "segment_name"], as_index=False)
        .agg(
            n_items=("item_id", "nunique"),
            forecast_units=("item_cnt_month", "sum"),
            avg_prediction=("item_cnt_month", "mean"),
        )
        .sort_values("forecast_units", ascending=False)
    )
    write_parquet(summary, f"{base}/forecast_summary_by_shop_segment.parquet")

    category_summary = (
        result.groupby(["item_category_id", "item_category_name", "category_group"], as_index=False)
        .agg(
            n_items=("item_id", "nunique"),
            forecast_units=("item_cnt_month", "sum"),
            avg_prediction=("item_cnt_month", "mean"),
        )
        .sort_values("forecast_units", ascending=False)
    )
    write_parquet(category_summary, f"{base}/forecast_summary_by_category.parquet")

    metadata = {
        "n_predictions": int(len(result)),
        "output_dir": output_dir,
        "n_segment_models": len(bundle.get("segment_models", {})),
        "submission_column": "item_cnt_month",
    }
    write_json(metadata, f"{base}/prediction_metadata.json")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--inference-features-path", required=True)
    parser.add_argument("--inference-pairs-path", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = score_batch(
        model_path=args.model_path,
        inference_features_path=args.inference_features_path,
        inference_pairs_path=args.inference_pairs_path,
        output_dir=args.output_dir,
    )
    print(metadata)


if __name__ == "__main__":
    main()
