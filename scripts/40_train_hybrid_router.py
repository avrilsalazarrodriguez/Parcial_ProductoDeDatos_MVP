#!/usr/bin/env python3
"""Train/evaluate a hybrid router for intermittent demand.

This script does not change the Streamlit frontend. It produces the same
ModelOps artifact layout that the app already consumes.

Example:
    PYTHONPATH=. uv run python scripts/40_train_hybrid_router.py \
      --train-path data/prep/train.parquet \
      --valid-path data/prep/valid.parquet \
      --test-features-path data/prep/test_features.parquet \
      --test-pairs-path data/prep/test_pairs.parquet \
      --forecast-detail-path modelops_outputs/latest/predictions/forecast_detail.parquet \
      --model-path artifacts/model.joblib \
      --output-root modelops_outputs_hybrid
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.modelops_hybrid.hybrid_router import (
    TARGET_COL,
    add_regime_features,
    component_predictions,
    compute_metrics,
    demand_curve,
    enrich_catalog,
    feature_columns,
    load_two_stage_model,
    predict_two_stage,
    route_predictions,
    train_recurrent_specialist,
    write_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/prep/train.parquet")
    parser.add_argument("--valid-path", default="data/prep/valid.parquet")
    parser.add_argument("--test-features-path", default="data/prep/test_features.parquet")
    parser.add_argument("--test-pairs-path", default="data/prep/test_pairs.parquet")
    parser.add_argument("--forecast-detail-path", default="modelops_outputs/latest/predictions/forecast_detail.parquet")
    parser.add_argument("--model-path", default="artifacts/model.joblib")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-root", default="modelops_outputs_hybrid")
    parser.add_argument("--specialist-min-rows", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = Path(args.train_path)
    valid_path = Path(args.valid_path)
    test_features_path = Path(args.test_features_path)
    test_pairs_path = Path(args.test_pairs_path)
    forecast_detail_path = Path(args.forecast_detail_path)
    model_path = Path(args.model_path)
    data_dir = Path(args.data_dir)
    output_root = Path(args.output_root)

    train_df = pd.read_parquet(train_path)
    valid_df = pd.read_parquet(valid_path)
    test_features = pd.read_parquet(test_features_path)
    test_pairs = pd.read_parquet(test_pairs_path)
    forecast_detail = pd.read_parquet(forecast_detail_path)

    if TARGET_COL not in train_df.columns or TARGET_COL not in valid_df.columns:
        raise ValueError("train/valid parquet must contain target column 'y'")

    train_df = add_regime_features(train_df)
    valid_df = add_regime_features(valid_df)
    test_features = add_regime_features(test_features)

    candidate_features = feature_columns(train_df)
    specialist_model, specialist_features = train_recurrent_specialist(
        train_df,
        feature_cols=candidate_features,
        min_rows=args.specialist_min_rows,
    )

    two_stage = load_two_stage_model(model_path)
    pred_champion_valid = predict_two_stage(two_stage, valid_df)
    pred_champion_test = forecast_detail["prediction"].to_numpy() if "prediction" in forecast_detail.columns else predict_two_stage(two_stage, test_features)

    valid_components = component_predictions(valid_df, pred_champion_valid, specialist_model, specialist_features)
    valid_router = route_predictions(valid_components)

    # Evaluation detail
    eval_detail = enrich_catalog(valid_router.copy(), data_dir)
    eval_detail["abs_error"] = (eval_detail["prediction"] - eval_detail[TARGET_COL]).abs()
    eval_detail["squared_error"] = (eval_detail["prediction"] - eval_detail[TARGET_COL]) ** 2
    eval_detail["error_signed"] = eval_detail["prediction"] - eval_detail[TARGET_COL]

    # Candidate metrics
    metrics = [
        compute_metrics(valid_router[TARGET_COL], valid_router["prediction"], "hybrid_router_v1", "Hybrid Router: inactive + naive + specialist + HGB", family="hybrid"),
        compute_metrics(valid_router[TARGET_COL], valid_router["pred_champion"], "hurdle_hgb", "Hurdle HGB", family="challenger"),
        compute_metrics(valid_router[TARGET_COL], valid_router["pred_naive_lag1"], "naive_lag1", "Naive último periodo", family="baseline"),
        compute_metrics(valid_router[TARGET_COL], valid_router["pred_rolling_mean_3"], "rolling_mean_3", "Rolling mean últimos 3 lags", family="baseline"),
        compute_metrics(valid_router[TARGET_COL], valid_router["pred_specialist_recurrent"], "specialist_recurrent", "Especialista demanda recurrente", family="specialist"),
    ]

    # RMSE primary, MAE tie-breaker
    best = sorted(metrics, key=lambda m: (m.rmse, m.mae))[0]
    champion_id = best.model_id

    curves = demand_curve(
        valid_router,
        {
            "hybrid_router_v1": "prediction",
            "hurdle_hgb": "pred_champion",
            "naive_lag1": "pred_naive_lag1",
            "rolling_mean_3": "pred_rolling_mean_3",
            "specialist_recurrent": "pred_specialist_recurrent",
        },
    )

    # Future forecast
    test_components = component_predictions(test_features, pred_champion_test, specialist_model, specialist_features)
    test_router = route_predictions(test_components)

    forecast = forecast_detail.copy()
    for col in ["prediction", "model_scope", "routing_reason", "pred_champion", "pred_naive_lag1", "pred_rolling_mean_3", "pred_specialist_recurrent"]:
        forecast[col] = test_router[col].to_numpy()

    if {"shop_id", "item_id"}.issubset(test_pairs.columns):
        if "shop_id" not in forecast.columns:
            forecast["shop_id"] = test_pairs["shop_id"].to_numpy()
        if "item_id" not in forecast.columns:
            forecast["item_id"] = test_pairs["item_id"].to_numpy()

    forecast = enrich_catalog(forecast, data_dir)
    write_outputs(output_root, forecast, eval_detail, metrics, curves, champion_id)

    print("Hybrid router outputs written to:", output_root)
    print("Champion:", champion_id)
    for m in sorted(metrics, key=lambda m: (m.rmse, m.mae)):
        print(f"{m.model_id:24s} rmse={m.rmse:.4f} mae={m.mae:.4f} wape={m.wape:.4f}")


if __name__ == "__main__":
    main()
