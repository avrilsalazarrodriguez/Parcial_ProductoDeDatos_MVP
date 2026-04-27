"""Train global and segmented forecasting models.

The model strategy is intentionally pragmatic for the exam:
1. Train one global model across all shop-item pairs.
2. Train selected segment-specific models only when the segment has enough data.
3. Compare against a naive baseline.
4. At inference time, each row uses its segment model if available; otherwise it
   falls back to the global model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from modelops.config import FeatureConfig, NUMERIC_FEATURES
from modelops.io import read_parquet, write_json, write_parquet


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE, robust to zeros."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(2 * np.abs(y_pred - y_true) / denom))


def make_model(random_state: int) -> HistGradientBoostingRegressor:
    """Create the default model."""
    return HistGradientBoostingRegressor(
        max_iter=180,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        random_state=random_state,
    )


def evaluate_predictions(df: pd.DataFrame, pred_col: str) -> dict:
    """Evaluate one prediction column."""
    return {
        "rmse": rmse(df["target"], df[pred_col]),
        "mae": float(mean_absolute_error(df["target"], df[pred_col])),
        "smape": smape(df["target"], df[pred_col]),
    }


def train_models(
    train_path: str,
    valid_path: str,
    model_output_path: str,
    evaluation_output_dir: str,
    cfg: FeatureConfig | None = None,
) -> dict:
    """Train global and segment-specific models."""
    cfg = cfg or FeatureConfig()
    train = read_parquet(train_path)
    valid = read_parquet(valid_path)

    feature_cols = [col for col in NUMERIC_FEATURES if col in train.columns]
    train = train.copy()
    valid = valid.copy()
    train["target"] = train["target"].clip(0, cfg.target_clip)
    valid["target"] = valid["target"].clip(0, cfg.target_clip)

    global_model = make_model(cfg.random_state)
    global_model.fit(train[feature_cols], train["target"])

    valid["pred_global"] = np.clip(global_model.predict(valid[feature_cols]), 0, cfg.target_clip)
    valid["pred_naive"] = valid["cnt_lag_1"].fillna(0).clip(0, cfg.target_clip)

    global_metrics = evaluate_predictions(valid, "pred_global")
    naive_metrics = evaluate_predictions(valid, "pred_naive")

    segment_counts = (
        train.groupby("segment_key", as_index=False)
        .size()
        .rename(columns={"size": "n_train"})
        .merge(
            valid.groupby("segment_key", as_index=False).size().rename(columns={"size": "n_valid"}),
            on="segment_key",
            how="left",
        )
        .fillna({"n_valid": 0})
        .sort_values("n_train", ascending=False)
    )

    candidate_segments = segment_counts[
        (segment_counts["n_train"] >= cfg.min_segment_train_rows)
        & (segment_counts["n_valid"] >= cfg.min_segment_valid_rows)
    ]["segment_key"].head(cfg.max_segment_models).tolist()

    segment_models = {}
    segment_metric_rows = []
    valid["pred_final"] = valid["pred_global"]
    valid["model_scope"] = "global"

    for segment_key in candidate_segments:
        seg_train = train[train["segment_key"] == segment_key]
        seg_valid = valid[valid["segment_key"] == segment_key].copy()
        if seg_train.empty or seg_valid.empty:
            continue

        seg_model = make_model(cfg.random_state)
        seg_model.fit(seg_train[feature_cols], seg_train["target"])

        seg_valid["pred_segment"] = np.clip(seg_model.predict(seg_valid[feature_cols]), 0, cfg.target_clip)
        seg_valid["pred_global"] = valid.loc[seg_valid.index, "pred_global"]
        seg_valid["pred_naive"] = valid.loc[seg_valid.index, "pred_naive"]

        metrics_segment = evaluate_predictions(seg_valid, "pred_segment")
        metrics_global = evaluate_predictions(seg_valid, "pred_global")
        metrics_naive = evaluate_predictions(seg_valid, "pred_naive")

        # Keep segment model only if it beats global or naive on RMSE.
        keep_segment_model = metrics_segment["rmse"] <= min(
            metrics_global["rmse"],
            metrics_naive["rmse"],
        )

        segment_metric_rows.append(
            {
                "segment_key": segment_key,
                "n_train": int(len(seg_train)),
                "n_valid": int(len(seg_valid)),
                "kept": bool(keep_segment_model),
                "rmse_segment": metrics_segment["rmse"],
                "rmse_global": metrics_global["rmse"],
                "rmse_naive": metrics_naive["rmse"],
                "mae_segment": metrics_segment["mae"],
                "mae_global": metrics_global["mae"],
                "mae_naive": metrics_naive["mae"],
                "smape_segment": metrics_segment["smape"],
                "smape_global": metrics_global["smape"],
                "smape_naive": metrics_naive["smape"],
            }
        )

        if keep_segment_model:
            segment_models[segment_key] = seg_model
            valid.loc[seg_valid.index, "pred_final"] = seg_valid["pred_segment"]
            valid.loc[seg_valid.index, "model_scope"] = "segment"

    valid["abs_error_final"] = (valid["target"] - valid["pred_final"]).abs()
    valid["abs_error_naive"] = (valid["target"] - valid["pred_naive"]).abs()
    valid["beats_naive"] = valid["abs_error_final"] <= valid["abs_error_naive"]

    final_metrics = evaluate_predictions(valid, "pred_final")
    metrics = {
        "global": global_metrics,
        "naive": naive_metrics,
        "final": final_metrics,
        "n_segment_models": len(segment_models),
        "feature_cols": feature_cols,
    }

    bundle = {
        "global_model": global_model,
        "segment_models": segment_models,
        "feature_cols": feature_cols,
        "target_clip": cfg.target_clip,
        "metrics": metrics,
        "model_strategy": "global_plus_selected_segment_models",
    }

    model_path = Path(model_output_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)

    eval_base = evaluation_output_dir.rstrip("/")
    segment_metrics_df = pd.DataFrame(segment_metric_rows)
    if segment_metrics_df.empty:
        segment_metrics_df = pd.DataFrame(
            columns=[
                "segment_key",
                "n_train",
                "n_valid",
                "kept",
                "rmse_segment",
                "rmse_global",
                "rmse_naive",
            ]
        )

    write_parquet(valid, f"{eval_base}/evaluation_detail.parquet")
    write_parquet(segment_metrics_df, f"{eval_base}/evaluation_by_segment.parquet")

    by_item = (
        valid.groupby(["item_id", "segment_key"], as_index=False)
        .agg(
            n=("target", "size"),
            y_true_mean=("target", "mean"),
            pred_mean=("pred_final", "mean"),
            mae=("abs_error_final", "mean"),
            naive_mae=("abs_error_naive", "mean"),
            beats_naive_rate=("beats_naive", "mean"),
        )
        .sort_values("mae", ascending=False)
    )
    write_parquet(by_item, f"{eval_base}/evaluation_by_item.parquet")
    write_json(metrics, f"{eval_base}/model_metrics.json")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--valid-path", required=True)
    parser.add_argument("--model-output-path", required=True)
    parser.add_argument("--evaluation-output-dir", required=True)
    parser.add_argument("--min-segment-train-rows", type=int, default=25_000)
    parser.add_argument("--min-segment-valid-rows", type=int, default=1_000)
    parser.add_argument("--max-segment-models", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = FeatureConfig(
        min_segment_train_rows=args.min_segment_train_rows,
        min_segment_valid_rows=args.min_segment_valid_rows,
        max_segment_models=args.max_segment_models,
    )
    metrics = train_models(
        train_path=args.train_path,
        valid_path=args.valid_path,
        model_output_path=args.model_output_path,
        evaluation_output_dir=args.evaluation_output_dir,
        cfg=cfg,
    )
    print(metrics)


if __name__ == "__main__":
    main()
