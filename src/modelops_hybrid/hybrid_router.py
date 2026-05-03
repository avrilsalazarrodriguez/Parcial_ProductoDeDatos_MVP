"""Hybrid router for intermittent demand forecasting.

This module is intentionally independent from the Streamlit frontend. It creates
ModelOps outputs that the current app already knows how to read.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


TARGET_COL = "y"
MAX_PREDICTION = 20.0


@dataclass
class Metrics:
    model_id: str
    model_name: str
    mae: float
    rmse: float
    smape: float
    wape: float
    bias: float
    nonzero_recall: float
    n_valid: int
    status: str = "candidate"
    model_family: str = "hybrid"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_num(s: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(default)


def clip_pred(values: Any, max_value: float = MAX_PREDICTION) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=max_value, neginf=0.0)
    return np.clip(arr, 0.0, max_value)


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add demand-regime features from historical lag columns.

    The rules only use features available before prediction time.
    """
    out = df.copy()
    lag_cols = [c for c in [f"cnt_lag_{i}" for i in range(1, 13)] if c in out.columns]

    if "cnt_lag_1" not in out.columns:
        out["cnt_lag_1"] = 0.0

    if lag_cols:
        lags = out[lag_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        out["rolling_mean_3_router"] = lags[[c for c in lag_cols if c in {"cnt_lag_1", "cnt_lag_2", "cnt_lag_3"}]].mean(axis=1)
        out["rolling_mean_6_router"] = lags[[c for c in lag_cols if c in {f"cnt_lag_{i}" for i in range(1, 7)}]].mean(axis=1)
        out["rolling_sum_6_router"] = lags[[c for c in lag_cols if c in {f"cnt_lag_{i}" for i in range(1, 7)}]].sum(axis=1)
        out["nonzero_rate_6_router"] = (lags[[c for c in lag_cols if c in {f"cnt_lag_{i}" for i in range(1, 7)}]] > 0).mean(axis=1)
    else:
        out["rolling_mean_3_router"] = 0.0
        out["rolling_mean_6_router"] = 0.0
        out["rolling_sum_6_router"] = 0.0
        out["nonzero_rate_6_router"] = 0.0

    if "recency" in out.columns:
        recency = safe_num(out["recency"], default=99)
    else:
        recency = pd.Series(0, index=out.index)

    out["router_inactive_flag"] = ((recency >= 99) | (out["rolling_sum_6_router"] <= 0)).astype(int)
    out["router_recent_flag"] = (safe_num(out["cnt_lag_1"]) >= 1).astype(int)
    out["router_recurrent_flag"] = (
        (out["rolling_mean_3_router"] >= 1)
        | (out["rolling_mean_6_router"] >= 1)
        | (out["nonzero_rate_6_router"] >= 0.40)
    ).astype(int)
    return out


def feature_columns(df: pd.DataFrame, target_col: str = TARGET_COL) -> list[str]:
    exclude = {
        target_col,
        "item_cnt_month",
        "prediction",
        "pred_champion",
        "pred_naive_lag1",
        "pred_rolling_mean_3",
        "pred_specialist_recurrent",
        "pred_hybrid_router",
        "model_scope",
        "routing_reason",
    }
    numeric_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    return numeric_cols


def train_recurrent_specialist(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = TARGET_COL,
    min_rows: int = 500,
) -> tuple[Any | None, list[str]]:
    train_df = add_regime_features(train_df)
    mask = (
        (train_df["router_inactive_flag"] == 0)
        & ((train_df["router_recent_flag"] == 1) | (train_df["router_recurrent_flag"] == 1))
        & train_df[target_col].notna()
    )
    subset = train_df.loc[mask].copy()

    if len(subset) < min_rows:
        return None, feature_cols

    feature_cols = [c for c in feature_cols if c in subset.columns]
    x_train = subset[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y_train = np.log1p(clip_pred(subset[target_col]))

    model = HistGradientBoostingRegressor(
        max_iter=180,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=42,
    )
    model.fit(x_train, y_train)
    return model, feature_cols


def predict_specialist(model: Any | None, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    if model is None:
        return clip_pred(df["rolling_mean_3_router"] if "rolling_mean_3_router" in df.columns else 0.0)
    x = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    pred_log = model.predict(x)
    return clip_pred(np.expm1(pred_log))


def load_two_stage_model(model_path: Path) -> dict[str, Any] | None:
    if not model_path.exists():
        return None
    payload = joblib.load(model_path)
    if isinstance(payload, dict):
        return payload.get("bundle", payload)
    return None


def predict_two_stage(payload: dict[str, Any] | None, features: pd.DataFrame) -> np.ndarray:
    if payload is None:
        return np.zeros(len(features), dtype=float)

    clf = payload.get("clf")
    reg = payload.get("reg")
    cols = payload.get("feature_cols")

    if clf is None or reg is None or cols is None:
        return np.zeros(len(features), dtype=float)

    x = features[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    proba = clf.predict_proba(x)[:, 1]
    magnitude = reg.predict(x)
    return clip_pred(proba * magnitude)


def component_predictions(
    df: pd.DataFrame,
    champion_pred: np.ndarray,
    specialist_model: Any | None,
    specialist_features: list[str],
) -> pd.DataFrame:
    out = add_regime_features(df)
    out["pred_champion"] = clip_pred(champion_pred)
    out["pred_naive_lag1"] = clip_pred(out["cnt_lag_1"] if "cnt_lag_1" in out.columns else 0.0)
    out["pred_rolling_mean_3"] = clip_pred(out["rolling_mean_3_router"])
    out["pred_specialist_recurrent"] = predict_specialist(specialist_model, out, specialist_features)
    out["pred_inactive_zero"] = 0.0
    return out


def route_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Apply frozen production-like routing policy."""
    out = df.copy()

    conditions = [
        out["router_inactive_flag"].eq(1),
        out["router_recent_flag"].eq(1),
        out["router_recurrent_flag"].eq(1),
    ]
    scopes = [
        "inactive:no_recent_sales",
        "baseline:naive_recent_demand",
        "specialist:recurrent_demand",
    ]
    reasons = [
        "recency>=99_or_rolling_sum_6==0",
        "cnt_lag_1>=1",
        "rolling_mean_or_nonzero_rate_signal",
    ]

    out["model_scope"] = "challenger:hurdle_hgb"
    out["routing_reason"] = "default_sparse_low_demand"
    out["prediction"] = out["pred_champion"]

    for cond, scope, reason in zip(conditions, scopes, reasons):
        out.loc[cond, "model_scope"] = scope
        out.loc[cond, "routing_reason"] = reason

    out.loc[out["model_scope"].eq("inactive:no_recent_sales"), "prediction"] = out.loc[
        out["model_scope"].eq("inactive:no_recent_sales"), "pred_inactive_zero"
    ]
    out.loc[out["model_scope"].eq("baseline:naive_recent_demand"), "prediction"] = out.loc[
        out["model_scope"].eq("baseline:naive_recent_demand"), "pred_naive_lag1"
    ]
    out.loc[out["model_scope"].eq("specialist:recurrent_demand"), "prediction"] = out.loc[
        out["model_scope"].eq("specialist:recurrent_demand"), "pred_specialist_recurrent"
    ]

    out["prediction"] = clip_pred(out["prediction"])
    return out


def compute_metrics(y_true: pd.Series, y_pred: pd.Series, model_id: str, model_name: str, family: str = "hybrid") -> Metrics:
    y = np.asarray(pd.to_numeric(y_true, errors="coerce").fillna(0.0), dtype=float)
    p = clip_pred(pd.to_numeric(y_pred, errors="coerce").fillna(0.0))
    err = p - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = np.abs(y) + np.abs(p)
    smape = float(np.mean(np.where(denom == 0, 0.0, 2 * np.abs(err) / denom)))
    wape = float(np.sum(np.abs(err)) / max(np.sum(np.abs(y)), 1e-9))
    bias = float(np.mean(err))
    positive_true = y > 0
    recall = float(np.mean(p[positive_true] > 0)) if positive_true.any() else 0.0
    return Metrics(model_id, model_name, mae, rmse, smape, wape, bias, recall, len(y), model_family=family)


def demand_curve(eval_df: pd.DataFrame, predictions: dict[str, str]) -> pd.DataFrame:
    df = eval_df.copy()
    df["demand_bucket"] = pd.cut(df[TARGET_COL], bins=[-0.1, 0, 1, 3, 7, 20], include_lowest=True).astype(str)
    rows = []
    for model_id, pred_col in predictions.items():
        if pred_col not in df.columns:
            continue
        part = (
            df.groupby("demand_bucket", as_index=False, observed=False)
            .agg(real_mean=(TARGET_COL, "mean"), pred_mean=(pred_col, "mean"))
        )
        part["model_id"] = model_id
        part["model_name"] = {
            "hybrid_router_v1": "Hybrid Router",
            "hurdle_hgb": "Hurdle HGB",
            "naive_lag1": "Naive",
            "rolling_mean_3": "Rolling mean 3",
            "specialist_recurrent": "Recurrent specialist",
        }.get(model_id, model_id)
        rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def evaluation_by_segment(eval_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [c for c in ["item_category_id", "item_category_name", "category_group"] if c in eval_df.columns]
    if not group_cols:
        group_cols = ["model_scope"]
    return (
        eval_df.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            n=("prediction", "size"),
            y_mean=(TARGET_COL, "mean"),
            pred_mean=("prediction", "mean"),
            mae=("abs_error", "mean"),
            rmse=("squared_error", lambda x: float(np.sqrt(np.mean(x)))),
            wape=("abs_error", lambda x: float(np.sum(x) / max(np.sum(np.abs(eval_df.loc[x.index, TARGET_COL])), 1e-9))),
        )
        .sort_values("rmse", ascending=False)
    )


def evaluation_by_item(eval_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [c for c in ["item_id", "item_name"] if c in eval_df.columns]
    if not group_cols:
        group_cols = ["item_id"] if "item_id" in eval_df.columns else ["model_scope"]
    return (
        eval_df.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            n=("prediction", "size"),
            y_mean=(TARGET_COL, "mean"),
            pred_mean=("prediction", "mean"),
            mae=("abs_error", "mean"),
            rmse=("squared_error", lambda x: float(np.sqrt(np.mean(x)))),
        )
        .sort_values("rmse", ascending=False)
    )


def enrich_catalog(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    out = df.copy()
    raw = data_dir / "raw"

    items_path = next((p for p in [raw / "items_en.csv", raw / "items.csv", data_dir / "items_en.csv", data_dir / "items.csv"] if p.exists()), None)
    cats_path = next((p for p in [raw / "item_categories_en.csv", raw / "item_categories.csv", data_dir / "item_categories_en.csv", data_dir / "item_categories.csv"] if p.exists()), None)
    shops_path = next((p for p in [raw / "shops_en.csv", raw / "shops.csv", data_dir / "shops_en.csv", data_dir / "shops.csv"] if p.exists()), None)

    if items_path and "item_id" in out.columns:
        items = pd.read_csv(items_path)
        if "item_name" not in items.columns:
            name_col = next((c for c in items.columns if "name" in c.lower()), None)
            if name_col:
                items["item_name"] = items[name_col]
        keep = [c for c in ["item_id", "item_name", "item_category_id"] if c in items.columns]
        out = out.drop(columns=[c for c in ["item_name", "item_category_id"] if c in out.columns], errors="ignore")
        out = out.merge(items[keep].drop_duplicates("item_id"), on="item_id", how="left")

    if cats_path and "item_category_id" in out.columns:
        cats = pd.read_csv(cats_path)
        if "item_category_name" not in cats.columns:
            name_col = next((c for c in cats.columns if "category" in c.lower() and c != "item_category_id"), None)
            if name_col:
                cats["item_category_name"] = cats[name_col]
        if "item_category_name" in cats.columns:
            cats["category_group"] = cats["item_category_name"].astype(str).str.split(" - ").str[0]
            keep = ["item_category_id", "item_category_name", "category_group"]
            out = out.drop(columns=[c for c in ["item_category_name", "category_group"] if c in out.columns], errors="ignore")
            out = out.merge(cats[keep].drop_duplicates("item_category_id"), on="item_category_id", how="left")

    if shops_path and "shop_id" in out.columns:
        shops = pd.read_csv(shops_path)
        if "shop_name" not in shops.columns:
            name_col = next((c for c in shops.columns if "name" in c.lower()), None)
            if name_col:
                shops["shop_name"] = shops[name_col]
        if "shop_name" in shops.columns:
            out = out.drop(columns=["shop_name"], errors="ignore")
            out = out.merge(shops[["shop_id", "shop_name"]].drop_duplicates("shop_id"), on="shop_id", how="left")

    return out


def forecast_summaries(forecast: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "prediction" not in forecast.columns:
        return pd.DataFrame(), pd.DataFrame()
    cat_cols = [c for c in ["item_category_id", "item_category_name", "category_group"] if c in forecast.columns]
    if cat_cols:
        by_cat = (
            forecast.groupby(cat_cols, dropna=False, as_index=False)
            .agg(n_items=("prediction", "size"), forecast_units=("prediction", "sum"), avg_prediction=("prediction", "mean"))
            .sort_values("forecast_units", ascending=False)
        )
    else:
        by_cat = pd.DataFrame()

    shop_cols = [c for c in ["shop_id", "shop_name", "model_scope"] if c in forecast.columns]
    by_shop = (
        forecast.groupby(shop_cols, dropna=False, as_index=False)
        .agg(n_items=("prediction", "size"), forecast_units=("prediction", "sum"), avg_prediction=("prediction", "mean"))
        .sort_values("forecast_units", ascending=False)
        if shop_cols
        else pd.DataFrame()
    )
    return by_cat, by_shop


def write_outputs(
    output_root: Path,
    forecast: pd.DataFrame,
    eval_detail: pd.DataFrame,
    metrics: list[Metrics],
    curves: pd.DataFrame,
    champion_id: str,
) -> None:
    latest = output_root / "latest"
    registry = output_root / "registry"
    for sub in ["predictions", "evaluation", "review", "model"]:
        ensure_dir(latest / sub)
    ensure_dir(registry)

    forecast.to_parquet(latest / "predictions" / "forecast_detail.parquet", index=False)
    by_cat, by_shop = forecast_summaries(forecast)
    if not by_cat.empty:
        by_cat.to_parquet(latest / "predictions" / "forecast_summary_by_category.parquet", index=False)
    if not by_shop.empty:
        by_shop.to_parquet(latest / "predictions" / "forecast_summary_by_shop_segment.parquet", index=False)

    eval_detail.to_parquet(latest / "evaluation" / "evaluation_detail.parquet", index=False)
    evaluation_by_segment(eval_detail).to_parquet(latest / "evaluation" / "evaluation_by_segment.parquet", index=False)
    evaluation_by_item(eval_detail).to_parquet(latest / "evaluation" / "evaluation_by_item.parquet", index=False)
    curves.to_parquet(latest / "evaluation" / "evaluation_curves_by_model.parquet", index=False)

    eval_detail.sort_values("error_signed", ascending=False).head(100).to_parquet(latest / "review" / "overestimated_products.parquet", index=False)
    eval_detail.sort_values("error_signed", ascending=True).head(100).to_parquet(latest / "review" / "underestimated_products.parquet", index=False)
    eval_detail.sort_values("abs_error", ascending=False).head(200).to_parquet(latest / "review" / "review_suggestions.parquet", index=False)

    metrics_df = pd.DataFrame([asdict(m) for m in metrics]).sort_values(["rmse", "mae"])
    metrics_df["is_champion"] = metrics_df["model_id"].eq(champion_id)
    metrics_df.to_csv(registry / "model_runs.csv", index=False)

    champion = metrics_df.loc[metrics_df["model_id"].eq(champion_id)].iloc[0].to_dict()
    champion["selection_policy"] = "primary=rmse; tie_breaker=mae; hybrid router learned on train and evaluated on validation"
    (registry / "champion.json").write_text(json.dumps(champion, indent=2, ensure_ascii=False), encoding="utf-8")

    payload = {
        "selection_policy": champion["selection_policy"],
        "champion": champion,
        "global": champion,
        "all_models": metrics_df.to_dict(orient="records"),
    }
    (latest / "evaluation" / "model_metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
