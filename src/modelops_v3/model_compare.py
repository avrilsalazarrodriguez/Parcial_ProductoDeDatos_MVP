"""Compare the original two-stage model against challengers and publish ModelOps outputs.

This script preserves the original model design as an incumbent candidate. It
adds challenger models and publishes a registry plus UI-friendly evaluation
artifacts.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.modelops_v3.metrics import ModelRunResult, compute_metrics, results_df, select_champion, clip_target

MODEL_DESCRIPTIONS = {
    "incumbent_two_stage": "Modelo original de dos etapas del proyecto: clasifica venta/no venta y luego estima unidades positivas.",
    "naive_lag1": "Baseline naive que usa la venta del último periodo observado.",
    "item_mean_fallback": "Promedio histórico por producto como baseline simple.",
    "hgb_poisson": "HistGradientBoosting con pérdida Poisson, útil para conteos no negativos.",
    "hurdle_hgb": "Modelo de dos etapas que separa probabilidad de venta y unidades cuando sí hay venta.",
}
TARGET_COL = "y"


def _load_original_payload(model_path: Path) -> dict[str, Any]:
    return joblib.load(model_path)


def predict_original_two_stage(payload: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    bundle = payload.get("bundle", payload)
    x = df[bundle["feature_cols"]]
    prob = bundle["clf"].predict_proba(x)[:, 1].astype(float)
    mu = bundle["reg"].predict(x).astype(float)
    return np.clip(prob * mu, 0, 20)


class HurdleHGB:
    def __init__(self, feature_cols: list[str], random_state: int = 42) -> None:
        self.feature_cols = feature_cols
        self.imputer = SimpleImputer(strategy="median")
        self.clf = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.06, max_leaf_nodes=31, random_state=random_state)
        self.reg = HistGradientBoostingRegressor(max_iter=160, learning_rate=0.05, max_leaf_nodes=31, random_state=random_state)
        self.positive_mean = 0.0

    def fit(self, df: pd.DataFrame) -> None:
        x = self.imputer.fit_transform(df[self.feature_cols])
        y = clip_target(df[TARGET_COL])
        y_bin = (y > 0).astype(int)
        self.clf.fit(x, y_bin)
        pos = y > 0
        if pos.sum() >= 25:
            self.reg.fit(x[pos], y[pos])
            self.positive_mean = float(np.mean(y[pos]))
        else:
            self.reg = None
            self.positive_mean = float(np.mean(y)) if len(y) else 0.0

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        x = self.imputer.transform(df[self.feature_cols])
        p = self.clf.predict_proba(x)[:, 1].astype(float)
        mu = self.reg.predict(x).astype(float) if self.reg is not None else np.full(len(df), self.positive_mean, dtype=float)
        return np.clip(p * mu, 0, 20)


class PoissonHGB:
    def __init__(self, feature_cols: list[str], random_state: int = 42) -> None:
        self.feature_cols = feature_cols
        self.model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            HistGradientBoostingRegressor(loss="poisson", max_iter=160, learning_rate=0.05, max_leaf_nodes=31, random_state=random_state),
        )

    def fit(self, df: pd.DataFrame) -> None:
        self.model.fit(df[self.feature_cols], clip_target(df[TARGET_COL]))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model.predict(df[self.feature_cols]), 0, 20)


def _feature_cols(train_df: pd.DataFrame, payload: dict[str, Any]) -> list[str]:
    bundle = payload.get("bundle", payload)
    return list(bundle.get("feature_cols", [c for c in train_df.columns if c != TARGET_COL]))


def _build_bucket_curve(df: pd.DataFrame, prediction_col: str, model_id: str, model_rank: int) -> pd.DataFrame:
    temp = df.copy()
    temp["demand_bucket"] = pd.cut(temp[TARGET_COL], bins=[-0.1, 0.0, 1.0, 3.0, 7.0, 20.0], include_lowest=True).astype(str)
    curve = temp.groupby("demand_bucket", as_index=False, observed=False).agg(real_mean=(TARGET_COL, "mean"), pred_mean=(prediction_col, "mean"), n=(TARGET_COL, "size"))
    rows = []
    for _, row in curve.iterrows():
        rows.append({"model_id": model_id, "model_rank": model_rank, "series": "real_mean", "demand_bucket": row["demand_bucket"], "mean_value": row["real_mean"], "n": row["n"]})
        rows.append({"model_id": model_id, "model_rank": model_rank, "series": "pred_mean", "demand_bucket": row["demand_bucket"], "mean_value": row["pred_mean"], "n": row["n"]})
    return pd.DataFrame(rows)


def compare_and_publish(
    train_path: Path,
    valid_path: Path,
    test_features_path: Path,
    test_pairs_path: Path,
    model_path: Path,
    output_root: Path,
    max_train_rows: int | None = None,
    min_improvement_over_incumbent: float = 0.0,
) -> pd.DataFrame:
    train_df = pd.read_parquet(train_path)
    valid_df = pd.read_parquet(valid_path)
    test_features = pd.read_parquet(test_features_path) if test_features_path.exists() else pd.DataFrame()
    test_pairs = pd.read_parquet(test_pairs_path) if test_pairs_path.exists() else pd.DataFrame()
    payload = _load_original_payload(model_path)
    feature_cols = _feature_cols(train_df, payload)

    if max_train_rows and len(train_df) > max_train_rows:
        train_df = train_df.sample(max_train_rows, random_state=42)

    results: list[ModelRunResult] = []
    pred_cols: dict[str, np.ndarray] = {}

    incumbent_pred = predict_original_two_stage(payload, valid_df)
    incumbent_metrics = compute_metrics(valid_df[TARGET_COL], incumbent_pred)
    results.append(ModelRunResult(model_id="incumbent_two_stage", model_name="LightGBM original dos etapas", model_family="incumbent", model_description=MODEL_DESCRIPTIONS["incumbent_two_stage"], n_valid=len(valid_df), model_scope="original_two_stage", **incumbent_metrics))
    pred_cols["incumbent_two_stage"] = incumbent_pred

    if "cnt_lag_1" in valid_df.columns:
        naive_pred = clip_target(valid_df["cnt_lag_1"])
    else:
        naive_pred = np.zeros(len(valid_df), dtype=float)
    naive_metrics = compute_metrics(valid_df[TARGET_COL], naive_pred)
    results.append(ModelRunResult(model_id="naive_lag1", model_name="Naive último periodo", model_family="baseline", model_description=MODEL_DESCRIPTIONS["naive_lag1"], n_valid=len(valid_df), model_scope="baseline", **naive_metrics))
    pred_cols["naive_lag1"] = naive_pred

    item_mean = train_df.groupby("item_id", as_index=False)[TARGET_COL].mean().rename(columns={TARGET_COL: "item_mean"}) if "item_id" in train_df.columns else pd.DataFrame()
    if not item_mean.empty and "item_id" in valid_df.columns:
        merged = valid_df[["item_id"]].merge(item_mean, on="item_id", how="left")
        item_mean_pred = clip_target(merged["item_mean"].fillna(train_df[TARGET_COL].mean()))
        item_mean_metrics = compute_metrics(valid_df[TARGET_COL], item_mean_pred)
        results.append(ModelRunResult(model_id="item_mean_fallback", model_name="Promedio histórico por producto", model_family="baseline", model_description=MODEL_DESCRIPTIONS["item_mean_fallback"], n_valid=len(valid_df), model_scope="baseline", **item_mean_metrics))
        pred_cols["item_mean_fallback"] = item_mean_pred

    hurdle = HurdleHGB(feature_cols)
    hurdle.fit(train_df)
    hurdle_pred = hurdle.predict(valid_df)
    hurdle_metrics = compute_metrics(valid_df[TARGET_COL], hurdle_pred)
    results.append(ModelRunResult(model_id="hurdle_hgb", model_name="Hurdle HGB dos etapas", model_family="challenger", model_description=MODEL_DESCRIPTIONS["hurdle_hgb"], n_valid=len(valid_df), model_scope="challenger:hurdle_hgb", **hurdle_metrics))
    pred_cols["hurdle_hgb"] = hurdle_pred

    poisson = PoissonHGB(feature_cols)
    poisson.fit(train_df)
    poisson_pred = poisson.predict(valid_df)
    poisson_metrics = compute_metrics(valid_df[TARGET_COL], poisson_pred)
    results.append(ModelRunResult(model_id="hgb_poisson", model_name="HistGradientBoosting Poisson", model_family="challenger", model_description=MODEL_DESCRIPTIONS["hgb_poisson"], n_valid=len(valid_df), model_scope="challenger:hgb_poisson", **poisson_metrics))
    pred_cols["hgb_poisson"] = poisson_pred

    results = select_champion(results, incumbent_id="incumbent_two_stage", min_improvement=min_improvement_over_incumbent)
    runs = results_df(results)
    champion_id = runs.loc[runs["is_champion"], "model_id"].iloc[0]
    second_place_id = runs.loc[~runs["is_champion"], "model_id"].iloc[0] if (~runs["is_champion"]).any() else champion_id

    latest_root = output_root / "latest"
    registry_root = output_root / "registry"
    for path in [latest_root / "predictions", latest_root / "evaluation", latest_root / "review", registry_root]:
        path.mkdir(parents=True, exist_ok=True)

    # Evaluation detail champion
    eval_detail = valid_df.copy()
    eval_detail["prediction"] = pred_cols[champion_id]
    eval_detail["naive_prediction"] = pred_cols.get("naive_lag1", np.zeros(len(valid_df)))
    eval_detail["second_place_prediction"] = pred_cols.get(second_place_id, eval_detail["prediction"].values)
    eval_detail["model_id"] = champion_id
    eval_detail["second_place_model_id"] = second_place_id
    eval_detail["abs_error"] = np.abs(eval_detail["prediction"] - clip_target(eval_detail[TARGET_COL]))
    eval_detail.to_parquet(latest_root / "evaluation" / "evaluation_detail.parquet", index=False)

    # Curves for champion + naive + second place
    curves = []
    rank_map = {row.model_id: idx+1 for idx, row in runs.iterrows()}
    for model_id in [champion_id, "naive_lag1", second_place_id]:
        if model_id not in pred_cols:
            continue
        temp = valid_df.copy()
        temp[model_id] = pred_cols[model_id]
        curves.append(_build_bucket_curve(temp, model_id, model_id, rank_map.get(model_id, 99)))
    curves_df = pd.concat(curves, ignore_index=True).drop_duplicates(["model_id", "series", "demand_bucket"])
    curves_df.to_parquet(latest_root / "evaluation" / "evaluation_curves_by_model.parquet", index=False)

    # KPI by item and by category id
    eval_for_kpi = valid_df.copy()
    eval_for_kpi["prediction"] = pred_cols[champion_id]
    eval_for_kpi["abs_error"] = np.abs(eval_for_kpi["prediction"] - clip_target(eval_for_kpi[TARGET_COL]))
    eval_for_kpi["squared_error"] = (eval_for_kpi["prediction"] - clip_target(eval_for_kpi[TARGET_COL])) ** 2
    if "item_id" in eval_for_kpi.columns:
        by_item = eval_for_kpi.groupby("item_id", as_index=False).agg(n=(TARGET_COL, "size"), y_mean=(TARGET_COL, "mean"), pred_mean=("prediction", "mean"), mae=("abs_error", "mean"), rmse=("squared_error", lambda s: float(np.sqrt(np.mean(s)))))
        by_item = by_item.sort_values("rmse", ascending=False)
        by_item.to_parquet(latest_root / "evaluation" / "evaluation_by_item.parquet", index=False)
    if "item_category_id" in eval_for_kpi.columns:
        by_segment = eval_for_kpi.groupby("item_category_id", as_index=False).agg(n=(TARGET_COL, "size"), y_mean=(TARGET_COL, "mean"), pred_mean=("prediction", "mean"), mae=("abs_error", "mean"), rmse=("squared_error", lambda s: float(np.sqrt(np.mean(s)))))
        by_segment = by_segment.sort_values("rmse", ascending=False)
        by_segment.to_parquet(latest_root / "evaluation" / "evaluation_by_segment.parquet", index=False)

    # Review suggestions and overestimated table
    review = eval_detail.copy()
    review["reason"] = np.where(review["prediction"] > review[TARGET_COL], "Sobreestimado", "Subestimado")
    review.sort_values("abs_error", ascending=False).head(100).to_parquet(latest_root / "review" / "review_suggestions.parquet", index=False)
    over = review.assign(over_error=review["prediction"] - review[TARGET_COL]).query("over_error > 0").sort_values("over_error", ascending=False).head(30)
    over.to_parquet(latest_root / "review" / "overestimated_products.parquet", index=False)

    # Forecast detail for future pairs uses champion if incumbent or incumbent fallback to avoid code duplication risk
    if not test_features.empty and not test_pairs.empty:
        if champion_id == "incumbent_two_stage":
            future_pred = predict_original_two_stage(payload, test_features)
        elif champion_id == "hurdle_hgb":
            future_pred = hurdle.predict(test_features)
        elif champion_id == "hgb_poisson":
            future_pred = poisson.predict(test_features)
        else:
            future_pred = predict_original_two_stage(payload, test_features)
        forecast = test_pairs.copy()
        forecast["prediction"] = future_pred
        forecast["model_id"] = champion_id
        forecast["model_scope"] = "original_two_stage" if champion_id == "incumbent_two_stage" else f"challenger:{champion_id}"
        forecast.to_parquet(latest_root / "predictions" / "forecast_detail.parquet", index=False)
        # summaries
        if "item_category_id" in forecast.columns:
            forecast.groupby("item_category_id", as_index=False).agg(n=("prediction", "size"), total_prediction=("prediction", "sum"), mean_prediction=("prediction", "mean")).to_parquet(latest_root / "predictions" / "forecast_summary_by_category.parquet", index=False)
        group_cols=[c for c in ["shop_id", "segment_key", "item_category_id", "model_scope"] if c in forecast.columns]
        if group_cols:
            forecast.groupby(group_cols, as_index=False).agg(n=("prediction", "size"), total_prediction=("prediction", "sum"), mean_prediction=("prediction", "mean")).to_parquet(latest_root / "predictions" / "forecast_summary_by_shop_segment.parquet", index=False)

    champion_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_policy": "primary=rmse; tie_breaker=mae; original design kept as incumbent candidate",
        "champion": runs.loc[runs["is_champion"]].iloc[0].to_dict(),
        "second_place": runs.loc[~runs["is_champion"]].head(1).to_dict(orient="records")[0] if (~runs["is_champion"]).any() else {},
        "naive": runs.loc[runs["model_id"] == "naive_lag1"].head(1).to_dict(orient="records")[0] if (runs["model_id"] == "naive_lag1").any() else {},
        "all_models": runs.to_dict(orient="records"),
    }
    (registry_root / "champion.json").write_text(json.dumps(champion_payload, indent=2, default=str), encoding="utf-8")
    runs.to_csv(registry_root / "model_runs.csv", index=False)
    (latest_root / "evaluation" / "model_metrics.json").write_text(json.dumps(champion_payload, indent=2, default=str), encoding="utf-8")
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/prep/train.parquet")
    parser.add_argument("--valid-path", default="data/prep/valid.parquet")
    parser.add_argument("--test-features-path", default="data/prep/test_features.parquet")
    parser.add_argument("--test-pairs-path", default="data/prep/test_pairs.parquet")
    parser.add_argument("--model-path", default="artifacts/model.joblib")
    parser.add_argument("--output-root", default="modelops_outputs")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--min-improvement-over-incumbent", type=float, default=0.0)
    args = parser.parse_args()
    runs = compare_and_publish(
        train_path=Path(args.train_path),
        valid_path=Path(args.valid_path),
        test_features_path=Path(args.test_features_path),
        test_pairs_path=Path(args.test_pairs_path),
        model_path=Path(args.model_path),
        output_root=Path(args.output_root),
        max_train_rows=args.max_train_rows,
        min_improvement_over_incumbent=args.min_improvement_over_incumbent,
    )
    print(runs.to_string(index=False))


if __name__ == "__main__":
    main()
