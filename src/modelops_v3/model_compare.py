"""Compare the original LightGBM design against challenger models.

This module intentionally preserves the original modeling design from the repo:
classifier P(sale) + regressor E(units | sale), target column `y`, and the
calibration exponent alpha=0.90 used in `src/training/train.py`.

The goal is not to force LightGBM as champion. The goal is to include it as the
incumbent model and compare all candidates under the same validation split. The
selection metric is RMSE, then MAE as tie breaker.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from src.modelops_v3.catalogs import attach_labels
from src.modelops_v3.metrics import build_metrics, clip_sales

LOGGER = logging.getLogger(__name__)
TARGET_COL = "y"
TARGET_MIN = 0.0
TARGET_MAX = 20.0
DEFAULT_ALPHA = 0.90


class Candidate(Protocol):
    model_id: str
    model_name: str
    model_family: str

    def fit(self, train: pd.DataFrame, feature_cols: list[str]) -> None: ...

    def predict(self, data: pd.DataFrame, feature_cols: list[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class DataPaths:
    data_dir: Path
    artifacts_dir: Path
    output_dir: Path


class OriginalLightGBMTwoStage:
    """Incumbent model loaded from artifacts/model.joblib."""

    model_id = "original_lightgbm_two_stage"
    model_name = "LightGBM original dos etapas"
    model_family = "incumbent"

    def __init__(self, model_path: Path, alpha: float = DEFAULT_ALPHA) -> None:
        self.model_path = model_path
        self.alpha = alpha
        self.payload: dict | None = None

    def fit(self, train: pd.DataFrame, feature_cols: list[str]) -> None:  # noqa: ARG002
        self.payload = joblib.load(self.model_path)

    def predict(self, data: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:  # noqa: ARG002
        if self.payload is None:
            raise RuntimeError("Original model was not loaded. Call fit() first.")
        bundle = self.payload["bundle"]
        model_feature_cols = bundle["feature_cols"]
        x = data[model_feature_cols]
        prob = bundle["clf"].predict_proba(x)[:, 1].astype(np.float32)
        mu = bundle["reg"].predict(x).astype(np.float32)
        return np.clip((prob**self.alpha) * mu, TARGET_MIN, TARGET_MAX)


class NaiveLag1:
    model_id = "naive_lag1"
    model_name = "Naive último periodo"
    model_family = "baseline"

    def fit(self, train: pd.DataFrame, feature_cols: list[str]) -> None:  # noqa: ARG002
        self.default_ = float(train[TARGET_COL].mean())

    def predict(self, data: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:  # noqa: ARG002
        for col in ["cnt_lag_1", "lag_1", "item_cnt_month_lag_1"]:
            if col in data.columns:
                return clip_sales(data[col])
        return np.repeat(self.default_, len(data))


class ItemMeanFallback:
    model_id = "item_mean_fallback"
    model_name = "Promedio histórico por producto"
    model_family = "fallback_rule"

    def fit(self, train: pd.DataFrame, feature_cols: list[str]) -> None:  # noqa: ARG002
        self.global_mean_ = float(train[TARGET_COL].mean())
        self.item_mean_ = train.groupby("item_id")[TARGET_COL].mean().to_dict() if "item_id" in train.columns else {}

    def predict(self, data: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:  # noqa: ARG002
        if "item_id" not in data.columns:
            return np.repeat(self.global_mean_, len(data))
        pred = data["item_id"].map(self.item_mean_).fillna(self.global_mean_)
        return clip_sales(pred)


class HGBPoisson:
    model_id = "hgb_poisson"
    model_name = "HistGradientBoosting Poisson"
    model_family = "boosting_count"

    def __init__(self, random_state: int = 42) -> None:
        self.model = HistGradientBoostingRegressor(
            loss="poisson",
            max_iter=140,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=random_state,
            early_stopping=True,
        )

    def fit(self, train: pd.DataFrame, feature_cols: list[str]) -> None:
        self.model.fit(_matrix(train, feature_cols), clip_sales(train[TARGET_COL]))

    def predict(self, data: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        return clip_sales(self.model.predict(_matrix(data, feature_cols)))


class HurdleHGB:
    model_id = "hurdle_hgb"
    model_name = "Hurdle HGB dos etapas"
    model_family = "hurdle"

    def __init__(self, random_state: int = 42) -> None:
        self.clf = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=random_state,
            early_stopping=True,
        )
        self.reg = HistGradientBoostingRegressor(
            loss="poisson",
            max_iter=140,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=random_state,
            early_stopping=True,
        )
        self.can_fit_regression = True

    def fit(self, train: pd.DataFrame, feature_cols: list[str]) -> None:
        x = _matrix(train, feature_cols)
        y = clip_sales(train[TARGET_COL])
        y_bin = (y > 0).astype(int)
        self.clf.fit(x, y_bin)
        mask = y > 0
        self.can_fit_regression = bool(mask.sum() >= 50)
        if self.can_fit_regression:
            self.reg.fit(x.loc[mask], y[mask])
        else:
            self.positive_mean_ = float(y[mask].mean()) if mask.sum() else 0.0

    def predict(self, data: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        x = _matrix(data, feature_cols)
        prob = self.clf.predict_proba(x)[:, 1]
        if self.can_fit_regression:
            mu = self.reg.predict(x)
        else:
            mu = np.repeat(self.positive_mean_, len(x))
        return clip_sales(prob * mu)


def _matrix(data: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    x = data.reindex(columns=feature_cols).copy()
    for col in x.columns:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    return x.fillna(0)


def load_meta_feature_cols(data_dir: Path, model_path: Path) -> list[str]:
    meta_path = data_dir / "prep" / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))["feature_cols"]
    payload = joblib.load(model_path)
    return list(payload["bundle"]["feature_cols"])


def fit_and_evaluate(
    candidate: Candidate,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_cols: list[str],
    y_naive: np.ndarray,
) -> tuple[dict, np.ndarray]:
    LOGGER.info("action=train_candidate status=started model_id=%s", candidate.model_id)
    candidate.fit(train, feature_cols)
    pred = candidate.predict(valid, feature_cols)
    metrics = build_metrics(
        candidate.model_id,
        candidate.model_name,
        candidate.model_family,
        clip_sales(valid[TARGET_COL]),
        pred,
        y_naive,
    )
    LOGGER.info(
        "action=train_candidate status=success model_id=%s mae=%.6f rmse=%.6f",
        candidate.model_id,
        metrics["mae"],
        metrics["rmse"],
    )
    return metrics, pred


def make_candidates(model_path: Path) -> list[Candidate]:
    return [
        OriginalLightGBMTwoStage(model_path=model_path),
        NaiveLag1(),
        ItemMeanFallback(),
        HGBPoisson(),
        HurdleHGB(),
    ]


def build_evaluation_outputs(
    valid: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    champion_id: str,
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    detail = valid.copy()
    detail["prediction"] = predictions[champion_id]
    detail["naive_prediction"] = predictions["naive_lag1"]
    detail["error"] = detail["prediction"] - detail[TARGET_COL]
    detail["abs_error"] = detail["error"].abs()
    detail["model_id"] = champion_id
    detail = attach_labels(detail, data_dir)

    segment_cols = [col for col in ["segment_key", "segment_name", "item_category_id", "item_category_name", "category_group"] if col in detail.columns]
    if not segment_cols:
        segment_cols = ["item_category_id"] if "item_category_id" in detail.columns else []
    by_segment = _group_eval(detail, segment_cols) if segment_cols else pd.DataFrame()
    by_item = _group_eval(
        detail,
        [col for col in ["item_id", "item_name", "item_category_id", "item_category_name", "category_group"] if col in detail.columns],
    )
    return detail, by_segment, by_item


def _group_eval(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = (
        df.groupby(cols, dropna=False)
        .agg(
            n=(TARGET_COL, "size"),
            y_mean=(TARGET_COL, "mean"),
            pred_mean=("prediction", "mean"),
            mae=("abs_error", "mean"),
            rmse=("error", lambda s: float(np.sqrt(np.mean(np.square(s))))),
        )
        .reset_index()
        .sort_values("mae", ascending=False)
    )
    return out


def build_forecast(
    test_features: pd.DataFrame,
    test_pairs: pd.DataFrame,
    champion: Candidate,
    feature_cols: list[str],
    champion_id: str,
    data_dir: Path,
) -> pd.DataFrame:
    pred = champion.predict(test_features, feature_cols)
    forecast = test_pairs.copy()
    forecast["prediction"] = pred
    forecast["model_id"] = champion_id
    forecast["model_scope"] = forecast.get("model_scope", "global")
    return attach_labels(forecast, data_dir)


def write_outputs(
    paths: DataPaths,
    metrics_df: pd.DataFrame,
    champion_id: str,
    forecast: pd.DataFrame,
    eval_detail: pd.DataFrame,
    eval_segment: pd.DataFrame,
    eval_item: pd.DataFrame,
) -> None:
    out = paths.output_dir
    for rel in ["predictions", "evaluation", "registry", "review"]:
        (out / rel).mkdir(parents=True, exist_ok=True)

    champion_row = metrics_df.loc[metrics_df["model_id"] == champion_id].iloc[0].to_dict()
    now = datetime.now(timezone.utc).isoformat()
    metrics_df = metrics_df.copy()
    metrics_df["created_at"] = now
    metrics_df["is_champion"] = metrics_df["model_id"].eq(champion_id)
    metrics_df["status"] = np.where(metrics_df["is_champion"], "champion", "challenger")

    forecast.to_parquet(out / "predictions" / "forecast_detail.parquet", index=False)
    forecast.to_csv(out / "predictions" / "forecast_detail.csv", index=False)
    eval_detail.to_parquet(out / "evaluation" / "evaluation_detail.parquet", index=False)
    eval_segment.to_parquet(out / "evaluation" / "evaluation_by_segment.parquet", index=False)
    eval_item.to_parquet(out / "evaluation" / "evaluation_by_item.parquet", index=False)
    metrics_df.to_csv(out / "registry" / "model_runs.csv", index=False)
    metrics_df.to_csv(out / "evaluation" / "model_zoo_summary.csv", index=False)

    model_metrics = {
        "created_at": now,
        "target_column": TARGET_COL,
        "selection_policy": "primary=rmse; tie_breaker=mae; original LightGBM is included as incumbent candidate",
        "champion": champion_row,
        "global": champion_row,
        "naive": metrics_df.loc[metrics_df["model_id"] == "naive_lag1"].iloc[0].to_dict()
        if "naive_lag1" in set(metrics_df["model_id"])
        else {},
        "all_models": metrics_df.to_dict(orient="records"),
    }
    (out / "evaluation" / "model_metrics.json").write_text(
        json.dumps(model_metrics, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    champion_payload = {
        "model_run_id": "local-modelops-v3",
        "model_id": champion_id,
        "model_name": champion_row.get("model_name"),
        "created_at": now,
        "status": "champion",
        "is_champion": True,
        "selection_metric": "rmse",
        "promotion_reason": "selected_by_lowest_rmse_with_mae_tie_breaker",
        **champion_row,
    }
    (out / "registry" / "champion.json").write_text(
        json.dumps(champion_payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    suggestions = eval_detail.sort_values("abs_error", ascending=False).head(250).copy()
    suggestions["reason"] = np.where(
        suggestions["prediction"] > suggestions[TARGET_COL],
        "Posible sobrepronóstico",
        "Posible subpronóstico",
    )
    review_cols = [
        col
        for col in [
            "shop_id",
            "shop_name",
            "item_id",
            "item_name",
            "item_category_name",
            TARGET_COL,
            "prediction",
            "abs_error",
            "reason",
        ]
        if col in suggestions.columns
    ]
    suggestions[review_cols].to_parquet(out / "review" / "review_suggestions.parquet", index=False)
    suggestions[review_cols].to_csv(out / "review" / "review_suggestions.csv", index=False)


def run_model_comparison(paths: DataPaths, max_train_rows: int | None = 300_000) -> dict:
    model_path = paths.artifacts_dir / "model.joblib"
    prep_dir = paths.data_dir / "prep"
    train = pd.read_parquet(prep_dir / "train.parquet")
    valid = pd.read_parquet(prep_dir / "valid.parquet")
    test_features = pd.read_parquet(prep_dir / "test_features.parquet")
    test_pairs = pd.read_parquet(prep_dir / "test_pairs.parquet")
    feature_cols = load_meta_feature_cols(paths.data_dir, model_path)

    if max_train_rows and len(train) > max_train_rows:
        train = train.sample(max_train_rows, random_state=42).copy()

    naive = NaiveLag1()
    naive.fit(train, feature_cols)
    y_naive = naive.predict(valid, feature_cols)

    candidates = make_candidates(model_path)
    metrics: list[dict] = []
    predictions: dict[str, np.ndarray] = {}
    fitted_candidates: dict[str, Candidate] = {}
    for candidate in candidates:
        row, pred = fit_and_evaluate(candidate, train, valid, feature_cols, y_naive)
        metrics.append(row)
        predictions[candidate.model_id] = pred
        fitted_candidates[candidate.model_id] = candidate

    metrics_df = pd.DataFrame(metrics).sort_values(["rmse", "mae"], ascending=True).reset_index(drop=True)
    champion_id = str(metrics_df.iloc[0]["model_id"])
    champion = fitted_candidates[champion_id]
    LOGGER.info("action=model_selection status=success champion_id=%s", champion_id)

    forecast = build_forecast(test_features, test_pairs, champion, feature_cols, champion_id, paths.data_dir)
    eval_detail, eval_segment, eval_item = build_evaluation_outputs(
        valid,
        predictions,
        champion_id,
        paths.data_dir,
    )
    write_outputs(paths, metrics_df, champion_id, forecast, eval_detail, eval_segment, eval_item)
    return {"champion_id": champion_id, "metrics": metrics_df.to_dict(orient="records")}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output-dir", type=Path, default=Path("modelops_outputs"))
    parser.add_argument("--max-train-rows", type=int, default=300_000)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = build_parser().parse_args()
    result = run_model_comparison(
        DataPaths(args.data_dir, args.artifacts_dir, args.output_dir),
        max_train_rows=args.max_train_rows,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
