"""Sparse-sales model zoo with clean dimensions and friendly output tables.

This module trains several candidates for a retail demand distribution dominated
by zeros and very small counts.  It also fixes the dashboard usability issues:
shop names are joined everywhere, category/item metadata are cleaned, and
sentinel recency values such as 99 are converted into explicit labels.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor, TweedieRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modelops.data_quality_dimensions import (
    build_item_dimension,
    build_shop_dimension,
    clean_feature_frame_for_modeling,
    enrich_with_dimensions,
)

LOGGER = logging.getLogger(__name__)
TARGET_COL = "y"
MAX_PREDICTION = 20.0
EPS = 1e-9

METADATA_EXCLUDES = {
    TARGET_COL,
    "ID",
    "date",
    "item_name",
    "shop_name",
    "shop_label",
    "item_label",
    "category_label",
    "item_category_name",
    "segment_name",
    "category_group",
    "price_tier",
    "demand_tier",
    "recency_label",
    "model_id",
    "model_family",
    "prediction",
    "naive_prediction",
    "error",
    "abs_error",
}

NAIVE_CANDIDATE_COLUMNS = [
    "cnt_lag_1",
    "item_cnt_month_lag_1",
    "lag_1",
    "target_lag_1",
    "sales_lag_1",
]


@dataclass(frozen=True)
class CandidateSpec:
    """A candidate model definition for the registry."""

    model_id: str
    model_family: str
    description: str


class NaiveLagModel:
    """Baseline model using the best available lag column."""

    def __init__(self) -> None:
        self.lag_column_: str | None = None
        self.fallback_: float = 0.0

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "NaiveLagModel":
        self.lag_column_ = find_naive_column(x)
        self.fallback_ = float(y.mean()) if len(y) else 0.0
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.lag_column_ and self.lag_column_ in x.columns:
            values = pd.to_numeric(x[self.lag_column_], errors="coerce").fillna(self.fallback_)
            return clip(values.to_numpy(dtype=float))
        return clip(np.full(len(x), self.fallback_, dtype=float))


class HurdleHGBModel:
    """Two-stage model: P(sale > 0) × E(sale | sale > 0)."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.feature_cols_: list[str] = []
        self.classifier_: HistGradientBoostingClassifier | None = None
        self.regressor_: HistGradientBoostingRegressor | None = None
        self.positive_fallback_: float = 0.0

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "HurdleHGBModel":
        self.feature_cols_ = list(x.columns)
        y_values = y.to_numpy(dtype=float)
        y_binary = (y_values > 0).astype(int)

        self.classifier_ = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            random_state=self.random_state,
        )
        self.classifier_.fit(x, y_binary)

        positive_mask = y_values > 0
        self.positive_fallback_ = float(y_values[positive_mask].mean()) if positive_mask.any() else 0.0
        if int(positive_mask.sum()) >= 100:
            self.regressor_ = HistGradientBoostingRegressor(
                loss="poisson",
                max_iter=160,
                learning_rate=0.06,
                max_leaf_nodes=31,
                min_samples_leaf=40,
                random_state=self.random_state,
            )
            self.regressor_.fit(x.loc[positive_mask], y_values[positive_mask])
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.classifier_ is None:
            raise RuntimeError("Hurdle model has not been fitted")
        x = x[self.feature_cols_]
        proba = self.classifier_.predict_proba(x)[:, 1]
        if self.regressor_ is None:
            positive = np.full(len(x), self.positive_fallback_, dtype=float)
        else:
            positive = self.regressor_.predict(x)
        return clip(proba * positive)


class SegmentedHurdleModel:
    """Train hurdle models for large segments with global fallback."""

    def __init__(
        self,
        segment_column: str = "segment_key",
        max_segments: int = 12,
        min_rows: int = 15_000,
        min_positive_rows: int = 300,
        random_state: int = 42,
    ) -> None:
        self.segment_column = segment_column
        self.max_segments = max_segments
        self.min_rows = min_rows
        self.min_positive_rows = min_positive_rows
        self.random_state = random_state
        self.feature_cols_: list[str] = []
        self.global_model_: HurdleHGBModel | None = None
        self.segment_models_: dict[str, HurdleHGBModel] = {}

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "SegmentedHurdleModel":
        if self.segment_column not in x.columns:
            self.segment_column = "__all__"
            x = x.copy()
            x[self.segment_column] = "all"

        self.feature_cols_ = [col for col in x.columns if col != self.segment_column]
        self.global_model_ = HurdleHGBModel(random_state=self.random_state).fit(x[self.feature_cols_], y)

        stats = (
            pd.DataFrame({"segment": x[self.segment_column].astype(str), "y": y.to_numpy(dtype=float)})
            .groupby("segment", as_index=False)
            .agg(n=("y", "size"), positive_n=("y", lambda values: int((values > 0).sum())))
            .sort_values(["n", "positive_n"], ascending=False)
        )
        eligible = stats.query(
            "n >= @self.min_rows and positive_n >= @self.min_positive_rows"
        ).head(self.max_segments)

        for segment in eligible["segment"].tolist():
            mask = x[self.segment_column].astype(str) == segment
            model = HurdleHGBModel(random_state=self.random_state).fit(x.loc[mask, self.feature_cols_], y.loc[mask])
            self.segment_models_[segment] = model
            LOGGER.info("action=train_segment_model status=success segment=%s rows=%s", segment, int(mask.sum()))
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.global_model_ is None:
            raise RuntimeError("Segmented model has not been fitted")
        if self.segment_column not in x.columns:
            return self.global_model_.predict(x[self.feature_cols_])
        predictions = self.global_model_.predict(x[self.feature_cols_])
        segments = x[self.segment_column].astype(str)
        for segment, model in self.segment_models_.items():
            mask = segments == segment
            if mask.any():
                predictions[mask.to_numpy()] = model.predict(x.loc[mask, self.feature_cols_])
        return clip(predictions)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def clip(values: np.ndarray | pd.Series) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=MAX_PREDICTION, neginf=0.0)
    return np.clip(arr, 0.0, MAX_PREDICTION)


def find_naive_column(df: pd.DataFrame) -> str | None:
    for column in NAIVE_CANDIDATE_COLUMNS:
        if column in df.columns:
            return column
    lag_cols = [col for col in df.columns if "lag" in col.lower()]
    return lag_cols[0] if lag_cols else None


def infer_segment_key(df: pd.DataFrame) -> pd.Series:
    if "segment_key" in df.columns:
        return df["segment_key"].astype(str)
    if "item_category_id" in df.columns:
        return "cat_" + pd.to_numeric(df["item_category_id"], errors="coerce").fillna(-1).astype(int).astype(str)
    return pd.Series(["all"] * len(df), index=df.index)


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    feature_cols = []
    for column in df.columns:
        if column in METADATA_EXCLUDES:
            continue
        if pd.api.types.is_numeric_dtype(df[column]) or pd.api.types.is_bool_dtype(df[column]):
            feature_cols.append(column)
    if not feature_cols:
        raise ValueError("No numeric feature columns found for model zoo")
    return feature_cols


def sample_training_data(df: pd.DataFrame, max_rows: int, random_state: int = 42) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df.reset_index(drop=True)
    positive = df[df[TARGET_COL] > 0]
    zero = df[df[TARGET_COL] <= 0]
    positive_quota = min(len(positive), max_rows // 2)
    zero_quota = max_rows - positive_quota
    parts = []
    if positive_quota > 0:
        parts.append(positive.sample(positive_quota, random_state=random_state))
    if zero_quota > 0:
        parts.append(zero.sample(min(len(zero), zero_quota), random_state=random_state))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def make_feature_matrices(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    inference_df: pd.DataFrame,
    max_train_rows: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, list[str]]:
    train_df = clean_feature_frame_for_modeling(train_df)
    valid_df = clean_feature_frame_for_modeling(valid_df)
    inference_df = clean_feature_frame_for_modeling(inference_df)
    train_sample = sample_training_data(train_df, max_train_rows)
    feature_cols = select_feature_columns(train_sample)

    def matrix(df: pd.DataFrame) -> pd.DataFrame:
        x = df.reindex(columns=feature_cols, fill_value=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        for col in x.columns:
            if pd.api.types.is_bool_dtype(x[col]):
                x[col] = x[col].astype(int)
        return x.astype(np.float32)

    x_train = matrix(train_sample)
    y_train = train_sample[TARGET_COL].clip(0, MAX_PREDICTION).astype(float)
    x_valid = matrix(valid_df)
    y_valid = valid_df[TARGET_COL].clip(0, MAX_PREDICTION).astype(float)
    x_inference = matrix(inference_df)

    # Segment is metadata for segmented model, not a numeric model feature.
    x_train["segment_key"] = infer_segment_key(train_sample).to_numpy()
    x_valid["segment_key"] = infer_segment_key(valid_df).to_numpy()
    x_inference["segment_key"] = infer_segment_key(inference_df).to_numpy()
    return x_train, y_train, x_valid, y_valid, x_inference, feature_cols


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    abs_error = np.abs(y_true - y_pred)
    positive_mask = y_true > 0
    return {
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mae": float(np.mean(abs_error)),
        "wape": float(abs_error.sum() / (np.abs(y_true).sum() + EPS)),
        "bias": float(np.mean(y_pred - y_true)),
        "nonzero_recall": float((y_pred[positive_mask] > 0.05).mean()) if positive_mask.any() else float("nan"),
    }


def group_metrics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        y = group["y_true"].to_numpy(dtype=float)
        pred = group["prediction"].to_numpy(dtype=float)
        naive = group["naive_prediction"].to_numpy(dtype=float)
        row = {col: value for col, value in zip(group_cols, keys)}
        row.update({f"model_{k}": v for k, v in metrics(y, pred).items()})
        row.update({f"naive_{k}": v for k, v in metrics(y, naive).items()})
        row["n"] = int(len(group))
        row["beats_naive_wape"] = bool(row["model_wape"] <= row["naive_wape"])
        for meta_col in ["shop_name", "segment_name", "item_category_name", "category_group"]:
            if meta_col in group.columns:
                row[meta_col] = str(group[meta_col].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def build_candidates(include_slow_models: bool = False) -> list[CandidateSpec]:
    candidates = [
        CandidateSpec("naive_lag1", "baseline", "Baseline: último periodo observado"),
        CandidateSpec("poisson_glm", "count_glm", "GLM Poisson para conteos no negativos"),
        CandidateSpec("tweedie_glm_p15", "count_glm", "GLM Tweedie power=1.5 para target sesgado"),
        CandidateSpec("hgb_poisson", "boosting", "HistGradientBoosting con pérdida Poisson"),
        CandidateSpec("hurdle_hgb", "hurdle", "Dos etapas: probabilidad de venta × demanda positiva"),
        CandidateSpec("segmented_hurdle_hgb", "segmented_hurdle", "Hurdle por segmento/categoría con fallback global"),
    ]
    if include_slow_models:
        # Placeholder for future expensive candidates; no external deps for MVP stability.
        pass
    return candidates


def train_model(spec: CandidateSpec, x_train: pd.DataFrame, y_train: pd.Series, feature_cols: list[str]) -> Any:
    numeric_train = x_train[feature_cols]
    if spec.model_id == "naive_lag1":
        return NaiveLagModel().fit(numeric_train, y_train)
    if spec.model_id == "poisson_glm":
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", PoissonRegressor(alpha=0.05, max_iter=300)),
            ]
        )
        return model.fit(numeric_train, y_train)
    if spec.model_id == "tweedie_glm_p15":
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", TweedieRegressor(power=1.5, alpha=0.03, max_iter=300)),
            ]
        )
        return model.fit(numeric_train, y_train)
    if spec.model_id == "hgb_poisson":
        model = HistGradientBoostingRegressor(
            loss="poisson",
            max_iter=180,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            random_state=42,
        )
        return model.fit(numeric_train, y_train)
    if spec.model_id == "hurdle_hgb":
        return HurdleHGBModel(random_state=42).fit(numeric_train, y_train)
    if spec.model_id == "segmented_hurdle_hgb":
        return SegmentedHurdleModel(random_state=42).fit(x_train[feature_cols + ["segment_key"]], y_train)
    raise ValueError(f"Unknown model_id: {spec.model_id}")


def predict_model(model: Any, spec: CandidateSpec, x: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    if spec.model_id == "segmented_hurdle_hgb":
        return clip(model.predict(x[feature_cols + ["segment_key"]]))
    return clip(model.predict(x[feature_cols]))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_dimensions(shops_path: str | None, items_path: str | None, categories_path: str | None) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    shop_dim = None
    item_dim = None
    if shops_path:
        shop_dim = build_shop_dimension(pd.read_csv(shops_path))
    if items_path:
        categories = pd.read_csv(categories_path) if categories_path else None
        item_dim = build_item_dimension(pd.read_csv(items_path), categories)
    return shop_dim, item_dim


def run_model_zoo(
    *,
    train_path: Path,
    valid_path: Path,
    inference_features_path: Path,
    inference_pairs_path: Path,
    output_dir: Path,
    latest_dir: Path | None,
    registry_dir: Path | None,
    run_id: str,
    base_s3_uri: str,
    shops_path: Path | None = None,
    items_path: Path | None = None,
    categories_path: Path | None = None,
    max_train_rows: int = 500_000,
    include_slow_models: bool = False,
) -> dict[str, Any]:
    configure_logging()
    output_dir = ensure_dir(output_dir)
    predictions_dir = ensure_dir(output_dir / "predictions")
    evaluation_dir = ensure_dir(output_dir / "evaluation")
    models_dir = ensure_dir(output_dir / "models")
    model_dir = ensure_dir(output_dir / "model")

    train_df = pd.read_parquet(train_path)
    valid_df = pd.read_parquet(valid_path)
    inference_features = pd.read_parquet(inference_features_path)
    inference_pairs = pd.read_parquet(inference_pairs_path)

    shop_dim, item_dim = load_dimensions(
        str(shops_path) if shops_path else None,
        str(items_path) if items_path else None,
        str(categories_path) if categories_path else None,
    )

    x_train, y_train, x_valid, y_valid, x_infer, feature_cols = make_feature_matrices(
        train_df, valid_df, inference_features, max_train_rows
    )

    # Metadata gets dimensions and cleaned display labels.
    valid_meta = enrich_with_dimensions(valid_df.drop(columns=[TARGET_COL], errors="ignore"), shop_dimension=shop_dim, item_dimension=item_dim)
    infer_meta_source = inference_pairs.copy() if len(inference_pairs) == len(inference_features) else inference_features.copy()
    inference_meta = enrich_with_dimensions(infer_meta_source, shop_dimension=shop_dim, item_dimension=item_dim)

    naive_model = NaiveLagModel().fit(x_train[feature_cols], y_train)
    naive_valid = naive_model.predict(x_valid[feature_cols])
    naive_metrics = metrics(y_valid.to_numpy(dtype=float), naive_valid)

    summary_rows: list[dict[str, Any]] = []
    valid_frames: list[pd.DataFrame] = []
    forecast_frames: list[pd.DataFrame] = []
    created_at = datetime.now(timezone.utc).isoformat()

    for spec in build_candidates(include_slow_models):
        LOGGER.info("action=train_candidate status=started model_id=%s", spec.model_id)
        try:
            model = train_model(spec, x_train, y_train, feature_cols)
            valid_pred = predict_model(model, spec, x_valid, feature_cols)
            infer_pred = predict_model(model, spec, x_infer, feature_cols)
            model_metrics = metrics(y_valid.to_numpy(dtype=float), valid_pred)

            model_artifact_dir = ensure_dir(models_dir / spec.model_id)
            artifact_path = model_artifact_dir / "model_bundle.joblib"
            joblib.dump(
                {
                    "model_id": spec.model_id,
                    "model_family": spec.model_family,
                    "description": spec.description,
                    "model": model,
                    "feature_cols": feature_cols,
                    "created_at": created_at,
                },
                artifact_path,
            )

            valid_out = valid_meta.copy()
            valid_out["model_id"] = spec.model_id
            valid_out["model_family"] = spec.model_family
            valid_out["y_true"] = y_valid.to_numpy(dtype=float)
            valid_out["prediction"] = valid_pred
            valid_out["naive_prediction"] = naive_valid
            valid_out["error"] = valid_out["prediction"] - valid_out["y_true"]
            valid_out["abs_error"] = valid_out["error"].abs()
            valid_frames.append(valid_out)

            forecast_out = inference_meta.copy()
            forecast_out["model_id"] = spec.model_id
            forecast_out["model_family"] = spec.model_family
            forecast_out["prediction"] = infer_pred
            forecast_out.to_parquet(
                ensure_dir(predictions_dir / "by_model" / spec.model_id) / "forecast_detail.parquet",
                index=False,
            )
            forecast_frames.append(forecast_out)

            row = {
                "run_id": run_id,
                "model_id": spec.model_id,
                "model_family": spec.model_family,
                "description": spec.description,
                "status": "candidate",
                "created_at": created_at,
                "n_train_rows": int(len(x_train)),
                "n_valid_rows": int(len(x_valid)),
                "n_features": int(len(feature_cols)),
                "model_uri": f"{base_s3_uri}/model-zoo-runs/{run_id}/models/{spec.model_id}/model_bundle.joblib",
                "forecast_uri": f"{base_s3_uri}/model-zoo-runs/{run_id}/predictions/by_model/{spec.model_id}/forecast_detail.parquet",
                "metrics_uri": f"{base_s3_uri}/model-zoo-runs/{run_id}/evaluation/model_zoo_summary.csv",
                **model_metrics,
                "naive_rmse": naive_metrics["rmse"],
                "naive_mae": naive_metrics["mae"],
                "naive_wape": naive_metrics["wape"],
                "beats_naive_rmse": bool(model_metrics["rmse"] <= naive_metrics["rmse"]),
                "beats_naive_wape": bool(model_metrics["wape"] <= naive_metrics["wape"]),
                "promotion_reason": "pending",
            }
            summary_rows.append(row)
            LOGGER.info(
                "action=train_candidate status=success model_id=%s rmse=%.5f wape=%.5f",
                spec.model_id,
                model_metrics["rmse"],
                model_metrics["wape"],
            )
        except Exception as exc:  # noqa: BLE001 - keep other candidates running.
            LOGGER.exception(
                "action=train_candidate status=failure model_id=%s error_type=%s",
                spec.model_id,
                type(exc).__name__,
            )

    if not summary_rows:
        raise RuntimeError("No model-zoo candidates trained successfully")

    summary_df = pd.DataFrame(summary_rows)
    # Choose champion by WAPE first; require finite metrics and prefer beating naive.
    sortable = summary_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["wape"])
    sortable = sortable.sort_values(["beats_naive_wape", "wape", "rmse"], ascending=[False, True, True])
    champion_id = str(sortable.iloc[0]["model_id"])
    summary_df["status"] = np.where(summary_df["model_id"] == champion_id, "champion", "evaluated")
    summary_df.loc[summary_df["model_id"] == champion_id, "promotion_reason"] = "lowest_wape_with_sparse_demand_model_zoo"
    summary_df.to_csv(evaluation_dir / "model_zoo_summary.csv", index=False)

    all_valid = pd.concat(valid_frames, ignore_index=True)
    all_valid.to_parquet(evaluation_dir / "model_zoo_valid_predictions.parquet", index=False)
    by_model_segment = group_metrics(all_valid, ["model_id", "segment_key"])
    by_model_segment.to_parquet(evaluation_dir / "evaluation_by_model_segment.parquet", index=False)
    by_model_item = group_metrics(all_valid, ["model_id", "item_id", "segment_key"]) if "item_id" in all_valid.columns else by_model_segment.copy()
    by_model_item.to_parquet(evaluation_dir / "evaluation_by_model_item.parquet", index=False)

    champion_valid = all_valid.query("model_id == @champion_id").copy()
    champion_forecast = next(frame for frame in forecast_frames if str(frame["model_id"].iloc[0]) == champion_id).copy()
    champion_segment = by_model_segment.query("model_id == @champion_id").copy()
    champion_item = by_model_item.query("model_id == @champion_id").copy()

    champion_forecast.to_parquet(predictions_dir / "forecast_detail.parquet", index=False)
    group_col = "category_label" if "category_label" in champion_forecast.columns else "segment_key"
    champion_forecast.groupby(group_col, as_index=False).agg(
        total_prediction=("prediction", "sum"),
        mean_prediction=("prediction", "mean"),
        n=("prediction", "size"),
    ).to_parquet(predictions_dir / "forecast_summary_by_category.parquet", index=False)

    shop_group_cols = [col for col in ["shop_id", "shop_name", group_col] if col in champion_forecast.columns]
    champion_forecast.groupby(shop_group_cols, as_index=False).agg(
        total_prediction=("prediction", "sum"),
        mean_prediction=("prediction", "mean"),
        n=("prediction", "size"),
    ).to_parquet(predictions_dir / "forecast_summary_by_shop_segment.parquet", index=False)

    submission = pd.DataFrame()
    submission["ID"] = champion_forecast["ID"].astype(int) if "ID" in champion_forecast.columns else np.arange(len(champion_forecast))
    submission["item_cnt_month"] = champion_forecast["prediction"].clip(0, MAX_PREDICTION)
    submission.to_csv(predictions_dir / "submission.csv", index=False)

    champion_valid.to_parquet(evaluation_dir / "evaluation_detail.parquet", index=False)
    champion_segment.to_parquet(evaluation_dir / "evaluation_by_segment.parquet", index=False)
    champion_item.to_parquet(evaluation_dir / "evaluation_by_item.parquet", index=False)
    champion_row = summary_df.query("model_id == @champion_id").iloc[0].to_dict()
    model_metrics = {
        "run_id": run_id,
        "champion_model_id": champion_id,
        "global": champion_row,
        "naive": naive_metrics,
        "model_zoo": summary_df.to_dict(orient="records"),
        "feature_cols": feature_cols,
        "data_quality_notes": {
            "recency_99_treated_as": "Sin ventas recientes / sin historial",
            "missing_metadata_treated_as": "Sin clasificar",
            "shop_names_joined": shop_dim is not None,
            "item_category_names_joined": item_dim is not None,
        },
    }
    write_json(evaluation_dir / "model_metrics.json", model_metrics)
    shutil.copy2(models_dir / champion_id / "model_bundle.joblib", model_dir / "model_bundle.joblib")

    champion_payload = dict(champion_row)
    champion_payload["is_champion"] = True
    if registry_dir:
        registry_dir = ensure_dir(registry_dir)
        write_json(registry_dir / "champion.json", champion_payload)
        summary_df.to_csv(registry_dir / "model_runs.csv", index=False)

    if latest_dir:
        latest_dir = ensure_dir(latest_dir)
        for subdir in ["predictions", "evaluation", "model"]:
            target = latest_dir / subdir
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(output_dir / subdir, target)
        if registry_dir:
            latest_registry = ensure_dir(latest_dir.parent / "registry")
            write_json(latest_registry / "champion.json", champion_payload)
            summary_df.to_csv(latest_registry / "model_runs.csv", index=False)

    return {
        "run_id": run_id,
        "champion_model_id": champion_id,
        "summary": summary_df.to_dict(orient="records"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--valid-path", required=True)
    parser.add_argument("--inference-features-path", required=True)
    parser.add_argument("--inference-pairs-path", required=True)
    parser.add_argument("--shops-path", default=None)
    parser.add_argument("--items-path", default=None)
    parser.add_argument("--categories-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--latest-dir", default=None)
    parser.add_argument("--registry-dir", default=None)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-s3-uri", required=True)
    parser.add_argument("--max-train-rows", type=int, default=500_000)
    parser.add_argument("--include-slow-models", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_model_zoo(
        train_path=Path(args.train_path),
        valid_path=Path(args.valid_path),
        inference_features_path=Path(args.inference_features_path),
        inference_pairs_path=Path(args.inference_pairs_path),
        shops_path=Path(args.shops_path) if args.shops_path else None,
        items_path=Path(args.items_path) if args.items_path else None,
        categories_path=Path(args.categories_path) if args.categories_path else None,
        output_dir=Path(args.output_dir),
        latest_dir=Path(args.latest_dir) if args.latest_dir else None,
        registry_dir=Path(args.registry_dir) if args.registry_dir else None,
        run_id=args.run_id,
        base_s3_uri=args.base_s3_uri.rstrip("/"),
        max_train_rows=args.max_train_rows,
        include_slow_models=args.include_slow_models,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
