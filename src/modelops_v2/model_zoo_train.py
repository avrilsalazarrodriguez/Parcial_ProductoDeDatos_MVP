"""Entrenamiento de catálogo de modelos para el MVP de ventas.

Este script crea un pequeño model zoo pensado para el caso de 1C Company:
- conserva la variable target original ``y`` usada por ``src/training/train.py``;
- evalúa todos los modelos contra el mismo set de validación;
- elige champion con WAPE como métrica principal y naive como baseline;
- genera outputs enriquecidos con nombres de tienda/producto/categoría;
- guarda artefactos en formato estable para que Streamlit los lea desde S3.

Ejemplo local:
    PYTHONPATH=. uv run python src/modelops_v2/model_zoo_train.py \
      --prep-dir data/prep \
      --data-dir data \
      --output-dir modelops_outputs

Ejemplo con upload a S3:
    PYTHONPATH=. uv run python src/modelops_v2/model_zoo_train.py \
      --prep-dir data/prep \
      --data-dir data \
      --output-dir modelops_outputs \
      --s3-bucket "$MODEL_BUCKET" \
      --s3-prefix modelops/latest \
      --registry-prefix modelops/registry
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from src.modelops_v2.catalogs import add_catalog_metadata
from src.modelops_v2.metrics import clip_sales_prediction, evaluate_predictions

try:  # LightGBM es opcional; si no está instalado, el script sigue con sklearn.
    import lightgbm as lgb
except Exception:  # noqa: BLE001
    lgb = None

LOGGER = logging.getLogger("modelops_v2.model_zoo_train")
TARGET_COLUMN = "y"
PREDICTION_MIN = 0.0
PREDICTION_MAX = 20.0
RANDOM_STATE = 42


@dataclass(frozen=True)
class ModelRun:
    """Resultado de una corrida candidata del catálogo."""

    model_id: str
    model_name: str
    model_family: str
    status: str
    rank: int
    is_champion: bool
    selection_metric: str
    promotion_reason: str
    metrics: dict[str, Any]
    created_at: str

    def to_row(self) -> dict[str, Any]:
        """Convierte una corrida a fila de registry."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_family": self.model_family,
            "status": self.status,
            "rank": self.rank,
            "is_champion": self.is_champion,
            "selection_metric": self.selection_metric,
            "promotion_reason": self.promotion_reason,
            "created_at": self.created_at,
            **self.metrics,
        }


class ForecastModel(Protocol):
    """Interfaz mínima de los modelos del catálogo."""

    model_id: str
    model_name: str
    model_family: str

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> "ForecastModel": ...

    def predict(self, x: pd.DataFrame) -> np.ndarray: ...

    def predict_scope(self, x: pd.DataFrame) -> pd.Series: ...


def configure_logging() -> None:
    """Configura logging con mensajes útiles y sin secretos."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s action=%(message)s",
    )


def utc_now() -> str:
    """Timestamp UTC ISO 8601."""
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    """Lee JSON si existe; de lo contrario regresa dict vacío."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_required_parquet(path: Path) -> pd.DataFrame:
    """Carga un parquet obligatorio con error explícito."""
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo requerido: {path}")
    return pd.read_parquet(path)


def infer_feature_columns(train_df: pd.DataFrame, meta: dict[str, Any]) -> list[str]:
    """Obtiene feature_cols desde meta.json o por exclusión del target."""
    if "feature_cols" in meta:
        return list(meta["feature_cols"])

    excluded = {
        TARGET_COLUMN,
        "item_cnt_month",
        "prediction",
        "naive_prediction",
        "error",
        "abs_error",
        "shop_name",
        "item_name",
        "item_category_name",
    }
    return [col for col in train_df.columns if col not in excluded]


def numeric_features(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    """Filtra features que pueden convertirse a numérico."""
    result: list[str] = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            result.append(col)
    if not result:
        raise ValueError("No se encontraron features numéricas para entrenar.")
    return result


def align_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Selecciona features y rellena columnas faltantes."""
    aligned = pd.DataFrame(index=df.index)
    for col in feature_cols:
        if col in df.columns:
            aligned[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            aligned[col] = np.nan
    return aligned


def build_naive_prediction(df: pd.DataFrame) -> np.ndarray:
    """Baseline naive usando el último mes observado.

    El orden de búsqueda contempla nombres comunes de features de rezago.
    """
    candidates = [
        "cnt_lag_1",
        "item_cnt_month_lag_1",
        "lag_1",
        "target_lag_1",
        "item_cnt_month_1",
    ]
    for col in candidates:
        if col in df.columns:
            return clip_sales_prediction(pd.to_numeric(df[col], errors="coerce"))
    return np.zeros(len(df), dtype=float)


class NaiveLagModel:
    """Baseline naive de último mes observado."""

    model_id = "naive_lag1"
    model_name = "Naive lag-1"
    model_family = "baseline"

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> "NaiveLagModel":
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return build_naive_prediction(x)

    def predict_scope(self, x: pd.DataFrame) -> pd.Series:
        return pd.Series(["naive_baseline"] * len(x), index=x.index)


class ItemMeanFallbackModel:
    """Modelo interpretable: promedio histórico por producto y fallback global."""

    model_id = "item_mean_fallback"
    model_name = "Promedio histórico por producto"
    model_family = "statistical"

    def __init__(self) -> None:
        self.item_means_: dict[int, float] = {}
        self.global_mean_: float = 0.0

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> "ItemMeanFallbackModel":
        train = x_train.copy()
        train[TARGET_COLUMN] = y_train.to_numpy()
        self.global_mean_ = float(train[TARGET_COLUMN].mean())
        if "item_id" in train.columns:
            grouped = train.groupby("item_id")[TARGET_COLUMN].mean()
            self.item_means_ = {int(k): float(v) for k, v in grouped.items()}
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if "item_id" not in x.columns:
            pred = np.full(len(x), self.global_mean_)
        else:
            pred = x["item_id"].map(self.item_means_).fillna(self.global_mean_).to_numpy()
        return clip_sales_prediction(pred)

    def predict_scope(self, x: pd.DataFrame) -> pd.Series:
        if "item_id" not in x.columns:
            return pd.Series(["global"] * len(x), index=x.index)
        known = x["item_id"].map(lambda item: int(item) in self.item_means_)
        return pd.Series(
            np.where(known, "item_mean_fallback", "global"),
            index=x.index,
        )


class HurdleHGBModel:
    """Modelo dos etapas con HistGradientBoosting: venta/no venta + unidades."""

    model_id = "hurdle_hgb"
    model_name = "Hurdle HGB global"
    model_family = "sklearn_hgb"

    def __init__(self, feature_cols: list[str]) -> None:
        self.feature_cols = feature_cols
        self.classifier = make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=120,
                learning_rate=0.06,
                max_leaf_nodes=31,
                l2_regularization=0.05,
                random_state=RANDOM_STATE,
            ),
        )
        self.regressor = make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                max_iter=160,
                learning_rate=0.05,
                max_leaf_nodes=31,
                l2_regularization=0.05,
                random_state=RANDOM_STATE,
            ),
        )
        self.mean_positive_: float = 0.0

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> "HurdleHGBModel":
        x = align_features(x_train, self.feature_cols)
        y = y_train.astype(float)
        y_bin = (y > 0).astype(int)
        self.classifier.fit(x, y_bin)
        positive_mask = y > 0
        self.mean_positive_ = float(y.loc[positive_mask].mean()) if positive_mask.any() else 0.0
        if positive_mask.sum() >= 10:
            self.regressor.fit(x.loc[positive_mask], y.loc[positive_mask])
        else:
            self.regressor = None  # type: ignore[assignment]
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        features = align_features(x, self.feature_cols)
        prob = self.classifier.predict_proba(features)[:, 1]
        if self.regressor is None:
            mu = np.full(len(features), self.mean_positive_)
        else:
            mu = self.regressor.predict(features)
        return clip_sales_prediction((prob**0.90) * np.maximum(mu, 0.0))

    def predict_scope(self, x: pd.DataFrame) -> pd.Series:
        return pd.Series(["global"] * len(x), index=x.index)


class PoissonHGBModel:
    """Regresor HGB con loss Poisson para demanda no negativa."""

    model_id = "poisson_hgb"
    model_name = "HGB Poisson global"
    model_family = "sklearn_hgb"

    def __init__(self, feature_cols: list[str]) -> None:
        self.feature_cols = feature_cols
        self.regressor = make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                loss="poisson",
                max_iter=180,
                learning_rate=0.045,
                max_leaf_nodes=31,
                l2_regularization=0.08,
                random_state=RANDOM_STATE,
            ),
        )

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> "PoissonHGBModel":
        x = align_features(x_train, self.feature_cols)
        y = y_train.astype(float).clip(lower=0)
        self.regressor.fit(x, y)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        features = align_features(x, self.feature_cols)
        return clip_sales_prediction(self.regressor.predict(features))

    def predict_scope(self, x: pd.DataFrame) -> pd.Series:
        return pd.Series(["global"] * len(x), index=x.index)


class LightGBMTwoStageModel:
    """Modelo dos etapas compatible con el entrenamiento original basado en LightGBM."""

    model_id = "lightgbm_two_stage"
    model_name = "LightGBM dos etapas"
    model_family = "lightgbm"

    def __init__(self, feature_cols: list[str]) -> None:
        if lgb is None:
            raise RuntimeError("lightgbm no está instalado")
        self.feature_cols = feature_cols
        self.classifier = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.04,
            num_leaves=48,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
        self.regressor = lgb.LGBMRegressor(
            n_estimators=700,
            learning_rate=0.035,
            num_leaves=64,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="regression",
            random_state=RANDOM_STATE,
            verbose=-1,
        )
        self.mean_positive_: float = 0.0

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> "LightGBMTwoStageModel":
        x = align_features(x_train, self.feature_cols)
        y = y_train.astype(float)
        y_bin = (y > 0).astype(int)
        self.classifier.fit(x, y_bin)
        positive_mask = y > 0
        self.mean_positive_ = float(y.loc[positive_mask].mean()) if positive_mask.any() else 0.0
        if positive_mask.sum() >= 10:
            self.regressor.fit(x.loc[positive_mask], y.loc[positive_mask])
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        features = align_features(x, self.feature_cols)
        prob = self.classifier.predict_proba(features)[:, 1]
        mu = self.regressor.predict(features) if self.mean_positive_ > 0 else np.zeros(len(features))
        return clip_sales_prediction((prob**0.90) * np.maximum(mu, 0.0))

    def predict_scope(self, x: pd.DataFrame) -> pd.Series:
        return pd.Series(["global"] * len(x), index=x.index)


class SegmentedHurdleHGBModel:
    """Entrena modelos Hurdle por categoría/segmento y fallback global."""

    model_id = "segmented_hurdle_hgb"
    model_name = "Hurdle HGB segmentado"
    model_family = "segmented"

    def __init__(self, feature_cols: list[str], segment_col: str = "item_category_id", min_rows: int = 1500) -> None:
        self.feature_cols = feature_cols
        self.segment_col = segment_col
        self.min_rows = min_rows
        self.global_model = HurdleHGBModel(feature_cols)
        self.segment_models: dict[int, HurdleHGBModel] = {}

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> "SegmentedHurdleHGBModel":
        self.global_model.fit(x_train, y_train)
        if self.segment_col not in x_train.columns:
            return self

        counts = x_train[self.segment_col].value_counts().head(8)
        eligible_segments = [int(seg) for seg, n in counts.items() if n >= self.min_rows]
        LOGGER.info("train_segmented_model status=started segments=%s", eligible_segments)
        for segment_id in eligible_segments:
            mask = x_train[self.segment_col].astype(int) == segment_id
            try:
                model = HurdleHGBModel(self.feature_cols)
                model.fit(x_train.loc[mask], y_train.loc[mask])
                self.segment_models[segment_id] = model
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "train_segmented_model status=skipped segment_id=%s error_type=%s error_message=%s",
                    segment_id,
                    type(exc).__name__,
                    str(exc)[:160],
                )
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        pred = self.global_model.predict(x)
        if self.segment_col not in x.columns:
            return pred

        pred_series = pd.Series(pred, index=x.index)
        for segment_id, model in self.segment_models.items():
            mask = x[self.segment_col].fillna(-1).astype(int) == segment_id
            if mask.any():
                pred_series.loc[mask] = model.predict(x.loc[mask])
        return clip_sales_prediction(pred_series.to_numpy())

    def predict_scope(self, x: pd.DataFrame) -> pd.Series:
        scope = pd.Series(["global"] * len(x), index=x.index)
        if self.segment_col not in x.columns:
            return scope
        for segment_id in self.segment_models:
            mask = x[self.segment_col].fillna(-1).astype(int) == segment_id
            scope.loc[mask] = f"segment:cat_{segment_id}"
        return scope


def build_candidate_models(feature_cols: list[str]) -> list[ForecastModel]:
    """Construye catálogo de modelos candidatos."""
    candidates: list[ForecastModel] = [
        NaiveLagModel(),
        ItemMeanFallbackModel(),
        HurdleHGBModel(feature_cols),
        PoissonHGBModel(feature_cols),
        SegmentedHurdleHGBModel(feature_cols),
    ]
    if lgb is not None:
        candidates.append(LightGBMTwoStageModel(feature_cols))
    return candidates


def train_and_evaluate_candidates(
    candidates: list[ForecastModel],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    naive_valid: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, ForecastModel]]:
    """Entrena candidatos y evalúa sobre validación."""
    rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    fitted_models: dict[str, ForecastModel] = {}

    for candidate in candidates:
        start = time.perf_counter()
        LOGGER.info("train_candidate status=started model_id=%s", candidate.model_id)
        try:
            model = candidate.fit(x_train, y_train)
            pred_valid = model.predict(x_valid)
            metrics = evaluate_predictions(y_valid, pred_valid, naive_valid).to_dict()
            duration = time.perf_counter() - start
            row = {
                "model_id": model.model_id,
                "model_name": model.model_name,
                "model_family": model.model_family,
                "status": "trained",
                "duration_seconds": round(duration, 3),
                **metrics,
            }
            rows.append(row)
            predictions[model.model_id] = pred_valid
            fitted_models[model.model_id] = model
            LOGGER.info(
                "train_candidate status=success model_id=%s wape=%.6f mae=%.6f",
                model.model_id,
                float(metrics["wape"]),
                float(metrics["mae"]),
            )
        except Exception as exc:  # noqa: BLE001
            duration = time.perf_counter() - start
            rows.append(
                {
                    "model_id": candidate.model_id,
                    "model_name": candidate.model_name,
                    "model_family": candidate.model_family,
                    "status": "failed",
                    "duration_seconds": round(duration, 3),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:240],
                }
            )
            LOGGER.exception("train_candidate status=failure model_id=%s", candidate.model_id)
    return rows, predictions, fitted_models


def choose_champion(registry_df: pd.DataFrame) -> str:
    """Selecciona champion con política WAPE -> MAE -> recall no cero.

    Se filtran primero modelos entrenados. Si al menos uno mejora al naive en WAPE,
    solo se consideran esos; si ninguno mejora, se elige el mejor entrenado para no
    detener el flujo y se documenta en el registry.
    """
    trained = registry_df.query("status == 'trained'").copy()
    if trained.empty:
        raise RuntimeError("No hubo modelos entrenados correctamente.")

    trained["beats_naive_wape"] = trained["beats_naive_wape"].fillna(False).astype(bool)
    eligible = trained.query("beats_naive_wape == True").copy()
    if eligible.empty:
        eligible = trained

    eligible = eligible.sort_values(
        by=["wape", "mae", "nonzero_recall"],
        ascending=[True, True, False],
    )
    return str(eligible.iloc[0]["model_id"])


def build_ranked_registry(raw_rows: list[dict[str, Any]], champion_id: str) -> pd.DataFrame:
    """Agrega rank, status final y banderas de champion."""
    df = pd.DataFrame(raw_rows)
    trained_mask = df["status"] == "trained"
    ranked = df.loc[trained_mask].sort_values(
        by=["wape", "mae", "nonzero_recall"],
        ascending=[True, True, False],
    )
    rank_map = {model_id: rank + 1 for rank, model_id in enumerate(ranked["model_id"].tolist())}

    df["rank"] = df["model_id"].map(rank_map).fillna(999).astype(int)
    df["is_champion"] = df["model_id"] == champion_id
    df["selection_metric"] = "wape_primary_mae_tiebreak_nonzero_recall"
    df["promotion_reason"] = np.where(
        df["is_champion"],
        "Menor WAPE elegible y mejora al baseline naive cuando hay modelos elegibles.",
        "Challenger registrado para comparación.",
    )
    df.loc[df["is_champion"], "status"] = "promoted"
    df.loc[(df["status"] == "trained") & (~df["is_champion"]), "status"] = "challenger"
    df["created_at"] = utc_now()
    return df.sort_values("rank")


def build_evaluation_detail(
    valid_df: pd.DataFrame,
    y_valid: pd.Series,
    pred_valid: np.ndarray,
    naive_valid: np.ndarray,
    scope: pd.Series,
    data_dir: Path,
) -> pd.DataFrame:
    """Construye evaluación fila a fila enriquecida."""
    eval_df = valid_df.copy()
    eval_df["y_true"] = y_valid.to_numpy()
    eval_df["prediction"] = clip_sales_prediction(pred_valid)
    eval_df["naive_prediction"] = clip_sales_prediction(naive_valid)
    eval_df["error"] = eval_df["prediction"] - eval_df["y_true"]
    eval_df["abs_error"] = eval_df["error"].abs()
    eval_df["model_scope"] = scope.to_numpy()
    return add_catalog_metadata(eval_df, data_dir=data_dir)


def aggregate_with_metrics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Agrega métricas por grupo para app y reporte."""
    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        metrics = evaluate_predictions(
            group["y_true"], group["prediction"], group["naive_prediction"]
        ).to_dict()
        row = {col: key for col, key in zip(group_cols, keys, strict=True)}
        row.update(metrics)
        row["mae"] = metrics["mae"]
        row["naive_mae"] = float(np.mean(np.abs(group["y_true"] - group["naive_prediction"])))
        row["beats_naive_rate"] = float(np.mean(group["abs_error"] < np.abs(group["y_true"] - group["naive_prediction"])))
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values("mae", ascending=False)


def build_review_suggestions(eval_df: pd.DataFrame, limit: int = 250) -> pd.DataFrame:
    """Lista de registros sugeridos para revisión del negocio/ML."""
    df = eval_df.copy()
    df["naive_abs_error"] = (df["y_true"] - df["naive_prediction"]).abs()
    df["model_worse_than_naive"] = df["abs_error"] > df["naive_abs_error"]

    def reason(row: pd.Series) -> str:
        if row.get("product_status") in {"Cold start o sin historial", "Sin ventas recientes"}:
            return "Producto con historial débil; validar continuidad operativa."
        if row["prediction"] > row["y_true"] + 1:
            return "Predicción muy alta vs ground truth."
        if row["prediction"] + 1 < row["y_true"]:
            return "Predicción muy baja vs ground truth."
        if row["model_worse_than_naive"]:
            return "El baseline naive se aproxima mejor que el modelo."
        return "Error absoluto alto; revisar serie."

    df["reason"] = df.apply(reason, axis=1)
    df["recommendation"] = np.where(
        df["model_worse_than_naive"],
        "Comparar features recientes y considerar fallback naive/segmentado.",
        "Revisar historial, categoría y estacionalidad del par tienda-producto.",
    )

    filtered = df.query("abs_error >= 1 or model_worse_than_naive == True").copy()
    if filtered.empty:
        filtered = df.copy()

    cols = [
        "shop_id",
        "shop_name",
        "shop_label",
        "item_id",
        "item_name",
        "item_label",
        "item_category_id",
        "item_category_name",
        "category_group",
        "segment_key",
        "segment_name",
        "y_true",
        "prediction",
        "naive_prediction",
        "abs_error",
        "naive_abs_error",
        "model_scope",
        "model_scope_explanation",
        "product_status",
        "recency_display",
        "reason",
        "recommendation",
    ]
    cols = [col for col in cols if col in filtered.columns]
    return filtered.sort_values(["model_worse_than_naive", "abs_error"], ascending=[False, False]).head(limit)[cols]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Escribe JSON bonito y reproducible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, default=str)


def write_outputs(
    output_dir: Path,
    forecast_detail: pd.DataFrame,
    evaluation_detail: pd.DataFrame,
    registry_df: pd.DataFrame,
    champion_payload: dict[str, Any],
    model_metrics: dict[str, Any],
    review_suggestions: pd.DataFrame,
    champion_model: ForecastModel,
) -> None:
    """Guarda todos los outputs en estructura local compatible con S3."""
    latest = output_dir / "latest"
    registry = output_dir / "registry"

    for path in [latest / "predictions", latest / "evaluation", latest / "model", latest / "diagnostics", registry]:
        path.mkdir(parents=True, exist_ok=True)

    forecast_detail.to_parquet(latest / "predictions" / "forecast_detail.parquet", index=False)

    if "item_category_id" in forecast_detail.columns:
        forecast_detail.groupby(
            ["item_category_id", "item_category_name", "category_group"],
            as_index=False,
            dropna=False,
        ).agg(
            records=("prediction", "size"),
            total_prediction=("prediction", "sum"),
            mean_prediction=("prediction", "mean"),
        ).to_parquet(latest / "predictions" / "forecast_summary_category.parquet", index=False)

    if "shop_id" in forecast_detail.columns and "segment_key" in forecast_detail.columns:
        forecast_detail.groupby(
            ["shop_id", "shop_name", "segment_key", "segment_name"],
            as_index=False,
            dropna=False,
        ).agg(
            records=("prediction", "size"),
            total_prediction=("prediction", "sum"),
            mean_prediction=("prediction", "mean"),
        ).to_parquet(latest / "predictions" / "forecast_summary_shop_segment.parquet", index=False)

    evaluation_detail.to_parquet(latest / "evaluation" / "evaluation_detail.parquet", index=False)

    by_shop_cols = [c for c in ["shop_id", "shop_name", "shop_label"] if c in evaluation_detail.columns]
    by_item_cols = [c for c in ["item_id", "item_name", "item_label", "item_category_name", "category_group"] if c in evaluation_detail.columns]
    by_segment_cols = [c for c in ["segment_key", "segment_name", "category_group"] if c in evaluation_detail.columns]

    if by_shop_cols:
        aggregate_with_metrics(evaluation_detail, by_shop_cols).to_parquet(
            latest / "evaluation" / "evaluation_by_shop.parquet", index=False
        )
    if by_item_cols:
        aggregate_with_metrics(evaluation_detail, by_item_cols).to_parquet(
            latest / "evaluation" / "evaluation_by_item.parquet", index=False
        )
    if by_segment_cols:
        aggregate_with_metrics(evaluation_detail, by_segment_cols).to_parquet(
            latest / "evaluation" / "evaluation_by_segment.parquet", index=False
        )

    review_suggestions.to_parquet(latest / "diagnostics" / "review_suggestions.parquet", index=False)
    write_json(latest / "evaluation" / "model_metrics.json", model_metrics)

    joblib.dump(
        {
            "champion": champion_payload,
            "model": champion_model,
            "created_at": utc_now(),
        },
        latest / "model" / "model_bundle.joblib",
    )

    registry_df.to_csv(registry / "model_runs.csv", index=False)
    write_json(registry / "champion.json", champion_payload)


def upload_directory_to_s3(local_dir: Path, bucket: str, prefix: str) -> None:
    """Sube un directorio completo a S3 usando boto3, sin depender de s3fs."""
    import boto3

    s3_client = boto3.client("s3")
    normalized_prefix = prefix.strip("/")
    for path in local_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(local_dir).as_posix()
        key = f"{normalized_prefix}/{rel}"
        s3_client.upload_file(str(path), bucket, key)
        LOGGER.info("s3_upload status=success bucket=%s key=%s", bucket, key)


def build_champion_payload(registry_df: pd.DataFrame, champion_id: str, s3_bucket: str | None, s3_prefix: str) -> dict[str, Any]:
    """Construye champion.json legible para UI y reporte."""
    row = registry_df.query("model_id == @champion_id").iloc[0].to_dict()
    base_uri = f"s3://{s3_bucket}/{s3_prefix.strip('/')}" if s3_bucket else f"local://{s3_prefix.strip('/')}"
    payload = {
        "model_id": row["model_id"],
        "model_name": row["model_name"],
        "model_family": row["model_family"],
        "created_at": row["created_at"],
        "status": "promoted",
        "is_champion": True,
        "selection_policy": "Menor WAPE, MAE como desempate y recall de ventas no cero como segundo desempate.",
        "business_reason": "WAPE prioriza error relativo al volumen total, más estable para demanda intermitente con muchos ceros.",
        "source_prefix": base_uri,
        "model_uri": f"{base_uri}/model/model_bundle.joblib",
        "forecast_uri": f"{base_uri}/predictions/forecast_detail.parquet",
        "metrics_uri": f"{base_uri}/evaluation/model_metrics.json",
        "review_suggestions_uri": f"{base_uri}/diagnostics/review_suggestions.parquet",
    }
    metric_keys = [
        "rmse",
        "mae",
        "wape",
        "smape",
        "bias",
        "nonzero_recall",
        "nonzero_precision",
        "beats_naive_mae",
        "beats_naive_wape",
    ]
    payload["metrics"] = {key: row.get(key) for key in metric_keys if key in row}
    return payload


def main(argv: list[str] | None = None) -> None:
    """Punto de entrada del entrenamiento model zoo."""
    configure_logging()
    args = build_parser().parse_args(argv)

    prep_dir = args.prep_dir
    data_dir = args.data_dir
    output_dir = args.output_dir
    if output_dir.exists() and args.clean_output:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("load_data status=started prep_dir=%s", prep_dir)
    train_df = load_required_parquet(prep_dir / "train.parquet")
    valid_df = load_required_parquet(prep_dir / "valid.parquet")
    test_features = load_required_parquet(prep_dir / "test_features.parquet")
    test_pairs = load_required_parquet(prep_dir / "test_pairs.parquet")
    meta = load_json(prep_dir / "meta.json")

    feature_cols = infer_feature_columns(train_df, meta)
    feature_cols = numeric_features(train_df, feature_cols)
    LOGGER.info(
        "load_data status=success train_rows=%d valid_rows=%d test_rows=%d features=%d target_column=%s",
        len(train_df),
        len(valid_df),
        len(test_features),
        len(feature_cols),
        TARGET_COLUMN,
    )

    if TARGET_COLUMN not in train_df.columns or TARGET_COLUMN not in valid_df.columns:
        raise ValueError(
            f"No se encontró target '{TARGET_COLUMN}'. Este script conserva la variable target del entrenamiento original."
        )

    x_train = train_df[feature_cols].copy()
    y_train = train_df[TARGET_COLUMN].astype(float).clip(PREDICTION_MIN, PREDICTION_MAX)
    x_valid = valid_df[feature_cols].copy()
    y_valid = valid_df[TARGET_COLUMN].astype(float).clip(PREDICTION_MIN, PREDICTION_MAX)
    x_test = align_features(test_features, feature_cols)
    naive_valid = build_naive_prediction(valid_df)

    candidates = build_candidate_models(feature_cols)
    raw_rows, predictions, fitted_models = train_and_evaluate_candidates(
        candidates, x_train, y_train, x_valid, y_valid, naive_valid
    )

    raw_registry = pd.DataFrame(raw_rows)
    champion_id = choose_champion(raw_registry)
    registry_df = build_ranked_registry(raw_rows, champion_id)
    champion_model = fitted_models[champion_id]

    pred_valid = predictions[champion_id]
    valid_scope = champion_model.predict_scope(valid_df)
    evaluation_detail = build_evaluation_detail(
        valid_df=valid_df,
        y_valid=y_valid,
        pred_valid=pred_valid,
        naive_valid=naive_valid,
        scope=valid_scope,
        data_dir=data_dir,
    )

    test_pred = champion_model.predict(x_test)
    test_scope = champion_model.predict_scope(test_features)
    forecast_detail = test_pairs.copy()
    forecast_detail["prediction"] = test_pred
    forecast_detail["model_scope"] = test_scope.to_numpy()
    for optional_col in ["recency", "months_active", "demand_tier", "price_tier", "item_category_id"]:
        if optional_col in test_features.columns and optional_col not in forecast_detail.columns:
            forecast_detail[optional_col] = test_features[optional_col].to_numpy()
    forecast_detail = add_catalog_metadata(forecast_detail, data_dir=data_dir)

    review_suggestions = build_review_suggestions(evaluation_detail)
    champion_payload = build_champion_payload(registry_df, champion_id, args.s3_bucket, args.s3_prefix)

    naive_metrics = evaluate_predictions(y_valid, naive_valid).to_dict()
    champion_metrics = evaluate_predictions(y_valid, pred_valid, naive_valid).to_dict()
    model_metrics = {
        "created_at": utc_now(),
        "target_column": TARGET_COLUMN,
        "selection_policy": champion_payload["selection_policy"],
        "champion": champion_payload,
        "global": champion_metrics,
        "naive": naive_metrics,
        "all_models": registry_df.to_dict(orient="records"),
    }

    write_outputs(
        output_dir=output_dir,
        forecast_detail=forecast_detail,
        evaluation_detail=evaluation_detail,
        registry_df=registry_df,
        champion_payload=champion_payload,
        model_metrics=model_metrics,
        review_suggestions=review_suggestions,
        champion_model=champion_model,
    )

    if args.s3_bucket:
        upload_directory_to_s3(output_dir / "latest", args.s3_bucket, args.s3_prefix)
        upload_directory_to_s3(output_dir / "registry", args.s3_bucket, args.registry_prefix)

    LOGGER.info(
        "model_zoo status=success champion_id=%s champion_name=%s output_dir=%s",
        champion_payload["model_id"],
        champion_payload["model_name"],
        output_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    """Parser CLI."""
    parser = argparse.ArgumentParser(description="Entrena catálogo de modelos ModelOps v2")
    parser.add_argument("--prep-dir", type=Path, default=Path("data/prep"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("modelops_outputs"))
    parser.add_argument("--s3-bucket", type=str, default=os.getenv("MODEL_BUCKET"))
    parser.add_argument("--s3-prefix", type=str, default=os.getenv("MODELOPS_PREFIX", "modelops/latest"))
    parser.add_argument(
        "--registry-prefix",
        type=str,
        default=os.getenv("MODELOPS_REGISTRY_PREFIX", "modelops/registry"),
    )
    parser.add_argument("--clean-output", action="store_true", default=True)
    return parser


if __name__ == "__main__":
    main()
