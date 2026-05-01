"""Stable ModelOps loaders for the Streamlit MVP.

This file is intentionally self-contained. It supports both local outputs and S3
outputs and exposes every loader used by `frontend/app.py`.

Local layout supported:
- modelops_outputs/latest/predictions/forecast_detail.parquet
- modelops_outputs/latest/evaluation/model_metrics.json
- modelops_outputs/registry/champion.json
- modelops_outputs/registry/model_runs.csv

It also supports older layouts without `latest/`, for example:
- modelops_outputs/predictions/forecast_detail.parquet
- modelops_outputs/evaluation/model_metrics.json
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import boto3
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelOpsConfig:
    use_s3: bool
    bucket: str | None
    latest_prefix: str
    registry_prefix: str
    local_root: Path
    region: str


def get_config() -> ModelOpsConfig:
    """Resolve storage configuration from environment variables."""
    return ModelOpsConfig(
        use_s3=os.getenv("USE_MODELOPS_S3", "true").lower() == "true",
        bucket=os.getenv("MODEL_BUCKET") or os.getenv("APP_EXPORT_BUCKET"),
        latest_prefix=os.getenv("MODELOPS_PREFIX", "modelops/latest").strip("/"),
        registry_prefix=os.getenv("MODELOPS_REGISTRY_PREFIX", "modelops/registry").strip("/"),
        local_root=Path(os.getenv("MODELOPS_LOCAL_ROOT", "modelops_outputs")),
        region=os.getenv("AWS_REGION", "us-east-1"),
    )


def _s3_client():
    return boto3.client("s3", region_name=get_config().region)


def _read_s3_bytes(bucket: str, key: str) -> bytes:
    LOGGER.info("action=s3_read bucket=%s key=%s", bucket, key)
    return _s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()


def _local_candidates_latest(relative_key: str) -> list[Path]:
    cfg = get_config()
    rel = relative_key.strip("/")
    return [
        cfg.local_root / "latest" / rel,
        cfg.local_root / rel,
    ]


def _local_candidates_registry(relative_key: str) -> list[Path]:
    cfg = get_config()
    rel = relative_key.strip("/")
    return [
        cfg.local_root / "registry" / rel,
        cfg.local_root / "latest" / "registry" / rel,
    ]


def _read_local_bytes(candidates: list[Path]) -> bytes:
    for path in candidates:
        if path.exists():
            LOGGER.info("action=local_read path=%s", path)
            return path.read_bytes()
    raise FileNotFoundError("No matching local file. Tried: " + ", ".join(str(p) for p in candidates))


def _read_latest_bytes(relative_key: str) -> bytes:
    cfg = get_config()
    rel = relative_key.strip("/")
    if cfg.use_s3:
        if not cfg.bucket:
            raise RuntimeError("MODEL_BUCKET or APP_EXPORT_BUCKET is required when USE_MODELOPS_S3=true")
        return _read_s3_bytes(cfg.bucket, f"{cfg.latest_prefix}/{rel}")
    return _read_local_bytes(_local_candidates_latest(rel))


def _read_registry_bytes(relative_key: str) -> bytes:
    cfg = get_config()
    rel = relative_key.strip("/")
    if cfg.use_s3:
        if not cfg.bucket:
            raise RuntimeError("MODEL_BUCKET or APP_EXPORT_BUCKET is required when USE_MODELOPS_S3=true")
        return _read_s3_bytes(cfg.bucket, f"{cfg.registry_prefix}/{rel}")
    return _read_local_bytes(_local_candidates_registry(rel))


def _read_parquet_latest(relative_key: str) -> pd.DataFrame:
    return pd.read_parquet(BytesIO(_read_latest_bytes(relative_key)))


def _read_csv_latest(relative_key: str) -> pd.DataFrame:
    return pd.read_csv(BytesIO(_read_latest_bytes(relative_key)))


def _read_json_latest(relative_key: str) -> dict[str, Any]:
    return json.loads(_read_latest_bytes(relative_key).decode("utf-8"))


def _read_csv_registry(relative_key: str) -> pd.DataFrame:
    return pd.read_csv(BytesIO(_read_registry_bytes(relative_key)))


def _read_json_registry(relative_key: str) -> dict[str, Any]:
    return json.loads(_read_registry_bytes(relative_key).decode("utf-8"))


def _optional_parquet_latest(*relative_keys: str) -> pd.DataFrame:
    for key in relative_keys:
        try:
            return _read_parquet_latest(key)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("action=optional_parquet status=missing key=%s error=%s", key, exc)
    return pd.DataFrame()


def _optional_csv_latest(*relative_keys: str) -> pd.DataFrame:
    for key in relative_keys:
        try:
            return _read_csv_latest(key)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("action=optional_csv status=missing key=%s error=%s", key, exc)
    return pd.DataFrame()


def _optional_json_latest(*relative_keys: str) -> dict[str, Any]:
    for key in relative_keys:
        try:
            return _read_json_latest(key)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("action=optional_json status=missing key=%s error=%s", key, exc)
    return {}


def load_forecast_detail() -> pd.DataFrame:
    """Load detailed future forecasts."""
    try:
        return _read_parquet_latest("predictions/forecast_detail.parquet")
    except Exception:
        return _read_csv_latest("predictions/forecast_detail.csv")


def load_forecast_summary_by_category() -> pd.DataFrame:
    """Load or derive forecast summary by category."""
    summary = _optional_parquet_latest(
        "predictions/forecast_summary_by_category.parquet",
        "predictions/forecast_summary_category.parquet",
    )
    if not summary.empty:
        return summary

    detail = load_forecast_detail()
    group_cols = [col for col in ["item_category_id", "item_category_name", "category_group"] if col in detail.columns]
    if not group_cols or "prediction" not in detail.columns:
        return pd.DataFrame()
    return (
        detail.groupby(group_cols, dropna=False, as_index=False)
        .agg(n=("prediction", "size"), total_prediction=("prediction", "sum"), mean_prediction=("prediction", "mean"))
        .sort_values("total_prediction", ascending=False)
    )


def load_forecast_summary_by_shop_segment() -> pd.DataFrame:
    """Load or derive forecast summary by shop/segment."""
    summary = _optional_parquet_latest(
        "predictions/forecast_summary_by_shop_segment.parquet",
        "predictions/forecast_summary_shop_segment.parquet",
    )
    if not summary.empty:
        return summary

    detail = load_forecast_detail()
    group_cols = [
        col
        for col in ["shop_id", "shop_name", "segment_key", "segment_name", "item_category_id", "model_scope"]
        if col in detail.columns
    ]
    if not group_cols or "prediction" not in detail.columns:
        return pd.DataFrame()
    return (
        detail.groupby(group_cols, dropna=False, as_index=False)
        .agg(n=("prediction", "size"), total_prediction=("prediction", "sum"), mean_prediction=("prediction", "mean"))
        .sort_values("total_prediction", ascending=False)
    )


def load_evaluation_detail() -> pd.DataFrame:
    return _optional_parquet_latest("evaluation/evaluation_detail.parquet")


def load_evaluation_by_segment() -> pd.DataFrame:
    return _optional_parquet_latest("evaluation/evaluation_by_segment.parquet")


def load_evaluation_by_item() -> pd.DataFrame:
    return _optional_parquet_latest("evaluation/evaluation_by_item.parquet")


def load_evaluation_by_shop() -> pd.DataFrame:
    return _optional_parquet_latest("evaluation/evaluation_by_shop.parquet")


def load_evaluation_curves_by_model() -> pd.DataFrame:
    """Load optional demand curves by model.

    Expected columns can be either:
    - model_name/model_id, demand_bucket, real_mean, pred_mean
    - Serie/demand_bucket/mean_value style; the app normalizes if needed.
    """
    curves = _optional_parquet_latest(
        "evaluation/evaluation_curves_by_model.parquet",
        "evaluation/evaluation_demand_curve.parquet",
    )
    if not curves.empty:
        return curves
    return _optional_csv_latest("evaluation/evaluation_curves_by_model.csv")


def load_model_metrics() -> dict[str, Any]:
    return _optional_json_latest("evaluation/model_metrics.json")


def _normalize_champion_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict) or not payload:
        return {}
    if "champion" in payload and isinstance(payload["champion"], dict):
        champion = payload["champion"].copy()
        for key in ["selection_policy", "created_at"]:
            if key not in champion and key in payload:
                champion[key] = payload[key]
        if "model_run_id" not in champion and "model_id" in champion:
            champion["model_run_id"] = champion.get("model_id")
        return champion
    if "model_run_id" not in payload and "model_id" in payload:
        payload = payload.copy()
        payload["model_run_id"] = payload.get("model_id")
    return payload


def load_champion() -> dict[str, Any]:
    """Load normalized champion metadata from registry or metrics fallback."""
    try:
        payload = _read_json_registry("champion.json")
        champion = _normalize_champion_payload(payload)
        if champion:
            return champion
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("action=load_champion status=registry_missing error=%s", exc)

    metrics = load_model_metrics()
    champion = _normalize_champion_payload(metrics.get("champion", {}) if isinstance(metrics, dict) else {})
    if champion:
        return champion

    # Last fallback: choose champion row from all_models.
    all_models = metrics.get("all_models", []) if isinstance(metrics, dict) else []
    if isinstance(all_models, list) and all_models:
        rows = pd.DataFrame(all_models)
        if "is_champion" in rows.columns and rows["is_champion"].astype(bool).any():
            return rows.loc[rows["is_champion"].astype(bool)].iloc[0].to_dict()
        metric = "rmse" if "rmse" in rows.columns else "mae"
        if metric in rows.columns:
            return rows.sort_values(metric, ascending=True).iloc[0].to_dict()
    return {}


def load_model_runs() -> pd.DataFrame:
    """Load model runs registry with fallback to all_models in model_metrics.json."""
    try:
        runs = _read_csv_registry("model_runs.csv")
        if not runs.empty:
            return runs
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("action=load_model_runs status=registry_missing error=%s", exc)

    metrics = load_model_metrics()
    all_models = metrics.get("all_models", []) if isinstance(metrics, dict) else []
    if isinstance(all_models, list) and all_models:
        return pd.DataFrame(all_models)
    return pd.DataFrame()


def load_review_suggestions() -> pd.DataFrame:
    suggestions = _optional_parquet_latest(
        "review/review_suggestions.parquet",
        "evaluation/review_suggestions.parquet",
        "diagnostics/review_suggestions.parquet",
    )
    if not suggestions.empty:
        return suggestions
    return _optional_csv_latest("review/review_suggestions.csv")


def load_overestimated_products() -> pd.DataFrame:
    over = _optional_parquet_latest("review/overestimated_products.parquet", "evaluation/overestimated_products.parquet")
    if not over.empty:
        return over
    detail = load_evaluation_detail()
    return build_overestimated_products(detail, top_n=30)


def load_underestimated_products() -> pd.DataFrame:
    under = _optional_parquet_latest("review/underestimated_products.parquet", "evaluation/underestimated_products.parquet")
    if not under.empty:
        return under
    detail = load_evaluation_detail()
    return build_underestimated_products(detail, top_n=100)


def _find_actual_pred_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    actual = next((c for c in ["y", "y_true", "actual", "target"] if c in df.columns), None)
    pred = next((c for c in ["prediction", "pred", "y_pred", "forecast"] if c in df.columns), None)
    return actual, pred


def build_overestimated_products(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """Return top overestimated rows: prediction > actual."""
    if df is None or df.empty:
        return pd.DataFrame()
    actual, pred = _find_actual_pred_columns(df)
    if actual is None or pred is None:
        return pd.DataFrame()
    out = df.copy()
    out["over_error"] = pd.to_numeric(out[pred], errors="coerce") - pd.to_numeric(out[actual], errors="coerce")
    out = out[out["over_error"] > 0].sort_values("over_error", ascending=False)
    return out.head(top_n)


def build_underestimated_products(df: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    """Return top underestimated rows: actual > prediction."""
    if df is None or df.empty:
        return pd.DataFrame()
    actual, pred = _find_actual_pred_columns(df)
    if actual is None or pred is None:
        return pd.DataFrame()
    out = df.copy()
    out["under_error"] = pd.to_numeric(out[actual], errors="coerce") - pd.to_numeric(out[pred], errors="coerce")
    out = out[out["under_error"] > 0].sort_values("under_error", ascending=False)
    return out.head(top_n)


def modelops_healthcheck() -> dict[str, Any]:
    cfg = get_config()
    if cfg.use_s3:
        latest_root = f"s3://{cfg.bucket}/{cfg.latest_prefix}"
        registry_root = f"s3://{cfg.bucket}/{cfg.registry_prefix}"
    else:
        latest_root = str(cfg.local_root / "latest")
        registry_root = str(cfg.local_root / "registry")
    return {
        "use_s3": cfg.use_s3,
        "bucket": cfg.bucket,
        "latest_prefix": cfg.latest_prefix,
        "registry_prefix": cfg.registry_prefix,
        "local_root": str(cfg.local_root),
        "forecast_detail": f"{latest_root}/predictions/forecast_detail.parquet",
        "evaluation_detail": f"{latest_root}/evaluation/evaluation_detail.parquet",
        "evaluation_by_segment": f"{latest_root}/evaluation/evaluation_by_segment.parquet",
        "evaluation_by_item": f"{latest_root}/evaluation/evaluation_by_item.parquet",
        "evaluation_curves_by_model": f"{latest_root}/evaluation/evaluation_curves_by_model.parquet",
        "model_metrics": f"{latest_root}/evaluation/model_metrics.json",
        "champion": f"{registry_root}/champion.json",
        "model_runs": f"{registry_root}/model_runs.csv",
    }
