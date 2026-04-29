"""Loaders for ModelOps outputs stored in Amazon S3.

The Streamlit app uses this module to read precomputed batch forecasts and
model-evaluation artifacts from S3. It intentionally uses boto3 instead of s3fs
to avoid dependency conflicts in ECS and SageMaker containers.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from io import BytesIO

import boto3
import pandas as pd
from botocore.exceptions import ClientError

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelOpsConfig:
    """Runtime configuration for S3 ModelOps artifacts."""

    bucket: str
    prefix: str = "modelops/latest"
    registry_prefix: str = "modelops/registry"


def get_config() -> ModelOpsConfig:
    """Read ModelOps settings from environment variables."""
    bucket = os.getenv("MODEL_BUCKET")
    if not bucket:
        raise RuntimeError("MODEL_BUCKET is not configured")

    return ModelOpsConfig(
        bucket=bucket,
        prefix=os.getenv("MODELOPS_PREFIX", "modelops/latest").strip("/"),
        registry_prefix=os.getenv("MODELOPS_REGISTRY_PREFIX", "modelops/registry").strip("/"),
    )


def _s3_client():
    """Create an S3 client using the default AWS credential chain."""
    return boto3.client("s3", region_name=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"))


def _read_s3_bytes(key: str) -> bytes:
    """Read an S3 object as bytes from the configured ModelOps bucket."""
    cfg = get_config()
    try:
        response = _s3_client().get_object(Bucket=cfg.bucket, Key=key)
        return response["Body"].read()
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "Unknown")
        LOGGER.warning(
            "action=s3_read status=failure bucket=%s key=%s error_code=%s",
            cfg.bucket,
            key,
            error_code,
        )
        raise


def _object_exists(key: str) -> bool:
    """Return True when an S3 key exists."""
    cfg = get_config()
    try:
        _s3_client().head_object(Bucket=cfg.bucket, Key=key)
        return True
    except ClientError:
        return False


def read_modelops_parquet(relative_key: str) -> pd.DataFrame:
    """Read a parquet file under MODELOPS_PREFIX."""
    cfg = get_config()
    key = f"{cfg.prefix}/{relative_key.lstrip('/')}"
    return pd.read_parquet(BytesIO(_read_s3_bytes(key)))


def read_modelops_csv(relative_key: str) -> pd.DataFrame:
    """Read a CSV file under MODELOPS_PREFIX."""
    cfg = get_config()
    key = f"{cfg.prefix}/{relative_key.lstrip('/')}"
    return pd.read_csv(BytesIO(_read_s3_bytes(key)))


def read_modelops_json(relative_key: str) -> dict:
    """Read a JSON file under MODELOPS_PREFIX."""
    cfg = get_config()
    key = f"{cfg.prefix}/{relative_key.lstrip('/')}"
    return json.loads(_read_s3_bytes(key).decode("utf-8"))


def read_registry_json(filename: str) -> dict:
    """Read a JSON file under MODELOPS_REGISTRY_PREFIX."""
    cfg = get_config()
    key = f"{cfg.registry_prefix}/{filename.lstrip('/')}"
    try:
        return json.loads(_read_s3_bytes(key).decode("utf-8"))
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404"}:
            return {}
        raise


def read_registry_csv(filename: str) -> pd.DataFrame:
    """Read a CSV file under MODELOPS_REGISTRY_PREFIX."""
    cfg = get_config()
    key = f"{cfg.registry_prefix}/{filename.lstrip('/')}"
    try:
        return pd.read_csv(BytesIO(_read_s3_bytes(key)))
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404"}:
            return pd.DataFrame()
        raise


def _ensure_prediction_column(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize forecast column names for the Streamlit app."""
    result = df.copy()
    if "prediction" not in result.columns:
        for candidate in ["y_hat", "forecast", "pred", "item_cnt_month"]:
            if candidate in result.columns:
                result["prediction"] = result[candidate]
                break
    if "prediction" in result.columns:
        result["prediction"] = result["prediction"].clip(0, 20)
    return result


def load_forecast_detail() -> pd.DataFrame:
    """Load precomputed forecasts by shop-item pair."""
    return _ensure_prediction_column(read_modelops_parquet("predictions/forecast_detail.parquet"))


def load_forecast_summary_by_category() -> pd.DataFrame:
    """Load forecast summary by item category."""
    return read_modelops_parquet("predictions/forecast_summary_by_category.parquet")


def load_forecast_summary_by_shop_segment() -> pd.DataFrame:
    """Load forecast summary by shop and product segment."""
    return read_modelops_parquet("predictions/forecast_summary_by_shop_segment.parquet")


def load_submission() -> pd.DataFrame:
    """Load Kaggle-style submission from ModelOps latest prefix."""
    return read_modelops_csv("predictions/submission.csv")


def load_evaluation_detail() -> pd.DataFrame:
    """Load evaluation detail with ground truth and predictions."""
    return read_modelops_parquet("evaluation/evaluation_detail.parquet")


def load_evaluation_by_segment() -> pd.DataFrame:
    """Load evaluation metrics by segment/category."""
    return read_modelops_parquet("evaluation/evaluation_by_segment.parquet")


def load_evaluation_by_item() -> pd.DataFrame:
    """Load evaluation metrics by product."""
    return read_modelops_parquet("evaluation/evaluation_by_item.parquet")


def load_model_metrics() -> dict:
    """Load global model metrics."""
    return read_modelops_json("evaluation/model_metrics.json")


def load_champion() -> dict:
    """Load current champion metadata from registry."""
    return read_registry_json("champion.json")


def load_model_runs() -> pd.DataFrame:
    """Load model registry table."""
    return read_registry_csv("model_runs.csv")


def modelops_healthcheck() -> dict:
    """Return expected S3 keys and whether they exist."""
    cfg = get_config()
    keys = {
        "forecast_detail": f"{cfg.prefix}/predictions/forecast_detail.parquet",
        "forecast_summary_by_category": f"{cfg.prefix}/predictions/forecast_summary_by_category.parquet",
        "forecast_summary_by_shop_segment": f"{cfg.prefix}/predictions/forecast_summary_by_shop_segment.parquet",
        "evaluation_by_segment": f"{cfg.prefix}/evaluation/evaluation_by_segment.parquet",
        "evaluation_by_item": f"{cfg.prefix}/evaluation/evaluation_by_item.parquet",
        "model_metrics": f"{cfg.prefix}/evaluation/model_metrics.json",
        "champion": f"{cfg.registry_prefix}/champion.json",
        "model_runs": f"{cfg.registry_prefix}/model_runs.csv",
    }
    return {
        "bucket": cfg.bucket,
        "prefix": cfg.prefix,
        "registry_prefix": cfg.registry_prefix,
        "objects": {
            name: {
                "s3_uri": f"s3://{cfg.bucket}/{key}",
                "exists": _object_exists(key),
            }
            for name, key in keys.items()
        },
    }
