"""S3 loaders for Streamlit ModelOps outputs with dimension enrichment.

This module replaces/extends the previous `backend/modelops_s3.py`. It reads
forecast/evaluation outputs from `modelops/latest`, enriches them with shop and
item/category names, and hides confusing sentinel values such as recency=99.
"""

from __future__ import annotations

import json
import os
from io import BytesIO

import boto3
import pandas as pd

from modelops.data_quality_dimensions import (
    build_item_dimension,
    build_shop_dimension,
    enrich_with_dimensions,
)

DEFAULT_PREFIX = "modelops/latest"
DEFAULT_REGISTRY_PREFIX = "modelops/registry"
RAW_CURRENT_PREFIX = "raw/current"


def _s3_client():
    return boto3.client("s3", region_name=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"))


def get_model_bucket() -> str:
    """Return the S3 bucket where ModelOps artifacts live."""
    bucket = os.getenv("MODEL_BUCKET")
    if not bucket:
        raise RuntimeError("MODEL_BUCKET is not configured")
    return bucket


def get_modelops_prefix() -> str:
    """Return current ModelOps artifact prefix."""
    return os.getenv("MODELOPS_PREFIX", DEFAULT_PREFIX).strip("/")


def get_registry_prefix() -> str:
    """Return model registry prefix."""
    return os.getenv("MODELOPS_REGISTRY_PREFIX", DEFAULT_REGISTRY_PREFIX).strip("/")


def _read_s3_bytes(key: str) -> bytes:
    obj = _s3_client().get_object(Bucket=get_model_bucket(), Key=key)
    return obj["Body"].read()


def _read_csv_key(key: str) -> pd.DataFrame:
    obj = _s3_client().get_object(Bucket=get_model_bucket(), Key=key)
    return pd.read_csv(obj["Body"])


def read_modelops_parquet(relative_key: str) -> pd.DataFrame:
    """Read parquet below MODELOPS_PREFIX."""
    key = f"{get_modelops_prefix()}/{relative_key.lstrip('/')}"
    return pd.read_parquet(BytesIO(_read_s3_bytes(key)))


def read_modelops_csv(relative_key: str) -> pd.DataFrame:
    """Read CSV below MODELOPS_PREFIX."""
    key = f"{get_modelops_prefix()}/{relative_key.lstrip('/')}"
    return _read_csv_key(key)


def read_modelops_json(relative_key: str) -> dict:
    """Read JSON below MODELOPS_PREFIX."""
    key = f"{get_modelops_prefix()}/{relative_key.lstrip('/')}"
    return json.loads(_read_s3_bytes(key).decode("utf-8"))


_shop_dimension_cache: pd.DataFrame | None = None
_item_dimension_cache: pd.DataFrame | None = None


def load_shop_dimension() -> pd.DataFrame | None:
    """Load and clean shops from raw/current/shops_en.csv."""
    global _shop_dimension_cache
    if _shop_dimension_cache is not None:
        return _shop_dimension_cache
    try:
        shops = _read_csv_key(f"{RAW_CURRENT_PREFIX}/shops_en.csv")
        _shop_dimension_cache = build_shop_dimension(shops)
        return _shop_dimension_cache
    except Exception:
        return None


def load_item_dimension() -> pd.DataFrame | None:
    """Load and clean items/categories from raw/current."""
    global _item_dimension_cache
    if _item_dimension_cache is not None:
        return _item_dimension_cache
    try:
        items = _read_csv_key(f"{RAW_CURRENT_PREFIX}/items_en.csv")
        try:
            categories = _read_csv_key(f"{RAW_CURRENT_PREFIX}/item_categories_en.csv")
        except Exception:
            categories = None
        _item_dimension_cache = build_item_dimension(items, categories)
        return _item_dimension_cache
    except Exception:
        return None


def enrich_table(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich an output table with friendly dimensions when possible."""
    return enrich_with_dimensions(
        df,
        shop_dimension=load_shop_dimension(),
        item_dimension=load_item_dimension(),
    )


def load_forecast_detail() -> pd.DataFrame:
    return enrich_table(read_modelops_parquet("predictions/forecast_detail.parquet"))


def load_forecast_summary_by_category() -> pd.DataFrame:
    return enrich_table(read_modelops_parquet("predictions/forecast_summary_by_category.parquet"))


def load_forecast_summary_by_shop_segment() -> pd.DataFrame:
    return enrich_table(read_modelops_parquet("predictions/forecast_summary_by_shop_segment.parquet"))


def load_submission() -> pd.DataFrame:
    return read_modelops_csv("predictions/submission.csv")


def load_evaluation_detail() -> pd.DataFrame:
    return enrich_table(read_modelops_parquet("evaluation/evaluation_detail.parquet"))


def load_evaluation_by_segment() -> pd.DataFrame:
    return enrich_table(read_modelops_parquet("evaluation/evaluation_by_segment.parquet"))


def load_evaluation_by_item() -> pd.DataFrame:
    return enrich_table(read_modelops_parquet("evaluation/evaluation_by_item.parquet"))


def load_model_metrics() -> dict:
    return read_modelops_json("evaluation/model_metrics.json")


def load_model_runs() -> pd.DataFrame:
    key = f"{get_registry_prefix()}/model_runs.csv"
    return _read_csv_key(key)


def load_champion() -> dict:
    key = f"{get_registry_prefix()}/champion.json"
    return json.loads(_read_s3_bytes(key).decode("utf-8"))


def modelops_healthcheck() -> dict:
    bucket = get_model_bucket()
    prefix = get_modelops_prefix()
    registry_prefix = get_registry_prefix()
    return {
        "bucket": bucket,
        "prefix": prefix,
        "registry_prefix": registry_prefix,
        "forecast_detail": f"s3://{bucket}/{prefix}/predictions/forecast_detail.parquet",
        "evaluation_by_segment": f"s3://{bucket}/{prefix}/evaluation/evaluation_by_segment.parquet",
        "evaluation_by_item": f"s3://{bucket}/{prefix}/evaluation/evaluation_by_item.parquet",
        "model_metrics": f"s3://{bucket}/{prefix}/evaluation/model_metrics.json",
        "champion": f"s3://{bucket}/{registry_prefix}/champion.json",
        "model_runs": f"s3://{bucket}/{registry_prefix}/model_runs.csv",
        "shops": f"s3://{bucket}/{RAW_CURRENT_PREFIX}/shops_en.csv",
        "items": f"s3://{bucket}/{RAW_CURRENT_PREFIX}/items_en.csv",
    }
