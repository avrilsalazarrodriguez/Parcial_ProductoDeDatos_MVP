"""Data-quality and dimension helpers for ModelOps outputs.

This module fixes the practical issues found during dashboard validation:

* Rows with category/shop metadata as null, "None", "nan", or "unknown".
* Sentinel recency values such as 99 that mean "no recent sales history".
* Forecast/evaluation tables without `shop_name` or friendly labels.

The functions are intentionally pure pandas utilities so they can run locally,
inside SageMaker Processing, or inside a Streamlit container without depending
on Spark/Athena.
"""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from typing import Iterable

import boto3
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
MISSING_STRINGS = {"", "none", "null", "nan", "na", "n/a", "unknown", "<na>"}
UNKNOWN_TEXT = "Sin clasificar"
UNKNOWN_SHOP_TEXT = "Tienda sin nombre"
UNKNOWN_ITEM_TEXT = "Producto sin nombre"
RECENCY_SENTINELS = {99, 999, -1}


def _normalise_text_value(value: object, fallback: str = UNKNOWN_TEXT) -> str:
    """Return a clean display string for object/categorical values."""
    if value is None or pd.isna(value):
        return fallback
    text = str(value).strip()
    if text.lower() in MISSING_STRINGS:
        return fallback
    return text


def clean_text_columns(df: pd.DataFrame, columns: Iterable[str], fallback: str = UNKNOWN_TEXT) -> pd.DataFrame:
    """Replace null-like strings in selected columns with a friendly fallback."""
    result = df.copy()
    for column in columns:
        if column in result.columns:
            result[column] = result[column].map(lambda value: _normalise_text_value(value, fallback))
    return result


def clean_recency_columns(df: pd.DataFrame, columns: Iterable[str] = ("recency",)) -> pd.DataFrame:
    """Make recency columns interpretable for modeling and the UI.

    The existing feature pipeline uses `99` as a sentinel for "no recent sales".
    That is useful internally, but it looks confusing in the dashboard. This
    function keeps the original column, adds an indicator, and creates a label:

    * `recency_is_missing_or_very_old`: 1 when value is 99/999/-1/null.
    * `recency_for_model`: numeric value where sentinels are replaced by 0.
    * `recency_label`: human-friendly text for the app.
    """
    result = df.copy()
    for column in columns:
        if column not in result.columns:
            continue
        numeric = pd.to_numeric(result[column], errors="coerce")
        is_missing = numeric.isna() | numeric.isin(list(RECENCY_SENTINELS)) | (numeric >= 90)
        result[f"{column}_is_missing_or_very_old"] = is_missing.astype(int)
        result[f"{column}_for_model"] = numeric.mask(is_missing, 0).fillna(0).clip(lower=0)
        result[f"{column}_label"] = np.where(
            is_missing,
            "Sin ventas recientes / sin historial",
            numeric.fillna(0).astype(int).astype(str) + " meses desde última venta",
        )
    return result


def build_shop_dimension(shops: pd.DataFrame) -> pd.DataFrame:
    """Build a clean shop dimension from shops_en.csv."""
    result = shops.copy()
    if "shop_id" not in result.columns:
        raise ValueError("shops table must contain shop_id")

    name_col = "shop_name" if "shop_name" in result.columns else None
    if name_col is None:
        candidates = [column for column in result.columns if column != "shop_id"]
        if not candidates:
            result["shop_name"] = UNKNOWN_SHOP_TEXT
        else:
            result["shop_name"] = result[candidates[0]]
    result["shop_id"] = pd.to_numeric(result["shop_id"], errors="coerce").astype("Int64")
    result["shop_name"] = result["shop_name"].map(
        lambda value: _normalise_text_value(value, UNKNOWN_SHOP_TEXT)
    )
    result["shop_label"] = result["shop_id"].astype(str) + " — " + result["shop_name"]
    return result[["shop_id", "shop_name", "shop_label"]].drop_duplicates("shop_id")


def build_item_dimension(items: pd.DataFrame, categories: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build a clean item/category dimension from items_en.csv and categories."""
    result = items.copy()
    if "item_id" not in result.columns:
        raise ValueError("items table must contain item_id")

    if "item_name" not in result.columns:
        candidates = [column for column in result.columns if column not in {"item_id", "item_category_id"}]
        result["item_name"] = result[candidates[0]] if candidates else UNKNOWN_ITEM_TEXT

    result["item_id"] = pd.to_numeric(result["item_id"], errors="coerce").astype("Int64")
    if "item_category_id" in result.columns:
        result["item_category_id"] = pd.to_numeric(
            result["item_category_id"], errors="coerce"
        ).fillna(-1).astype(int)
    else:
        result["item_category_id"] = -1

    result["item_name"] = result["item_name"].map(
        lambda value: _normalise_text_value(value, UNKNOWN_ITEM_TEXT)
    )

    if categories is not None and not categories.empty:
        cat = categories.copy()
        if "item_category_id" not in cat.columns:
            raise ValueError("categories table must contain item_category_id")
        if "item_category_name" not in cat.columns:
            candidates = [column for column in cat.columns if column != "item_category_id"]
            cat["item_category_name"] = cat[candidates[0]] if candidates else UNKNOWN_TEXT
        cat["item_category_id"] = pd.to_numeric(cat["item_category_id"], errors="coerce").fillna(-1).astype(int)
        cat["item_category_name"] = cat["item_category_name"].map(
            lambda value: _normalise_text_value(value, UNKNOWN_TEXT)
        )
        result = result.merge(
            cat[["item_category_id", "item_category_name"]].drop_duplicates("item_category_id"),
            on="item_category_id",
            how="left",
        )
    elif "item_category_name" not in result.columns:
        result["item_category_name"] = UNKNOWN_TEXT

    result["item_category_name"] = result["item_category_name"].map(
        lambda value: _normalise_text_value(value, UNKNOWN_TEXT)
    )
    result["category_group"] = result["item_category_name"].str.split("-").str[0].str.strip()
    result["category_group"] = result["category_group"].map(
        lambda value: _normalise_text_value(value, UNKNOWN_TEXT)
    )
    result["item_label"] = result["item_id"].astype(str) + " — " + result["item_name"]
    result["category_label"] = (
        result["item_category_id"].astype(str) + " — " + result["item_category_name"]
    )
    return result[
        [
            "item_id",
            "item_name",
            "item_label",
            "item_category_id",
            "item_category_name",
            "category_group",
            "category_label",
        ]
    ].drop_duplicates("item_id")


def enrich_with_dimensions(
    df: pd.DataFrame,
    *,
    shop_dimension: pd.DataFrame | None = None,
    item_dimension: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join shop and item/category names into a forecast/evaluation table."""
    result = df.copy()

    if shop_dimension is not None and "shop_id" in result.columns:
        safe_cols = ["shop_id", "shop_name", "shop_label"]
        result = result.drop(columns=[c for c in safe_cols[1:] if c in result.columns], errors="ignore")
        result = result.merge(shop_dimension[safe_cols], on="shop_id", how="left")

    if item_dimension is not None and "item_id" in result.columns:
        safe_cols = [
            "item_id",
            "item_name",
            "item_label",
            "item_category_id",
            "item_category_name",
            "category_group",
            "category_label",
        ]
        overlapping = [c for c in safe_cols[1:] if c in result.columns]
        result = result.drop(columns=overlapping, errors="ignore")
        result = result.merge(item_dimension[safe_cols], on="item_id", how="left")

    text_columns = [
        "shop_name",
        "shop_label",
        "item_name",
        "item_label",
        "item_category_name",
        "category_group",
        "category_label",
        "segment_key",
        "segment_name",
        "price_tier",
        "demand_tier",
    ]
    result = clean_text_columns(result, text_columns)

    if "shop_id" in result.columns and "shop_name" not in result.columns:
        result["shop_name"] = UNKNOWN_SHOP_TEXT
    if "shop_id" in result.columns and "shop_label" not in result.columns:
        result["shop_label"] = result["shop_id"].astype(str) + " — " + result["shop_name"]

    result = clean_recency_columns(result, [column for column in ["recency"] if column in result.columns])
    return result


def clean_feature_frame_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    """Clean null-like metadata and sentinel recency before model-zoo training."""
    result = df.copy()
    result = clean_recency_columns(result, [column for column in ["recency"] if column in result.columns])
    for column in result.columns:
        if pd.api.types.is_object_dtype(result[column]):
            result[column] = result[column].map(lambda value: _normalise_text_value(value, UNKNOWN_TEXT))
    return result


def read_local_or_s3_csv(path_or_uri: str) -> pd.DataFrame:
    """Read a CSV from local path or s3:// URI."""
    if path_or_uri.startswith("s3://"):
        bucket, key = path_or_uri.replace("s3://", "", 1).split("/", 1)
        obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
        return pd.read_csv(obj["Body"])
    return pd.read_csv(Path(path_or_uri))


def read_local_or_s3_parquet(path_or_uri: str) -> pd.DataFrame:
    """Read a parquet file from local path or s3:// URI."""
    if path_or_uri.startswith("s3://"):
        bucket, key = path_or_uri.replace("s3://", "", 1).split("/", 1)
        obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
        return pd.read_parquet(BytesIO(obj["Body"].read()))
    return pd.read_parquet(Path(path_or_uri))
