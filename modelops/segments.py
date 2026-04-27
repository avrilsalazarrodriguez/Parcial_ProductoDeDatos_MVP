"""Segmentation logic for product-level forecasting.

Interpretation for the exam:
- The dataset is retail sales, not a demographic population problem.
- The correct subsegments are product categories, shops, or operational product
  groups derived from demand/price history.
- With the translated Kaggle files, `item_category_id` and
  `item_category_name` become the primary product segment.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd


def safe_qcut(series: pd.Series, labels: list[str]) -> pd.Series:
    """Robust qcut that works when values repeat."""
    return pd.qcut(series.rank(method="first"), q=len(labels), labels=labels)


def normalize_items(items_df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Normalize item metadata from items.csv/items_en.csv."""
    if items_df is None:
        return None
    df = items_df.copy()
    if "item_id" not in df.columns:
        return None
    if "item_category_id" not in df.columns:
        df["item_category_id"] = -1
    if "item_name" not in df.columns:
        df["item_name"] = ""
    return df[["item_id", "item_name", "item_category_id"]].drop_duplicates("item_id")


def normalize_categories(categories_df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Normalize item category metadata from item_categories.csv/item_categories_en.csv."""
    if categories_df is None:
        return None
    df = categories_df.copy()
    if "item_category_id" not in df.columns:
        return None
    if "item_category_name" not in df.columns:
        df["item_category_name"] = "unknown"
    df["category_group"] = df["item_category_name"].map(extract_category_group)
    return df[["item_category_id", "item_category_name", "category_group"]].drop_duplicates(
        "item_category_id"
    )


def normalize_shops(shops_df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Normalize shop metadata from shops.csv/shops_en.csv."""
    if shops_df is None:
        return None
    df = shops_df.copy()
    if "shop_id" not in df.columns:
        return None
    if "shop_name" not in df.columns:
        df["shop_name"] = ""
    df["shop_city_proxy"] = df["shop_name"].fillna("unknown").str.split().str[0].fillna("unknown")
    return df[["shop_id", "shop_name", "shop_city_proxy"]].drop_duplicates("shop_id")


def extract_category_group(category_name: str) -> str:
    """Create a broad, human-readable category group from translated names.

    Examples:
    - "Accessories - PS4" -> "Accessories"
    - "PC - Headsets / Headphones" -> "PC"
    """
    if not isinstance(category_name, str) or not category_name.strip():
        return "unknown"
    raw = re.split(r"\s[-–—]\s", category_name.strip(), maxsplit=1)[0]
    raw = raw.strip()
    return raw if raw else "unknown"


def build_operational_segments(
    monthly: pd.DataFrame,
    items_df: pd.DataFrame | None = None,
    categories_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create a product dimension with category metadata and a segment key.

    If translated Kaggle metadata is available, `item_category_id` becomes the
    primary product segment. Otherwise, a derived segment is created from
    historical demand/value/price.
    """
    item_meta = normalize_items(items_df)
    category_meta = normalize_categories(categories_df)

    item_stats = (
        monthly.groupby("item_id", as_index=False)
        .agg(
            total_units=("target", "sum"),
            avg_units=("target", "mean"),
            months_active=("target", lambda s: int((s > 0).sum())),
            avg_price=("price_mean", "mean"),
            last_block=("date_block_num", "max"),
        )
    )
    item_stats["total_units"] = item_stats["total_units"].clip(lower=0)
    item_stats["avg_units"] = item_stats["avg_units"].clip(lower=0)
    item_stats["avg_price"] = item_stats["avg_price"].fillna(0).clip(lower=0)

    if item_meta is not None:
        item_stats = item_stats.merge(item_meta, on="item_id", how="left")
    else:
        item_stats["item_name"] = ""
        item_stats["item_category_id"] = -1

    item_stats["item_category_id"] = item_stats["item_category_id"].fillna(-1).astype(int)

    if category_meta is not None:
        item_stats = item_stats.merge(category_meta, on="item_category_id", how="left")
    else:
        item_stats["item_category_name"] = "unknown"
        item_stats["category_group"] = "unknown"

    item_stats["item_category_name"] = item_stats["item_category_name"].fillna("unknown")
    item_stats["category_group"] = item_stats["category_group"].fillna("unknown")
    item_stats["price_tier"] = safe_qcut(item_stats["avg_price"], ["low", "mid", "high"]).astype(str)

    item_stats["demand_tier"] = np.select(
        [
            (item_stats["months_active"] >= 18) & (item_stats["avg_units"] >= 5),
            (item_stats["months_active"] >= 6) & (item_stats["avg_units"] >= 1),
            item_stats["months_active"] >= 2,
        ],
        ["core", "seasonal", "intermittent"],
        default="sparse",
    )

    if (item_stats["item_category_id"] >= 0).any():
        # Category-level segment: better for business/readability and aligned with Kaggle metadata.
        item_stats["segment_key"] = "cat_" + item_stats["item_category_id"].astype(str)
        item_stats["segment_name"] = item_stats["item_category_name"]
        item_stats["segment_source"] = "item_category_id"
    else:
        # Fallback when category metadata is unavailable.
        item_stats["segment_key"] = (
            item_stats["demand_tier"].astype(str) + "_" + item_stats["price_tier"].astype(str)
        )
        item_stats["segment_name"] = item_stats["segment_key"]
        item_stats["segment_source"] = "derived_operational_group"

    return item_stats[
        [
            "item_id",
            "item_name",
            "item_category_id",
            "item_category_name",
            "category_group",
            "segment_key",
            "segment_name",
            "segment_source",
            "price_tier",
            "demand_tier",
            "months_active",
            "avg_units",
            "avg_price",
        ]
    ]
