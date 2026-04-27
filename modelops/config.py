"""Configuration helpers for ModelOps scripts.

The scripts can run locally or inside SageMaker Processing jobs. Paths can be
local paths or S3 URIs, depending on the runtime.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureConfig:
    """Feature engineering configuration for the 1C sales forecasting problem."""

    validation_block: int = 33
    inference_block: int = 34
    target_clip: float = 20.0
    min_segment_train_rows: int = 25_000
    min_segment_valid_rows: int = 1_000
    max_segment_models: int = 20
    random_state: int = 42


NUMERIC_FEATURES = [
    "shop_id",
    "item_id",
    "date_block_num",
    "item_category_id",
    "month",
    "cnt_lag_1",
    "cnt_lag_2",
    "cnt_lag_3",
    "cnt_lag_6",
    "mean_3",
    "mean_6",
    "shop_mean_3",
    "item_mean_3",
    "category_mean_3",
    "price_mean",
    "price_lag_1",
    "months_active",
    "recency",
    "is_new_pair",
]
