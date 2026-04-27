"""Build monthly features for segmented sales forecasting.

Inputs:
- sales_train.csv
- test.csv
- optional items_en.csv with item_category_id
- optional item_categories_en.csv with category names
- optional shops_en.csv with shop names

Outputs:
- train.parquet
- valid.parquet
- inference_features.parquet
- inference_pairs.parquet
- product_segments.parquet
- shop_dimension.parquet
- metadata.json
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from modelops.config import FeatureConfig, NUMERIC_FEATURES
from modelops.io import read_csv, write_json, write_parquet
from modelops.segments import build_operational_segments, normalize_shops


def load_optional_csv(path: str | None) -> pd.DataFrame | None:
    """Read optional CSV path."""
    if not path:
        return None
    try:
        return read_csv(path)
    except FileNotFoundError:
        return None


def aggregate_monthly_sales(sales: pd.DataFrame, target_clip: float) -> pd.DataFrame:
    """Aggregate daily sales to monthly shop-item level."""
    sales = sales.copy()
    sales["date_block_num"] = sales["date_block_num"].astype(int)
    sales["shop_id"] = sales["shop_id"].astype(int)
    sales["item_id"] = sales["item_id"].astype(int)
    sales["item_cnt_day"] = pd.to_numeric(sales["item_cnt_day"], errors="coerce").fillna(0)
    sales["item_price"] = pd.to_numeric(sales["item_price"], errors="coerce").fillna(0)
    sales["revenue"] = sales["item_cnt_day"] * sales["item_price"]

    monthly = (
        sales.groupby(["date_block_num", "shop_id", "item_id"], as_index=False)
        .agg(
            target=("item_cnt_day", "sum"),
            price_mean=("item_price", "mean"),
            revenue=("revenue", "sum"),
        )
        .sort_values(["shop_id", "item_id", "date_block_num"])
    )
    monthly["target"] = monthly["target"].clip(lower=0, upper=target_clip).astype("float32")
    monthly["price_mean"] = monthly["price_mean"].clip(lower=0).astype("float32")
    return monthly


def make_dense_grid(monthly: pd.DataFrame, test: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Create shop-item-month grid for train/valid/inference.

    Historical months use observed pairs to keep memory reasonable. The official
    test pairs are also created for validation_block and inference_block so the
    holdout resembles the final Kaggle inference universe.
    """
    historical = monthly[["date_block_num", "shop_id", "item_id"]].drop_duplicates()

    test_pairs = test[["shop_id", "item_id"]].drop_duplicates().copy()

    validation_pairs = test_pairs.copy()
    validation_pairs["date_block_num"] = cfg.validation_block

    inference_pairs = test_pairs.copy()
    inference_pairs["date_block_num"] = cfg.inference_block

    grid = pd.concat([historical, validation_pairs, inference_pairs], ignore_index=True).drop_duplicates()
    grid = grid.merge(monthly, on=["date_block_num", "shop_id", "item_id"], how="left")
    grid["target"] = grid["target"].fillna(0).astype("float32")
    grid["price_mean"] = grid["price_mean"].fillna(0).astype("float32")
    return grid.sort_values(["shop_id", "item_id", "date_block_num"]).reset_index(drop=True)


def add_lag_features(grid: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling, recency and activity features."""
    df = grid.sort_values(["shop_id", "item_id", "date_block_num"]).copy()
    pair_group = df.groupby(["shop_id", "item_id"])["target"]

    for lag in [1, 2, 3, 6]:
        df[f"cnt_lag_{lag}"] = pair_group.shift(lag).fillna(0).astype("float32")

    shifted = pair_group.shift(1)
    df["mean_3"] = (
        shifted.groupby([df["shop_id"], df["item_id"]])
        .rolling(3)
        .mean()
        .reset_index(level=[0, 1], drop=True)
        .fillna(0)
        .astype("float32")
    )
    df["mean_6"] = (
        shifted.groupby([df["shop_id"], df["item_id"]])
        .rolling(6)
        .mean()
        .reset_index(level=[0, 1], drop=True)
        .fillna(0)
        .astype("float32")
    )

    shop_month = (
        df.groupby(["date_block_num", "shop_id"], as_index=False)["target"]
        .mean()
        .rename(columns={"target": "shop_month_mean"})
        .sort_values(["shop_id", "date_block_num"])
    )
    shop_month["shop_mean_3"] = (
        shop_month.groupby("shop_id")["shop_month_mean"]
        .transform(lambda s: s.shift(1).rolling(3).mean())
        .fillna(0)
        .astype("float32")
    )
    df = df.merge(
        shop_month[["date_block_num", "shop_id", "shop_mean_3"]],
        on=["date_block_num", "shop_id"],
        how="left",
    )

    item_month = (
        df.groupby(["date_block_num", "item_id"], as_index=False)["target"]
        .mean()
        .rename(columns={"target": "item_month_mean"})
        .sort_values(["item_id", "date_block_num"])
    )
    item_month["item_mean_3"] = (
        item_month.groupby("item_id")["item_month_mean"]
        .transform(lambda s: s.shift(1).rolling(3).mean())
        .fillna(0)
        .astype("float32")
    )
    df = df.merge(
        item_month[["date_block_num", "item_id", "item_mean_3"]],
        on=["date_block_num", "item_id"],
        how="left",
    )

    df["price_lag_1"] = (
        df.groupby(["shop_id", "item_id"])["price_mean"].shift(1).fillna(df["price_mean"]).astype("float32")
    )

    positive = df["target"] > 0
    last_sale = df["date_block_num"].where(positive).groupby([df["shop_id"], df["item_id"]]).ffill()
    df["recency"] = (df["date_block_num"] - last_sale).fillna(99).clip(0, 99).astype("float32")
    df["months_active"] = positive.groupby([df["shop_id"], df["item_id"]]).cumsum().astype("float32")
    df["is_new_pair"] = (df["months_active"] <= 1).astype("int8")
    df["month"] = (df["date_block_num"] % 12).astype("int8")

    return df


def add_segments_and_category_means(features: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    """Attach product segments and segment-level historical features."""
    df = features.merge(segments, on="item_id", how="left")
    df["item_category_id"] = df["item_category_id"].fillna(-1).astype(int)
    df["segment_key"] = df["segment_key"].fillna("unknown")
    df["segment_name"] = df["segment_name"].fillna(df["segment_key"])
    df["category_group"] = df["category_group"].fillna("unknown")

    cat_month = (
        df.groupby(["date_block_num", "segment_key"], as_index=False)["target"]
        .mean()
        .rename(columns={"target": "category_month_mean"})
        .sort_values(["segment_key", "date_block_num"])
    )
    cat_month["category_mean_3"] = (
        cat_month.groupby("segment_key")["category_month_mean"]
        .transform(lambda s: s.shift(1).rolling(3).mean())
        .fillna(0)
        .astype("float32")
    )
    df = df.merge(
        cat_month[["date_block_num", "segment_key", "category_mean_3"]],
        on=["date_block_num", "segment_key"],
        how="left",
    )
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0
    return df


def build_features(
    sales_path: str,
    test_path: str,
    output_dir: str,
    items_path: str | None = None,
    categories_path: str | None = None,
    shops_path: str | None = None,
    cfg: FeatureConfig | None = None,
) -> dict:
    """Build all feature tables."""
    cfg = cfg or FeatureConfig()
    sales = read_csv(sales_path)
    test = read_csv(test_path)
    items = load_optional_csv(items_path)
    categories = load_optional_csv(categories_path)
    shops = load_optional_csv(shops_path)

    monthly = aggregate_monthly_sales(sales, target_clip=cfg.target_clip)
    segments = build_operational_segments(monthly, items_df=items, categories_df=categories)
    shops_dim = normalize_shops(shops)

    grid = make_dense_grid(monthly, test, cfg)
    features = add_lag_features(grid)
    features = add_segments_and_category_means(features, segments)

    train = features[features["date_block_num"] < cfg.validation_block].copy()
    valid = features[features["date_block_num"] == cfg.validation_block].copy()
    inference = features[features["date_block_num"] == cfg.inference_block].copy()
    inference_pairs = test.copy()

    if "ID" not in inference_pairs.columns:
        inference_pairs = inference_pairs.reset_index().rename(columns={"index": "ID"})

    base = output_dir.rstrip("/")
    write_parquet(train, f"{base}/train.parquet")
    write_parquet(valid, f"{base}/valid.parquet")
    write_parquet(inference, f"{base}/inference_features.parquet")
    write_parquet(inference_pairs, f"{base}/inference_pairs.parquet")
    write_parquet(segments, f"{base}/product_segments.parquet")
    if shops_dim is not None:
        write_parquet(shops_dim, f"{base}/shop_dimension.parquet")

    metadata = {
        "sales_path": sales_path,
        "test_path": test_path,
        "items_path": items_path,
        "categories_path": categories_path,
        "shops_path": shops_path,
        "n_train_rows": int(len(train)),
        "n_valid_rows": int(len(valid)),
        "n_inference_rows": int(len(inference)),
        "n_segments": int(segments["segment_key"].nunique()),
        "validation_block": cfg.validation_block,
        "inference_block": cfg.inference_block,
        "feature_cols": NUMERIC_FEATURES,
        "segment_source_counts": segments["segment_source"].value_counts().to_dict(),
    }
    write_json(metadata, f"{base}/metadata.json")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sales-path", required=True)
    parser.add_argument("--test-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--items-path", default=None)
    parser.add_argument("--categories-path", default=None)
    parser.add_argument("--shops-path", default=None)
    parser.add_argument("--validation-block", type=int, default=33)
    parser.add_argument("--inference-block", type=int, default=34)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = FeatureConfig(
        validation_block=args.validation_block,
        inference_block=args.inference_block,
    )
    metadata = build_features(
        sales_path=args.sales_path,
        test_path=args.test_path,
        items_path=args.items_path,
        categories_path=args.categories_path,
        shops_path=args.shops_path,
        output_dir=args.output_dir,
        cfg=cfg,
    )
    print(metadata)


if __name__ == "__main__":
    main()
