"""Catalog and display-label utilities for local ModelOps outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

NULL_LIKE_VALUES = {"", "none", "null", "nan", "na", "n/a", "unknown", "noce"}
RECENCY_SENTINEL = 99


def clean_text(value: object, default: str = "Sin información") -> str:
    if value is None or pd.isna(value):
        return default
    text = str(value).strip()
    if text.lower() in NULL_LIKE_VALUES:
        return default
    return text


def read_first_csv(paths: list[Path]) -> pd.DataFrame:
    for path in paths:
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


def first_text_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    text_cols = [col for col in df.columns if df[col].dtype == "object"]
    return text_cols[0] if text_cols else None


def category_group_from_name(value: object) -> str:
    text = clean_text(value, "Sin grupo")
    if " - " in text:
        return text.split(" - ")[0].strip()
    if "(" in text:
        return text.split("(")[0].strip()
    return text


def load_catalogs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_dir = data_dir / "raw"
    shops = read_first_csv([raw_dir / "shops_en.csv", raw_dir / "shops.csv"])
    items = read_first_csv([raw_dir / "items_en.csv", raw_dir / "items.csv"])
    categories = read_first_csv([raw_dir / "item_categories_en.csv", raw_dir / "item_categories.csv"])

    if not categories.empty and "item_category_id" in categories.columns:
        name_col = first_text_col(
            categories,
            ["item_category_name", "item_category_name_en", "category_name", "name"],
        )
        categories = categories.copy()
        categories["item_category_id"] = pd.to_numeric(categories["item_category_id"], errors="coerce").astype("Int64")
        if name_col:
            categories["item_category_name"] = categories[name_col].map(lambda x: clean_text(x, "Sin categoría"))
        else:
            categories["item_category_name"] = "Sin categoría"
        categories["category_group"] = categories["item_category_name"].map(category_group_from_name)
        categories = categories[["item_category_id", "item_category_name", "category_group"]].drop_duplicates("item_category_id")

    if not shops.empty and "shop_id" in shops.columns:
        name_col = first_text_col(shops, ["shop_name", "shop_name_en", "name", "shop"])
        shops = shops.copy()
        shops["shop_id"] = pd.to_numeric(shops["shop_id"], errors="coerce").astype("Int64")
        if name_col:
            shops["shop_name"] = shops[name_col].map(lambda x: clean_text(x, "Tienda sin nombre"))
        else:
            shops["shop_name"] = shops["shop_id"].map(lambda x: f"Tienda {int(x)}" if pd.notna(x) else "Tienda sin ID")
        shops["shop_label"] = shops.apply(
            lambda row: f"{int(row['shop_id'])} — {row['shop_name']}" if pd.notna(row["shop_id"]) else row["shop_name"],
            axis=1,
        )
        shops = shops[["shop_id", "shop_name", "shop_label"]].drop_duplicates("shop_id")

    if not items.empty and "item_id" in items.columns:
        name_col = first_text_col(items, ["item_name", "item_name_en", "name", "item"])
        items = items.copy()
        items["item_id"] = pd.to_numeric(items["item_id"], errors="coerce").astype("Int64")
        if name_col:
            items["item_name"] = items[name_col].map(lambda x: clean_text(x, "Producto sin nombre"))
        else:
            items["item_name"] = items["item_id"].map(lambda x: f"Producto {int(x)}" if pd.notna(x) else "Producto sin ID")
        if "item_category_id" in items.columns:
            items["item_category_id"] = pd.to_numeric(items["item_category_id"], errors="coerce").astype("Int64")
            if not categories.empty:
                items = items.merge(categories, on="item_category_id", how="left")
        items["item_label"] = items.apply(
            lambda row: f"{int(row['item_id'])} — {row['item_name']}" if pd.notna(row["item_id"]) else row["item_name"],
            axis=1,
        )
        keep = ["item_id", "item_name", "item_label", "item_category_id", "item_category_name", "category_group"]
        items = items[[c for c in keep if c in items.columns]].drop_duplicates("item_id")

    return shops, items, categories


def attach_labels(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    out = df.copy()
    shops, items, _ = load_catalogs(data_dir)
    stale = ["shop_name", "shop_label", "item_name", "item_label", "item_category_name", "category_group", "category_label"]
    out = out.drop(columns=[c for c in stale if c in out.columns], errors="ignore")

    if "shop_id" in out.columns:
        out["shop_id"] = pd.to_numeric(out["shop_id"], errors="coerce").astype("Int64")
        if not shops.empty:
            out = out.merge(shops, on="shop_id", how="left")
        out["shop_name"] = out.get("shop_name", pd.Series(index=out.index, dtype="object")).map(
            lambda x: clean_text(x, "Tienda sin nombre")
        )
        mask = out["shop_name"].eq("Tienda sin nombre") & out["shop_id"].notna()
        out.loc[mask, "shop_name"] = "Tienda " + out.loc[mask, "shop_id"].astype(int).astype(str)
        out["shop_label"] = out.apply(
            lambda row: f"{int(row['shop_id'])} — {row['shop_name']}" if pd.notna(row["shop_id"]) else row["shop_name"],
            axis=1,
        )

    if "item_id" in out.columns:
        out["item_id"] = pd.to_numeric(out["item_id"], errors="coerce").astype("Int64")
        if not items.empty:
            merge_cols = [c for c in items.columns if c != "item_category_id" or "item_category_id" not in out.columns]
            if "item_id" not in merge_cols:
                merge_cols.insert(0, "item_id")
            out = out.merge(items[merge_cols].drop_duplicates("item_id"), on="item_id", how="left")
        out["item_name"] = out.get("item_name", pd.Series(index=out.index, dtype="object")).map(
            lambda x: clean_text(x, "Producto sin nombre")
        )
        mask = out["item_name"].eq("Producto sin nombre") & out["item_id"].notna()
        out.loc[mask, "item_name"] = "Producto " + out.loc[mask, "item_id"].astype(int).astype(str)
        out["item_label"] = out.apply(
            lambda row: f"{int(row['item_id'])} — {row['item_name']}" if pd.notna(row["item_id"]) else row["item_name"],
            axis=1,
        )

    return clean_metadata(out)


def clean_metadata(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    defaults = {
        "segment_key": "sin_segmento",
        "segment_name": "Sin segmento",
        "segment_source": "Sin fuente",
        "item_category_name": "Sin categoría",
        "category_group": "Sin grupo",
        "price_tier": "Sin precio",
        "demand_tier": "Sin historial",
        "model_scope": "global",
    }
    for col, default in defaults.items():
        if col in out.columns:
            out[col] = out[col].map(lambda x, default=default: clean_text(x, default))

    if "item_category_id" in out.columns:
        cat = pd.to_numeric(out["item_category_id"], errors="coerce")
        out["item_category_id"] = cat.fillna(-1).astype(int)
        out["metadata_status"] = np.where(out["item_category_id"] >= 0, "metadata_ok", "metadata_missing")
        if "item_category_name" not in out.columns:
            out["item_category_name"] = "Sin categoría"
        out["category_label"] = out.apply(
            lambda row: f"{int(row['item_category_id'])} — {row['item_category_name']}" if int(row["item_category_id"]) >= 0 else "Sin categoría",
            axis=1,
        )

    if "recency" in out.columns:
        recency = pd.to_numeric(out["recency"], errors="coerce")
        out["recency_missing"] = recency.isna() | (recency >= RECENCY_SENTINEL)
        out["recency_clean"] = recency.mask(out["recency_missing"], np.nan)
        out["recency_label"] = out["recency_clean"].map(
            lambda x: "Sin ventas recientes / sin historial suficiente" if pd.isna(x) else f"{int(x)} meses"
        )
    return out
