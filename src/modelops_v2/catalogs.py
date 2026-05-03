"""Catálogos y enriquecimiento de tablas para la UI.

Este módulo evita que la aplicación muestre valores crudos como ``None``,
``unknown`` o ``recency=99`` sin contexto. También agrega nombres de tienda,
producto, categoría, segmento y etiquetas legibles para filtros.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


NULL_TOKENS = {"", "none", "nan", "null", "unknown", "unk", "noce", "n/a", "na"}


@dataclass(frozen=True)
class CatalogBundle:
    """Catálogos disponibles para enriquecer salidas del modelo."""

    shops: pd.DataFrame
    items: pd.DataFrame
    categories: pd.DataFrame


def _candidate_paths(data_dir: Path, filenames: Iterable[str]) -> list[Path]:
    """Regresa rutas candidatas dentro de ``data_dir`` y su carpeta raw."""
    paths: list[Path] = []
    for filename in filenames:
        paths.append(data_dir / filename)
        paths.append(data_dir / "raw" / filename)
        paths.append(data_dir / "catalogs" / filename)
    return paths


def _read_first_existing(paths: list[Path]) -> pd.DataFrame:
    """Lee el primer CSV existente; si no existe, regresa DataFrame vacío."""
    for path in paths:
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


def clean_text_value(value: object, fallback: str) -> str:
    """Normaliza textos vacíos o tokens como ``None``/``unknown``."""
    if value is None:
        return fallback
    if isinstance(value, float) and np.isnan(value):
        return fallback
    text = str(value).strip()
    if text.lower() in NULL_TOKENS:
        return fallback
    return text


def _safe_int(value: object, fallback: int = -1) -> int:
    """Convierte ids a entero de forma tolerante."""
    try:
        if pd.isna(value):
            return fallback
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _build_shop_catalog(data_dir: Path) -> pd.DataFrame:
    shops = _read_first_existing(_candidate_paths(data_dir, ["shops_en.csv", "shops.csv"]))
    if shops.empty:
        return pd.DataFrame(columns=["shop_id", "shop_name", "shop_label"])

    if "shop_name" not in shops.columns:
        name_cols = [c for c in shops.columns if "name" in c.lower()]
        shops["shop_name"] = shops[name_cols[0]] if name_cols else ""

    shops = shops[["shop_id", "shop_name"]].copy()
    shops["shop_id"] = shops["shop_id"].apply(_safe_int)
    shops["shop_name"] = shops.apply(
        lambda row: clean_text_value(row["shop_name"], f"Tienda {row['shop_id']}"),
        axis=1,
    )
    shops["shop_label"] = shops.apply(lambda r: f"{r['shop_id']} — {r['shop_name']}", axis=1)
    return shops.drop_duplicates("shop_id")


def _extract_category_group(category_name: object, category_id: int) -> str:
    """Crea grupo ejecutivo de categoría a partir del nombre de categoría."""
    cleaned = clean_text_value(category_name, f"Categoría {category_id}")
    if " - " in cleaned:
        return cleaned.split(" - ", maxsplit=1)[0].strip()
    if "-" in cleaned:
        return cleaned.split("-", maxsplit=1)[0].strip()
    if ":" in cleaned:
        return cleaned.split(":", maxsplit=1)[0].strip()
    words = cleaned.split()
    return words[0] if words else f"Categoría {category_id}"


def _build_category_catalog(data_dir: Path) -> pd.DataFrame:
    categories = _read_first_existing(
        _candidate_paths(data_dir, ["item_categories_en.csv", "item_categories.csv"])
    )
    if categories.empty:
        return pd.DataFrame(
            columns=["item_category_id", "item_category_name", "category_group"]
        )

    if "item_category_name" not in categories.columns:
        name_cols = [c for c in categories.columns if "name" in c.lower()]
        categories["item_category_name"] = categories[name_cols[0]] if name_cols else ""

    categories = categories[["item_category_id", "item_category_name"]].copy()
    categories["item_category_id"] = categories["item_category_id"].apply(_safe_int)
    categories["item_category_name"] = categories.apply(
        lambda row: clean_text_value(
            row["item_category_name"], f"Categoría {row['item_category_id']}"
        ),
        axis=1,
    )
    categories["category_group"] = categories.apply(
        lambda row: _extract_category_group(row["item_category_name"], row["item_category_id"]),
        axis=1,
    )
    return categories.drop_duplicates("item_category_id")


def _build_item_catalog(data_dir: Path, categories: pd.DataFrame) -> pd.DataFrame:
    items = _read_first_existing(_candidate_paths(data_dir, ["items_en.csv", "items.csv"]))
    if items.empty:
        return pd.DataFrame(
            columns=[
                "item_id",
                "item_name",
                "item_category_id",
                "item_category_name",
                "category_group",
                "item_label",
            ]
        )

    if "item_name" not in items.columns:
        name_cols = [c for c in items.columns if "name" in c.lower()]
        items["item_name"] = items[name_cols[0]] if name_cols else ""

    if "item_category_id" not in items.columns:
        items["item_category_id"] = -1

    items = items[["item_id", "item_name", "item_category_id"]].copy()
    items["item_id"] = items["item_id"].apply(_safe_int)
    items["item_category_id"] = items["item_category_id"].apply(_safe_int)
    items["item_name"] = items.apply(
        lambda row: clean_text_value(row["item_name"], f"Producto {row['item_id']}"),
        axis=1,
    )

    if not categories.empty:
        items = items.merge(categories, on="item_category_id", how="left")
    else:
        items["item_category_name"] = ""
        items["category_group"] = ""

    items["item_category_name"] = items.apply(
        lambda row: clean_text_value(
            row.get("item_category_name"), f"Categoría {row['item_category_id']}"
        ),
        axis=1,
    )
    items["category_group"] = items.apply(
        lambda row: clean_text_value(row.get("category_group"), "Sin grupo"),
        axis=1,
    )
    items["item_label"] = items.apply(lambda r: f"{r['item_id']} — {r['item_name']}", axis=1)
    return items.drop_duplicates("item_id")


def build_catalog_bundle(data_dir: str | Path = "data") -> CatalogBundle:
    """Construye catálogos de tienda, producto y categoría desde archivos CSV."""
    data_path = Path(data_dir)
    shops = _build_shop_catalog(data_path)
    categories = _build_category_catalog(data_path)
    items = _build_item_catalog(data_path, categories)
    return CatalogBundle(shops=shops, items=items, categories=categories)


def _ensure_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Garantiza que los ids principales existan y sean enteros cuando sea posible."""
    result = df.copy()
    for col in ["shop_id", "item_id", "item_category_id"]:
        if col in result.columns:
            result[col] = result[col].apply(_safe_int)
    return result


def add_recency_context(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega explicación de recency y estatus operacional del producto."""
    result = df.copy()

    if "recency" not in result.columns:
        result["recency"] = np.nan

    if "months_active" not in result.columns:
        result["months_active"] = np.nan

    def recency_display(value: object) -> str:
        recency = _safe_int(value, fallback=99)
        if recency >= 90:
            return "Sin ventas recientes / sin historial suficiente"
        if recency == 0:
            return "Vendido en el último mes observado"
        return f"Última venta hace {recency} mes(es)"

    def product_status(row: pd.Series) -> str:
        recency = _safe_int(row.get("recency"), fallback=99)
        months_active = _safe_int(row.get("months_active"), fallback=0)
        if recency >= 90 and months_active <= 0:
            return "Cold start o sin historial"
        if recency >= 90:
            return "Sin ventas recientes"
        if months_active <= 1:
            return "Historial corto"
        return "Activo"

    result["recency_display"] = result["recency"].apply(recency_display)
    result["product_status"] = result.apply(product_status, axis=1)
    return result


def add_segment_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega nombres legibles para segmentos del modelo."""
    result = df.copy()

    if "segment_key" not in result.columns:
        if "item_category_id" in result.columns:
            result["segment_key"] = "cat_" + result["item_category_id"].astype(str)
        else:
            result["segment_key"] = "global"

    if "segment_name" not in result.columns:
        if "item_category_name" in result.columns:
            result["segment_name"] = result["item_category_name"]
        elif "category_group" in result.columns:
            result["segment_name"] = result["category_group"]
        else:
            result["segment_name"] = "Modelo global"

    result["segment_key"] = result["segment_key"].apply(lambda x: clean_text_value(x, "global"))
    result["segment_name"] = result["segment_name"].apply(
        lambda x: clean_text_value(x, "Modelo global")
    )
    result["segment_label"] = result.apply(
        lambda r: f"{r['segment_key']} — {r['segment_name']}", axis=1
    )
    return result


def add_model_scope_description(df: pd.DataFrame) -> pd.DataFrame:
    """Explica la columna ``model_scope`` para negocio."""
    result = df.copy()
    if "model_scope" not in result.columns:
        result["model_scope"] = "global"

    def describe(scope: object) -> str:
        text = clean_text_value(scope, "global")
        if text.startswith("segment:"):
            return "Modelo especializado para el segmento/categoría del producto."
        if text == "global":
            return "Modelo global entrenado con todos los pares tienda-producto."
        if text == "naive_baseline":
            return "Baseline naive: usa ventas recientes como referencia."
        if text == "cold_start_fallback":
            return "Fallback para producto sin historial suficiente."
        if text == "item_mean_fallback":
            return "Fallback basado en promedio histórico del producto."
        return "Regla o modelo usado para generar esa predicción."

    result["model_scope"] = result["model_scope"].apply(lambda x: clean_text_value(x, "global"))
    result["model_scope_explanation"] = result["model_scope"].apply(describe)
    return result


def add_catalog_metadata(df: pd.DataFrame, data_dir: str | Path = "data") -> pd.DataFrame:
    """Enriquece una tabla con nombres y etiquetas legibles.

    La función es tolerante a catálogos faltantes. Si no encuentra nombres reales,
    genera nombres seguros como ``Tienda 2`` y ``Producto 5037`` para que la UI no
    muestre ``None`` o ``unknown``.
    """
    result = _ensure_id_columns(df)
    catalogs = build_catalog_bundle(data_dir)

    if "shop_id" in result.columns:
        if not catalogs.shops.empty:
            result = result.merge(catalogs.shops, on="shop_id", how="left")
        if "shop_name" not in result.columns:
            result["shop_name"] = ""
        result["shop_name"] = result.apply(
            lambda row: clean_text_value(row.get("shop_name"), f"Tienda {row['shop_id']}"),
            axis=1,
        )
        result["shop_label"] = result.apply(lambda r: f"{r['shop_id']} — {r['shop_name']}", axis=1)

    if "item_id" in result.columns:
        if not catalogs.items.empty:
            merge_cols = [
                "item_id",
                "item_name",
                "item_category_id",
                "item_category_name",
                "category_group",
                "item_label",
            ]
            merge_cols = [c for c in merge_cols if c in catalogs.items.columns]
            result = result.merge(catalogs.items[merge_cols], on="item_id", how="left", suffixes=("", "_catalog"))

            if "item_category_id_catalog" in result.columns:
                if "item_category_id" in result.columns:
                    result["item_category_id"] = result["item_category_id"].where(
                        result["item_category_id"].notna(), result["item_category_id_catalog"]
                    )
                else:
                    result["item_category_id"] = result["item_category_id_catalog"]
                result = result.drop(columns=["item_category_id_catalog"])

        if "item_name" not in result.columns:
            result["item_name"] = ""
        result["item_name"] = result.apply(
            lambda row: clean_text_value(row.get("item_name"), f"Producto {row['item_id']}"),
            axis=1,
        )
        if "item_label" not in result.columns:
            result["item_label"] = ""
        result["item_label"] = result.apply(lambda r: f"{r['item_id']} — {r['item_name']}", axis=1)

    if "item_category_id" in result.columns:
        result["item_category_id"] = result["item_category_id"].apply(_safe_int)
        if "item_category_name" not in result.columns:
            result["item_category_name"] = ""
        if "category_group" not in result.columns:
            result["category_group"] = ""
        result["item_category_name"] = result.apply(
            lambda row: clean_text_value(
                row.get("item_category_name"), f"Categoría {row['item_category_id']}"
            ),
            axis=1,
        )
        result["category_group"] = result.apply(
            lambda row: clean_text_value(row.get("category_group"), "Sin grupo"),
            axis=1,
        )

    result = add_recency_context(result)
    result = add_segment_labels(result)
    result = add_model_scope_description(result)
    return result


def preferred_forecast_columns(df: pd.DataFrame) -> list[str]:
    """Columnas recomendadas para tablas de pronóstico en Streamlit."""
    desired = [
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
        "segment_label",
        "prediction",
        "model_scope",
        "model_scope_explanation",
        "recency",
        "recency_display",
        "product_status",
        "months_active",
        "demand_tier",
        "price_tier",
    ]
    return [col for col in desired if col in df.columns]
