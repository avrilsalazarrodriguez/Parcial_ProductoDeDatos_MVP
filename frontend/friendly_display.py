"""Friendly display helpers for the Streamlit forecast dashboard.

The model outputs use machine-friendly identifiers (`shop_id`, `item_id`,
`item_category_id`).  This module adds human-readable labels and helps the UI
show filters such as `2 — Store name` instead of just `2`.
"""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd
import streamlit as st

MISSING_DISPLAY = "Sin clasificar"


def clean_display_value(value: object, fallback: str = MISSING_DISPLAY) -> str:
    """Return a readable string for UI display."""
    if value is None or pd.isna(value):
        return fallback
    text = str(value).strip()
    if text.lower() in {"", "none", "null", "nan", "unknown", "<na>"}:
        return fallback
    return text


def first_int_from_label(label: str | int | None) -> int | None:
    """Parse the first integer from labels such as `2 — Shop name`."""
    if label is None:
        return None
    if isinstance(label, int):
        return label
    match = re.match(r"^\s*(-?\d+)", str(label))
    if not match:
        return None
    return int(match.group(1))


def ensure_friendly_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add label columns and clean null-like display values."""
    result = df.copy()

    if "shop_id" in result.columns:
        if "shop_name" not in result.columns:
            result["shop_name"] = "Tienda sin nombre"
        result["shop_name"] = result["shop_name"].map(lambda v: clean_display_value(v, "Tienda sin nombre"))
        result["shop_label"] = result["shop_id"].astype(str) + " — " + result["shop_name"]

    if "item_id" in result.columns:
        if "item_name" not in result.columns:
            result["item_name"] = "Producto sin nombre"
        result["item_name"] = result["item_name"].map(lambda v: clean_display_value(v, "Producto sin nombre"))
        result["item_label"] = result["item_id"].astype(str) + " — " + result["item_name"]

    if "item_category_id" in result.columns:
        if "item_category_name" not in result.columns:
            result["item_category_name"] = MISSING_DISPLAY
        result["item_category_name"] = result["item_category_name"].map(clean_display_value)
        result["category_label"] = (
            result["item_category_id"].astype(str) + " — " + result["item_category_name"]
        )

    for column in ["segment_key", "segment_name", "category_group", "price_tier", "demand_tier"]:
        if column in result.columns:
            result[column] = result[column].map(clean_display_value)

    if "recency" in result.columns:
        numeric_recency = pd.to_numeric(result["recency"], errors="coerce")
        missing_recency = numeric_recency.isna() | (numeric_recency >= 90) | numeric_recency.isin([-1, 99, 999])
        result["recency_label"] = missing_recency.map(
            {True: "Sin ventas recientes / sin historial", False: ""}
        )
        result.loc[~missing_recency, "recency_label"] = (
            numeric_recency.loc[~missing_recency].astype(int).astype(str) + " meses"
        )

    return result


def make_options(df: pd.DataFrame, label_col: str, include_all: bool = True) -> list[str]:
    """Build sorted selectbox options from a label column."""
    if label_col not in df.columns:
        return ["Todos"] if include_all else []
    labels = sorted(df[label_col].dropna().astype(str).unique().tolist())
    return (["Todos"] if include_all else []) + labels


def filter_by_label(df: pd.DataFrame, label: str, id_col: str) -> pd.DataFrame:
    """Filter a DataFrame by an id parsed from a friendly label."""
    if label == "Todos" or id_col not in df.columns:
        return df
    selected_id = first_int_from_label(label)
    if selected_id is None:
        return df
    return df.loc[df[id_col] == selected_id].copy()


def filter_by_values(df: pd.DataFrame, column: str, values: Iterable[str]) -> pd.DataFrame:
    """Filter a DataFrame by a multi-select of string values."""
    selected_values = list(values)
    if not selected_values or column not in df.columns:
        return df
    return df.loc[df[column].astype(str).isin(selected_values)].copy()


def friendly_forecast_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a forecast table with the most useful columns first."""
    result = ensure_friendly_columns(df)
    preferred = [
        "shop_id",
        "shop_name",
        "item_id",
        "item_name",
        "prediction",
        "model_id",
        "model_scope",
        "segment_key",
        "segment_name",
        "item_category_id",
        "item_category_name",
        "category_group",
        "price_tier",
        "demand_tier",
        "months_active",
        "recency",
        "recency_label",
    ]
    cols = [col for col in preferred if col in result.columns]
    remaining = [col for col in result.columns if col not in cols]
    return result[cols + remaining]


def render_forecast_filters(df: pd.DataFrame, key_prefix: str = "forecast") -> pd.DataFrame:
    """Render friendly filters for forecast tables and return the filtered data."""
    working = ensure_friendly_columns(df)
    with st.expander("Filtros amigables", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            shop_label = st.selectbox(
                "Tienda",
                make_options(working, "shop_label", include_all=True),
                key=f"{key_prefix}_shop",
            )
            category_label = st.selectbox(
                "Categoría",
                make_options(working, "category_label", include_all=True),
                key=f"{key_prefix}_category",
            )
        with col2:
            demand_options = make_options(working, "demand_tier", include_all=False)
            selected_demand = st.multiselect(
                "Tipo de demanda",
                demand_options,
                default=[],
                key=f"{key_prefix}_demand",
            )
            price_options = make_options(working, "price_tier", include_all=False)
            selected_price = st.multiselect(
                "Nivel de precio",
                price_options,
                default=[],
                key=f"{key_prefix}_price",
            )

    filtered = filter_by_label(working, shop_label, "shop_id")
    filtered = filter_by_label(filtered, category_label, "item_category_id")
    filtered = filter_by_values(filtered, "demand_tier", selected_demand)
    filtered = filter_by_values(filtered, "price_tier", selected_price)
    return filtered


def render_table(df: pd.DataFrame, max_rows: int = 1000) -> None:
    """Render a friendly table with Streamlit's current width API."""
    st.dataframe(friendly_forecast_columns(df).head(max_rows), width="stretch")
