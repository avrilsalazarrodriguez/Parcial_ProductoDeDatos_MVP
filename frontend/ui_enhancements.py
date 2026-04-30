"""Incremental UI helpers for the existing Streamlit app.

The goal of this module is not to redesign the app. It adds business-friendly
labels, cleaner metric rendering, model ranking by RMSE, log-scale forecast
visualization and dual selectors where users can either choose from a dropdown
or type ids directly.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
RAW_DIR = DATA_DIR / "raw"

NULL_LIKE_VALUES = {"", "none", "null", "nan", "na", "n/a", "unknown", "noce", "sin dato"}
RECENCY_SENTINEL = 99
# Fallback names for the public 1C/Kaggle shop catalog. Used only when a
# previous preprocessing step replaced real names with generic labels like
# "Tienda 2".
DEFAULT_SHOP_NAMES = {
    0: "!Yakutsk Ordzhonikidze, 56 fran", 1: "!Yakutsk TC Central fran",
    2: "Adygea TC Mega", 3: "Balashikha TRC October-Kinomir",
    4: "Volzhsky TC Volga Mall", 5: "Vologda SEC Marmelad",
    6: "Voronezh Plekhanovskaya 13", 7: "Voronezh TRC Maksimir",
    8: "Voronezh TRC City-Park Grad", 9: "Outbound Trade",
    10: "Zhukovsky st. Chkalov 39m", 11: "Zhukovsky st. Chkalov 39m²",
    12: "Online shop emergencies", 13: "Kazan TC Behetle",
    14: "Kazan TC ParkHouse II", 15: "Kaluga TRC XXI century",
    16: "Kolomna TC Rio", 17: "Krasnoyarsk TC Vzletka Plaza",
    18: "Krasnoyarsk TC June", 19: "Kursk TC Pushkinsky",
    20: "Moscow Sale", 21: "Moscow MTRC Afi Mall", 22: "Moscow Shop C21",
    23: "Moscow TC Budenovskiy pav. A2", 24: "Moscow TC Budenovskiy pav. K7",
    25: "Moscow TRC Atrium", 26: "Moscow TC Areal Belyaevo",
    27: "Moscow TC MEGA Belaya Dacha II", 28: "Moscow TC MEGA Teply Stan II",
    29: "Moscow TC New Century Novokosino", 30: "Moscow TC Perlovskiy",
    31: "Moscow TC Semenovskiy", 32: "Moscow TC Serebryany Dom",
    33: "Mytishchi TRK XL-3", 34: "N. Novgorod TRC RIO",
    35: "N. Novgorod TRC Fantasy", 36: "Novosibirsk SEC Gallery Novosibirsk",
    37: "Novosibirsk TC Mega", 38: "Omsk TC Mega",
    39: "Rostov-on-Don TRK Megacenter Horizont",
    40: "Rostov-on-Don TRK Megacenter Horizont Ostrovnoy",
    41: "Rostov-on-Don TC Mega", 42: "SPb TC Nevsky Center",
    43: "SPb TK Sennaya", 44: "Samara TC Melody", 45: "Samara TC ParkHouse",
    46: "Sergiev Posad TC 7Ya", 47: "Surgut SEC City Mall",
    48: "Tomsk SEC Emerald City", 49: "Tyumen SEC Crystal",
    50: "Tyumen TC Goodwin", 51: "Tyumen TC Green Coast", 52: "Ufa TC Central",
    53: "Ufa TC Family 2", 54: "Khimki TC Mega",
    55: "Digital warehouse 1C-Online", 56: "Chekhov SEC Carnival",
    57: "Yakutsk Ordzhonikidze, 56", 58: "Yakutsk TC Central",
    59: "Yaroslavl TC Altair",
}


def _is_generic_shop_name(value: object, shop_id: object | None = None) -> bool:
    text = clean_text(value, "").lower()
    if not text:
        return True
    if shop_id is not None and pd.notna(shop_id):
        return text in {f"tienda {int(shop_id)}", f"shop {int(shop_id)}"}
    return bool(re.match(r"^(tienda|shop)\s+\d+$", text))


def clean_text(value: object, default: str = "Sin información") -> str:
    """Return a display-safe text value for tables and filters."""
    if value is None or pd.isna(value):
        return default
    text = str(value).strip()
    if text.lower() in NULL_LIKE_VALUES:
        return default
    return text


def _read_existing_csv(paths: list[Path]) -> pd.DataFrame:
    for path in paths:
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(ttl=900)
def load_catalogs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load shops/items/categories from the local data/raw folder.

    The repo has used both translated names (`shops_en.csv`) and original names
    (`shops.csv`) across different branches, so this function tries both.
    """
    shops = _read_existing_csv(
        [RAW_DIR / "shops_en.csv", RAW_DIR / "shops.csv", DATA_DIR / "raw" / "shops_en.csv"]
    )
    items = _read_existing_csv(
        [RAW_DIR / "items_en.csv", RAW_DIR / "items.csv", DATA_DIR / "raw" / "items_en.csv"]
    )
    categories = _read_existing_csv(
        [
            RAW_DIR / "item_categories_en.csv",
            RAW_DIR / "item_categories.csv",
            DATA_DIR / "raw" / "item_categories_en.csv",
        ]
    )

    if not shops.empty:
        shop_name_col = _first_existing_col(
            shops,
            ["shop_name", "shop_name_en", "name", "shop", "shop_name_translated"],
        )
        shops = shops.copy()
        shops["shop_id"] = pd.to_numeric(shops["shop_id"], errors="coerce").astype("Int64")
        shops["shop_name"] = shops.apply(
            lambda row: DEFAULT_SHOP_NAMES.get(int(row["shop_id"]), clean_text(row[shop_name_col], "Tienda sin nombre"))
            if pd.notna(row["shop_id"]) and _is_generic_shop_name(row[shop_name_col], row["shop_id"])
            else clean_text(row[shop_name_col], "Tienda sin nombre"),
            axis=1,
        )
        shops["shop_label"] = shops.apply(
            lambda row: f"{int(row['shop_id'])} — {row['shop_name']}"
            if pd.notna(row["shop_id"])
            else str(row["shop_name"]),
            axis=1,
        )
        shops = shops[["shop_id", "shop_name", "shop_label"]].drop_duplicates("shop_id")

    if not categories.empty:
        cat_name_col = _first_existing_col(
            categories,
            ["item_category_name", "item_category_name_en", "category_name", "name"],
        )
        categories = categories.copy()
        categories["item_category_id"] = pd.to_numeric(
            categories["item_category_id"], errors="coerce"
        ).astype("Int64")
        categories["item_category_name"] = categories[cat_name_col].map(
            lambda value: clean_text(value, "Sin categoría")
        )
        categories["category_group"] = categories["item_category_name"].map(_category_group_from_name)
        categories = categories[
            ["item_category_id", "item_category_name", "category_group"]
        ].drop_duplicates("item_category_id")

    if not items.empty:
        item_name_col = _first_existing_col(
            items,
            ["item_name", "item_name_en", "name", "item", "item_name_translated"],
        )
        items = items.copy()
        items["item_id"] = pd.to_numeric(items["item_id"], errors="coerce").astype("Int64")
        items["item_name"] = items[item_name_col].map(lambda value: clean_text(value, "Producto sin nombre"))
        if not categories.empty and "item_category_id" in items.columns:
            items["item_category_id"] = pd.to_numeric(
                items["item_category_id"], errors="coerce"
            ).astype("Int64")
            items = items.merge(categories, on="item_category_id", how="left")
        items["item_label"] = items.apply(
            lambda row: f"{int(row['item_id'])} — {row['item_name']}"
            if pd.notna(row["item_id"])
            else str(row["item_name"]),
            axis=1,
        )
        keep_cols = [
            "item_id",
            "item_name",
            "item_label",
            "item_category_id",
            "item_category_name",
            "category_group",
        ]
        items = items[[col for col in keep_cols if col in items.columns]].drop_duplicates("item_id")

    return shops, items, categories


def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    object_cols = [col for col in df.columns if df[col].dtype == "object"]
    if object_cols:
        return object_cols[0]
    raise ValueError(f"No text column found. Columns={list(df.columns)}")


def _category_group_from_name(value: object) -> str:
    text = clean_text(value, "Sin grupo")
    if " - " in text:
        return text.split(" - ")[0].strip()
    if "(" in text:
        return text.split("(")[0].strip()
    return text


def attach_catalog_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Attach real shop/product/category names without changing the app design."""
    out = df.copy()
    shops, items, _ = load_catalogs()

    # Avoid stale generic labels from earlier overlays, e.g. "Tienda 2".
    stale_cols = [
        "shop_name",
        "shop_label",
        "item_name",
        "item_label",
        "category_label",
        "item_category_name",
        "category_group",
    ]
    out = out.drop(columns=[col for col in stale_cols if col in out.columns], errors="ignore")

    if "shop_id" in out.columns:
        out["shop_id"] = pd.to_numeric(out["shop_id"], errors="coerce").astype("Int64")
        if not shops.empty:
            out = out.merge(shops, on="shop_id", how="left")
        out["shop_name"] = out.get("shop_name", pd.Series(index=out.index, dtype="object")).map(
            lambda value: clean_text(value, "Tienda sin nombre")
        )
        missing_shop_mask = out["shop_id"].notna() & out.apply(
            lambda row: _is_generic_shop_name(row.get("shop_name"), row.get("shop_id")),
            axis=1,
        )
        out.loc[missing_shop_mask, "shop_name"] = out.loc[missing_shop_mask, "shop_id"].astype(int).map(
            lambda sid: DEFAULT_SHOP_NAMES.get(sid, f"Tienda {sid}")
        )
        out["shop_label"] = out.apply(
            lambda row: f"{int(row['shop_id'])} — {row['shop_name']}"
            if pd.notna(row["shop_id"])
            else str(row["shop_name"]),
            axis=1,
        )

    if "item_id" in out.columns:
        out["item_id"] = pd.to_numeric(out["item_id"], errors="coerce").astype("Int64")
        if not items.empty:
            # Keep existing item_category_id if present, but prefer item dimension names.
            item_merge_cols = [col for col in items.columns if col != "item_category_id" or "item_category_id" not in out.columns]
            if "item_id" not in item_merge_cols:
                item_merge_cols.insert(0, "item_id")
            out = out.merge(items[item_merge_cols].drop_duplicates("item_id"), on="item_id", how="left")
        out["item_name"] = out.get("item_name", pd.Series(index=out.index, dtype="object")).map(
            lambda value: clean_text(value, "Producto sin nombre")
        )
        out.loc[out["item_name"].eq("Producto sin nombre") & out["item_id"].notna(), "item_name"] = (
            "Producto " + out.loc[out["item_name"].eq("Producto sin nombre") & out["item_id"].notna(), "item_id"].astype(int).astype(str)
        )
        out["item_label"] = out.apply(
            lambda row: f"{int(row['item_id'])} — {row['item_name']}"
            if pd.notna(row["item_id"])
            else str(row["item_name"]),
            axis=1,
        )

    out = clean_business_metadata(out)
    return out


def clean_business_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Clean null-like metadata and add recency labels for display."""
    out = df.copy()
    for col, default in {
        "segment_key": "sin_segmento",
        "segment_name": "Sin segmento",
        "segment_source": "Sin fuente",
        "item_category_name": "Sin categoría",
        "category_group": "Sin grupo",
        "price_tier": "Sin precio",
        "demand_tier": "Sin historial",
        "model_scope": "global",
    }.items():
        if col in out.columns:
            out[col] = out[col].map(lambda value, default=default: clean_text(value, default))

    if "item_category_id" in out.columns:
        cat = pd.to_numeric(out["item_category_id"], errors="coerce")
        out["has_category_metadata"] = cat.notna() & (cat >= 0)
        out["item_category_id"] = cat.fillna(-1).astype(int)
        if "item_category_name" not in out.columns:
            out["item_category_name"] = "Sin categoría"
        out["category_label"] = out.apply(
            lambda row: f"{int(row['item_category_id'])} — {row['item_category_name']}"
            if int(row["item_category_id"]) >= 0
            else "Sin categoría",
            axis=1,
        )

    if "recency" in out.columns:
        recency = pd.to_numeric(out["recency"], errors="coerce")
        out["recency_missing"] = recency.isna() | (recency >= RECENCY_SENTINEL)
        out["recency_clean"] = recency.mask(out["recency_missing"], np.nan)
        out["recency_label"] = out["recency_clean"].map(
            lambda value: "Sin ventas recientes / sin historial suficiente"
            if pd.isna(value)
            else f"{int(value)} meses"
        )

    return out


def preferred_forecast_columns(df: pd.DataFrame) -> list[str]:
    """Business-friendly column order for tables."""
    preferred = [
        "shop_id",
        "shop_name",
        "shop_label",
        "item_id",
        "item_name",
        "item_label",
        "prediction",
        "model_scope",
        "segment_key",
        "segment_name",
        "segment_source",
        "item_category_id",
        "item_category_name",
        "category_group",
        "category_label",
        "price_tier",
        "demand_tier",
        "months_active",
        "recency_label",
        "recency",
        "recency_clean",
        "metadata_status",
    ]
    return [col for col in preferred if col in df.columns] + [col for col in df.columns if col not in preferred]


def friendly_table(df: pd.DataFrame, max_rows: int = 500) -> None:
    """Render table with catalog labels and stable column order."""
    enriched = attach_catalog_labels(df)
    st.dataframe(enriched[preferred_forecast_columns(enriched)].head(max_rows), width="stretch")


def render_pretty_json(title: str, payload: dict[str, Any]) -> None:
    """Show technical JSON in an expander instead of a giant raw block."""
    with st.expander(title):
        st.code(json.dumps(payload, indent=2, ensure_ascii=False, default=str), language="json")


def flatten_metrics_dict(metrics: dict[str, Any]) -> pd.DataFrame:
    """Flatten nested metrics JSON into a readable table."""
    rows: list[dict[str, Any]] = []
    for section, value in metrics.items():
        if isinstance(value, dict):
            row = {"seccion": section}
            for key, item in value.items():
                if isinstance(item, (str, int, float, bool)) or item is None:
                    row[key] = item
            if len(row) > 1:
                rows.append(row)
    if not rows:
        return pd.json_normalize(metrics)
    return pd.DataFrame(rows)


def render_metrics_cards(metrics: dict[str, Any]) -> None:
    """Render ModelOps metrics as cards/tables instead of raw JSON."""
    if not metrics:
        st.info("No hay métricas remotas disponibles.")
        return

    global_metrics = metrics.get("global") or metrics.get("final") or metrics.get("champion") or {}
    naive_metrics = metrics.get("naive") or {}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE modelo", _fmt_metric(global_metrics.get("mae")))
    c2.metric("RMSE modelo", _fmt_metric(global_metrics.get("rmse")))
    c3.metric("MAE naive", _fmt_metric(naive_metrics.get("mae")))
    c4.metric("SMAPE", _fmt_metric(global_metrics.get("smape") or global_metrics.get("wape")))

    table = flatten_metrics_dict(metrics)
    display_cols = [
        col
        for col in ["seccion", "model_id", "model_name", "mae", "rmse", "smape", "wape", "bias", "nonzero_recall"]
        if col in table.columns
    ]
    if display_cols:
        st.dataframe(table[display_cols], width="stretch")
    else:
        st.dataframe(table, width="stretch")

    render_pretty_json("Ver JSON técnico de métricas", metrics)


def _fmt_metric(value: object) -> str:
    try:
        if value is None or pd.isna(value):
            return "N/D"
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def _natural_id_label_sort(label: object) -> tuple[int, str]:
    match = re.match(r"^\s*(\d+)", str(label))
    return (int(match.group(1)) if match else 10**9, str(label))


def make_label_options(df: pd.DataFrame, label_col: str) -> list[str]:
    labels = df[label_col].dropna().astype(str).unique().tolist()
    return sorted(labels, key=_natural_id_label_sort)


def first_int_from_label(label: object) -> int | None:
    match = re.match(r"^\s*(\d+)", str(label))
    if not match:
        return None
    return int(match.group(1))


def render_dual_shop_item_selector(
    df: pd.DataFrame,
    key_prefix: str,
    default_shop_id: int | None = None,
) -> tuple[int | None, int | None]:
    """Select shop/item either by dropdown or direct id input."""
    enriched = attach_catalog_labels(df)
    if "shop_label" not in enriched.columns or "item_label" not in enriched.columns:
        st.warning("No se pudieron construir etiquetas de tienda/producto.")
        return None, None

    shop_labels = make_label_options(enriched, "shop_label")
    if not shop_labels:
        return None, None

    default_shop_label = None
    if default_shop_id is not None:
        matches = [label for label in shop_labels if first_int_from_label(label) == default_shop_id]
        if matches:
            default_shop_label = matches[0]
    shop_index = shop_labels.index(default_shop_label) if default_shop_label in shop_labels else 0

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_shop_label = st.selectbox(
            "Selecciona tienda",
            options=shop_labels,
            index=shop_index,
            key=f"{key_prefix}_shop_select",
        )
    with col2:
        typed_shop = st.text_input(
            "O escribe shop_id",
            value="",
            placeholder="Ej. 31",
            key=f"{key_prefix}_shop_text",
        )

    selected_shop = _resolve_typed_or_selected_id(typed_shop, selected_shop_label, "shop_id")
    if selected_shop is None:
        return None, None

    item_pool = enriched.query("shop_id == @selected_shop")
    if item_pool.empty:
        st.warning("No hay productos disponibles para esa tienda.")
        return selected_shop, None

    item_labels = make_label_options(item_pool, "item_label")
    col3, col4 = st.columns([2, 1])
    with col3:
        selected_item_label = st.selectbox(
            "Selecciona producto",
            options=item_labels,
            key=f"{key_prefix}_item_select",
        )
    with col4:
        typed_item = st.text_input(
            "O escribe item_id",
            value="",
            placeholder="Ej. 5037",
            key=f"{key_prefix}_item_text",
        )
    selected_item = _resolve_typed_or_selected_id(typed_item, selected_item_label, "item_id")
    return selected_shop, selected_item


def _resolve_typed_or_selected_id(typed_value: str, selected_label: str, field_name: str) -> int | None:
    if typed_value.strip():
        try:
            return int(typed_value)
        except ValueError:
            st.error(f"{field_name} debe ser numérico.")
            return first_int_from_label(selected_label)
    return first_int_from_label(selected_label)


def render_top_shop_forecast_chart(batch_df: pd.DataFrame) -> None:
    """Render original top-shop chart but with real shop names."""
    enriched = attach_catalog_labels(batch_df)
    top_shops = (
        enriched.groupby(["shop_id", "shop_label"], as_index=False)
        .agg(total_forecast=("prediction", "sum"))
        .sort_values("total_forecast", ascending=False)
        .head(15)
    )
    fig = px.bar(
        top_shops,
        x="shop_label",
        y="total_forecast",
        title="Tiendas con mayor pronóstico total",
        labels={"shop_label": "Tienda", "total_forecast": "Unidades pronosticadas"},
    )
    fig.update_layout(xaxis_tickangle=-70)
    st.plotly_chart(fig, width="stretch")


def render_forecast_distribution(batch_df: pd.DataFrame) -> None:
    """Render forecast distribution with original and log1p options."""
    if batch_df.empty or "prediction" not in batch_df.columns:
        st.info("No hay predicciones para graficar distribución.")
        return

    plot_df = batch_df.copy()
    plot_df["prediction"] = pd.to_numeric(plot_df["prediction"], errors="coerce").fillna(0)
    plot_df["prediction_log1p"] = np.log1p(plot_df["prediction"])
    sample = plot_df.sample(min(25000, len(plot_df)), random_state=42)

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("P50", f"{plot_df['prediction'].quantile(0.50):.3f}")
    p2.metric("P90", f"{plot_df['prediction'].quantile(0.90):.3f}")
    p3.metric("P95", f"{plot_df['prediction'].quantile(0.95):.3f}")
    p4.metric("% < 0.1", f"{(plot_df['prediction'] < 0.1).mean() * 100:.1f}%")

    scale = st.radio(
        "Escala de la distribución",
        ["Original", "Log1p"],
        horizontal=True,
        key="forecast_distribution_scale",
        help="Log1p usa log(1 + pronóstico), útil cuando muchas predicciones están pegadas a cero.",
    )
    if scale == "Log1p":
        fig = px.histogram(
            sample,
            x="prediction_log1p",
            nbins=50,
            title="Distribución de pronósticos — escala log(1 + pronóstico)",
            labels={"prediction_log1p": "log(1 + unidades pronosticadas)"},
        )
    else:
        fig = px.histogram(
            sample,
            x="prediction",
            nbins=50,
            title="Distribución de pronósticos — escala original",
            labels={"prediction": "Unidades pronosticadas"},
        )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "La escala log1p no cambia los datos; solo ayuda a visualizar mejor una distribución con muchos valores cercanos a cero."
    )


def render_prediction_vs_actual_by_demand(eval_df: pd.DataFrame) -> None:
    """Keep and enrich the original mean predicted vs ground truth chart."""
    if eval_df.empty or not {"y", "prediction"}.issubset(eval_df.columns):
        st.info("No hay datos suficientes para graficar predicción vs valor real.")
        return

    plot_df = eval_df.copy()
    bins = [-0.1, 0, 1, 3, 7, 20]
    labels = ["(-0.1, 0.0]", "(0.0, 1.0]", "(1.0, 3.0]", "(3.0, 7.0]", "(7.0, 20.0]"]
    plot_df["demand_bucket"] = pd.cut(plot_df["y"], bins=bins, labels=labels, include_lowest=True)
    summary = (
        plot_df.groupby("demand_bucket", observed=True)
        .agg(
            real_mean=("y", "mean"),
            pred_mean=("prediction", "mean"),
            n=("y", "size"),
        )
        .reset_index()
    )
    summary["gap"] = summary["pred_mean"] - summary["real_mean"]
    summary["coverage_ratio"] = np.where(
        summary["real_mean"] > 0,
        summary["pred_mean"] / summary["real_mean"],
        np.nan,
    )
    melted = summary.melt(
        id_vars=["demand_bucket"],
        value_vars=["real_mean", "pred_mean"],
        var_name="Serie",
        value_name="Unidades promedio",
    )
    fig = px.line(
        melted,
        x="demand_bucket",
        y="Unidades promedio",
        color="Serie",
        markers=True,
        title="Promedio real vs promedio predicho por rango de demanda",
    )
    fig.update_layout(xaxis_title="Rango de demanda real")
    st.plotly_chart(fig, width="stretch")

    high = summary.tail(1).iloc[0]
    if pd.notna(high.get("coverage_ratio")) and high["coverage_ratio"] < 0.8:
        st.warning(
            "Diagnóstico: el modelo subestima la demanda alta. Esto no invalida el MVP, "
            "pero sí señala productos que deberían priorizarse en feedback y análisis de ML."
        )
    else:
        st.info(
            "Diagnóstico: la gráfica permite revisar si el modelo conserva la tendencia por rangos de demanda."
        )
    with st.expander("Ver tabla del diagnóstico por rango"):
        st.dataframe(summary, width="stretch")


def render_model_scope_help() -> None:
    """Explain model_scope without changing data semantics."""
    with st.expander("¿Qué significa model_scope?"):
        st.markdown(
            """
`model_scope` indica de dónde viene la predicción usada en esa fila:

- `global`: el modelo global predijo el par tienda-producto.
- `segment:*`: se usó un modelo o ajuste especializado para un segmento/categoría.
- `fallback`: se usó una regla de respaldo porque el producto tiene poco historial o metadata incompleta.

Esta columna no es un filtro de negocio; es una columna técnica para explicar trazabilidad del modelo.
            """.strip()
        )


def render_model_ranking_by_rmse(model_runs: pd.DataFrame, champion_id: str | None = None) -> None:
    """Plot all trained models ordered by champion first and then by RMSE."""
    if model_runs.empty:
        st.info("No hay corridas/modelos registrados todavía.")
        return
    df = model_runs.copy()
    if "rmse" not in df.columns:
        st.warning("El registry no tiene columna rmse; no se puede ordenar por RMSE.")
        st.dataframe(df, width="stretch")
        return
    df["rmse"] = pd.to_numeric(df["rmse"], errors="coerce")
    if champion_id and "model_id" in df.columns:
        df["is_current_champion"] = df["model_id"].astype(str).eq(str(champion_id))
    elif "is_champion" in df.columns:
        df["is_current_champion"] = df["is_champion"].astype(bool)
    else:
        df["is_current_champion"] = False
    df = df.sort_values(["is_current_champion", "rmse", "mae"], ascending=[False, True, True])
    y_col = "model_name" if "model_name" in df.columns else "model_id"
    fig = px.bar(
        df,
        x="rmse",
        y=y_col,
        orientation="h",
        title="Modelos entrenados ordenados por champion y RMSE",
        labels={"rmse": "RMSE menor es mejor", y_col: "Modelo"},
    )
    fig.update_layout(yaxis={"categoryorder": "array", "categoryarray": df[y_col].tolist()[::-1]})
    st.plotly_chart(fig, width="stretch")
    visible_cols = [
        col
        for col in [
            "model_id",
            "model_name",
            "model_family",
            "status",
            "is_current_champion",
            "mae",
            "rmse",
            "smape",
            "bias",
            "nonzero_recall",
            "beats_naive_mae",
        ]
        if col in df.columns
    ]
    st.dataframe(df[visible_cols], width="stretch")


def build_review_suggestions(eval_df: pd.DataFrame, limit: int = 100) -> pd.DataFrame:
    """Create problem-product suggestions from the evaluation sample.

    This mirrors the original idea of showing records the ML team should inspect,
    even when the RDS problem_products table is empty.
    """
    if eval_df.empty or "abs_error" not in eval_df.columns:
        return pd.DataFrame()
    df = attach_catalog_labels(eval_df).copy()
    df["abs_error"] = pd.to_numeric(df["abs_error"], errors="coerce").fillna(0)
    if "y" in df.columns and "prediction" in df.columns:
        df["reason"] = np.where(
            df["prediction"] > df["y"],
            "Posible sobrepronóstico",
            "Posible subpronóstico",
        )
    else:
        df["reason"] = "Error alto"
    cols = [
        col
        for col in [
            "shop_id",
            "shop_name",
            "item_id",
            "item_name",
            "item_category_name",
            "y",
            "prediction",
            "abs_error",
            "reason",
        ]
        if col in df.columns
    ]
    return df.sort_values("abs_error", ascending=False)[cols].head(limit)


def render_problem_product_suggestions(eval_df: pd.DataFrame, rds_df: pd.DataFrame | None = None) -> None:
    """Render RDS problem products, with evaluation-based fallback suggestions."""
    if rds_df is not None and not rds_df.empty:
        st.dataframe(attach_catalog_labels(rds_df), width="stretch")
        return
    suggestions = build_review_suggestions(eval_df, limit=100)
    if suggestions.empty:
        st.info("Sin productos sugeridos para revisión.")
    else:
        st.caption(
            "Sugerencias generadas con los mayores errores de evaluación. "
            "Sirven para que negocio o ML decidan qué productos investigar."
        )
        st.dataframe(suggestions, width="stretch")


def safe_metric_sort_key(value: object) -> float:
    try:
        if value is None or pd.isna(value):
            return math.inf
        return float(value)
    except Exception:
        return math.inf
