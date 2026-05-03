from __future__ import annotations

from typing import Callable

import pandas as pd
import plotly.express as px
import streamlit as st


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _low_activity_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    if "model_scope" in df.columns:
        mask = mask | df["model_scope"].astype(str).str.contains("inactive|no_recent", case=False, na=False)
    if "routing_reason" in df.columns:
        mask = mask | df["routing_reason"].astype(str).str.contains("recency|rolling_sum_6|no_recent", case=False, na=False)
    if "recency" in df.columns:
        mask = mask | (_num(df, "recency", 0.0) >= 99)
    if "recency_router" in df.columns:
        mask = mask | (_num(df, "recency_router", 0.0) >= 99)

    lag1_col = _first_existing(df, ["cnt_lag_1", "lag_1", "target_lag_1", "item_cnt_month_lag_1"])
    sum6_col = _first_existing(df, ["sum_6", "rolling_sum_6", "router_rolling_sum_6", "rolling_sum_6_router"])
    if lag1_col and sum6_col:
        mask = mask | ((_num(df, lag1_col, 0.0) <= 0) & (_num(df, sum6_col, 0.0) <= 0))
    return mask


def _curate_low_activity_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "shop_id",
        "shop_name",
        "item_id",
        "item_name",
        "item_category_id",
        "item_category_name",
        "category_group",
        "prediction",
        "decision_recommendation",
        "model_scope",
        "routing_reason",
        "recency",
        "recency_router",
        "cnt_lag_1",
        "lag_1",
        "sum_6",
        "rolling_sum_6",
        "router_rolling_sum_6",
        "rolling_sum_6_router",
        "mean_6",
        "rolling_mean_6",
        "router_rolling_mean_6",
        "nonzero_rate_6",
        "router_nonzero_rate_6",
    ]
    cols = [c for c in preferred if c in df.columns]
    cols.extend([c for c in df.columns if c not in cols and not c.endswith("_label")][:10])
    return df[cols]


def _short(value: object, max_len: int = 55) -> str:
    text = str(value)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def render_low_activity_products_table(
    batch_df: pd.DataFrame,
    *,
    enrich_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    render_dataframe_fn: Callable[..., None] | None = None,
) -> None:
    """Render KPI table and charts for products with low/recently inactive activity."""
    st.subheader("Productos con baja actividad reciente")
    st.caption(
        "Productos-tienda que parecen inactivos o con señal reciente muy baja. "
        "Útil para auditar reglas de predicción cero, productos posiblemente descontinuados o pares sin historial reciente."
    )

    if batch_df is None or batch_df.empty:
        st.info("No hay pronósticos batch para analizar baja actividad.")
        return

    df = batch_df.copy()
    if enrich_fn is not None:
        try:
            df = enrich_fn(df)
        except Exception:
            pass

    mask = _low_activity_mask(df)
    low = df[mask].copy()
    if low.empty:
        st.info("No encontré productos con baja actividad reciente usando model_scope, routing_reason, recency o lags.")
        return

    query = st.text_input("Buscar producto/tienda en baja actividad", value="", key="kpi_low_activity_search")
    if query.strip():
        qmask = pd.Series(False, index=low.index)
        for col in ["shop_id", "shop_name", "item_id", "item_name", "item_category_id", "item_category_name"]:
            if col in low.columns:
                qmask = qmask | low[col].astype(str).str.contains(query, case=False, na=False)
        low = low[qmask]

    c1, c2, c3 = st.columns(3)
    c1.metric("Registros baja actividad", f"{len(low):,}")
    c2.metric("% del catálogo", f"{(len(low) / max(len(df), 1)) * 100:.1f}%")
    c3.metric("Pronóstico total", f"{low['prediction'].sum():,.2f}" if "prediction" in low.columns else "N/D")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        if "shop_id" in low.columns:
            group_cols = ["shop_id"] + (["shop_name"] if "shop_name" in low.columns else [])
            by_shop = (
                low.groupby(group_cols, dropna=False, as_index=False)
                .agg(
                    productos_baja_actividad=("item_id", "nunique") if "item_id" in low.columns else ("prediction", "size"),
                    registros=("prediction", "size"),
                    pronostico_total=("prediction", "sum") if "prediction" in low.columns else ("shop_id", "size"),
                )
                .sort_values("productos_baja_actividad", ascending=False)
                .head(10)
            )
            label_col = "shop_name" if "shop_name" in by_shop.columns else "shop_id"
            by_shop["shop_display"] = by_shop[label_col].map(lambda x: _short(x, 42))
            fig = px.bar(
                by_shop.sort_values("productos_baja_actividad"),
                x="productos_baja_actividad",
                y="shop_display",
                orientation="h",
                title="Top 10 tiendas con más productos de baja actividad",
                labels={"productos_baja_actividad": "Productos baja actividad", "shop_display": "Tienda"},
            )
            st.plotly_chart(fig, width="stretch", key="kpi_low_activity_by_shop")
        else:
            st.info("No hay shop_id para graficar tiendas con baja actividad.")

    with chart_col2:
        if "item_category_id" in low.columns:
            group_cols = ["item_category_id"] + (["item_category_name"] if "item_category_name" in low.columns else [])
            by_cat = (
                low.groupby(group_cols, dropna=False, as_index=False)
                .agg(
                    productos_baja_actividad=("item_id", "nunique") if "item_id" in low.columns else ("prediction", "size"),
                    registros=("prediction", "size"),
                    pronostico_total=("prediction", "sum") if "prediction" in low.columns else ("item_category_id", "size"),
                )
                .sort_values("productos_baja_actividad", ascending=False)
                .head(10)
            )
            if "item_category_name" in by_cat.columns:
                by_cat["category_display"] = by_cat["item_category_id"].astype(str) + " — " + by_cat["item_category_name"].map(lambda x: _short(x, 34))
            else:
                by_cat["category_display"] = by_cat["item_category_id"].astype(str)
            fig = px.bar(
                by_cat.sort_values("productos_baja_actividad"),
                x="productos_baja_actividad",
                y="category_display",
                orientation="h",
                title="Top 10 categorías con más productos de baja actividad",
                labels={"productos_baja_actividad": "Productos baja actividad", "category_display": "Categoría"},
            )
            st.plotly_chart(fig, width="stretch", key="kpi_low_activity_by_category")
        else:
            st.info("No hay item_category_id para graficar categorías con baja actividad.")

    st.markdown("**Tabla de productos con baja actividad reciente**")
    show = _curate_low_activity_columns(low).head(1000)
    if render_dataframe_fn is not None:
        render_dataframe_fn(show, max_rows=1000, height=360)
    else:
        st.dataframe(show, width="stretch", height=360)
