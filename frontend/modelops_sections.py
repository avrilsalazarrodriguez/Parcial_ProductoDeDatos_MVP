"""Reusable Streamlit sections for ModelOps outputs.

Compatibility hotfix v4.2.

This module keeps the function names expected by the existing `frontend/app.py`:

- get_batch_df_with_modelops_fallback
- render_modelops_summary_block
- render_modelops_evaluation_block
- render_kpi_blocks
- render_model_registry_block
- render_model_scope_help
- render_review_suggestions_block
- render_modelops_debug_block

The UI keeps the original app structure and only improves presentation:
real shop/item names, readable metric cards, RMSE-based model ranking, and
technical JSON inside expanders instead of giant raw JSON blocks.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from backend.modelops_s3 import (
    load_champion,
    load_evaluation_by_item,
    load_evaluation_by_segment,
    load_forecast_detail,
    load_forecast_summary_by_category,
    load_forecast_summary_by_shop_segment,
    load_model_metrics,
    load_model_runs,
    load_review_suggestions,
    modelops_healthcheck,
)
from frontend.ui_enhancements import (
    attach_catalog_labels,
    friendly_table,
    render_metrics_cards,
    render_model_ranking_by_rmse,
    render_model_scope_help as _render_model_scope_help,
    render_pretty_json,
)

LOGGER = logging.getLogger(__name__)


@st.cache_data(ttl=600, show_spinner=False)
def _cached_forecast_detail() -> pd.DataFrame:
    """Load detailed forecasts from ModelOps storage."""
    return load_forecast_detail()


@st.cache_data(ttl=600, show_spinner=False)
def _cached_eval_segment() -> pd.DataFrame:
    """Load segment-level evaluation from ModelOps storage."""
    return load_evaluation_by_segment()


@st.cache_data(ttl=600, show_spinner=False)
def _cached_eval_item() -> pd.DataFrame:
    """Load item-level evaluation from ModelOps storage."""
    return load_evaluation_by_item()


@st.cache_data(ttl=600, show_spinner=False)
def _cached_model_metrics() -> dict[str, Any]:
    """Load global metrics from ModelOps storage."""
    return load_model_metrics()


@st.cache_data(ttl=600, show_spinner=False)
def _cached_model_runs() -> pd.DataFrame:
    """Load model registry runs from ModelOps storage."""
    return load_model_runs()


@st.cache_data(ttl=600, show_spinner=False)
def _cached_review_suggestions() -> pd.DataFrame:
    """Load review suggestions from ModelOps storage."""
    return load_review_suggestions()


def get_batch_df_with_modelops_fallback(local_batch_df: pd.DataFrame) -> pd.DataFrame:
    """Return ModelOps forecast table if available; otherwise local fallback."""
    try:
        forecast = _cached_forecast_detail()
        if forecast.empty:
            st.warning("ModelOps no devolvió registros. Uso predicciones locales empaquetadas.")
            return attach_catalog_labels(local_batch_df)
        return attach_catalog_labels(forecast)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("action=modelops_forecast_load status=fallback error=%s", exc)
        st.warning("No pude cargar ModelOps. Uso predicciones locales empaquetadas.")
        return attach_catalog_labels(local_batch_df)


def render_modelops_summary_block() -> None:
    """Render executive ModelOps summary using cards instead of raw JSON."""
    st.subheader("Resumen ModelOps")
    try:
        metrics = _cached_model_metrics()
        champion = load_champion()
        global_metrics = metrics.get("global") or metrics.get("final") or metrics.get("champion") or {}

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Champion run", champion.get("model_run_id", "sin registry"))
        col2.metric("Modelo final", champion.get("model_name") or champion.get("model_id", "N/D"))
        col3.metric("RMSE", _metric(global_metrics.get("rmse")))
        col4.metric("MAE", _metric(global_metrics.get("mae")))

        selection_policy = metrics.get("selection_policy") or champion.get("selection_policy")
        if selection_policy:
            st.caption(f"Política de selección: {selection_policy}")
        else:
            st.caption("Política de selección: RMSE como criterio principal; MAE como desempate.")

        render_metrics_cards(metrics)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"No se pudo cargar resumen ModelOps: {exc}")


def render_modelops_evaluation_block() -> None:
    """Render ModelOps evaluation tables and charts without raw JSON clutter."""
    st.subheader("Evaluación ModelOps")
    try:
        metrics = _cached_model_metrics()
        st.markdown("**Métricas globales**")
        render_metrics_cards(metrics)

        segment_df = attach_catalog_labels(_cached_eval_segment())
        st.markdown("**Performance por segmento/categoría**")
        if segment_df.empty:
            st.info("No hay evaluación por segmento disponible.")
        else:
            st.dataframe(segment_df, width="stretch")
            metric_col = _first_existing_col(segment_df, ["rmse", "mae", "abs_error"])
            if metric_col:
                y_col = _first_existing_col(segment_df, ["segment_name", "segment_key", "item_category_name"])
                plot_df = segment_df.sort_values(metric_col, ascending=False).head(20).copy()
                fig = px.bar(
                    plot_df,
                    x=metric_col,
                    y=y_col,
                    orientation="h",
                    title=f"Segmentos con mayor {metric_col.upper()}",
                    labels={metric_col: metric_col.upper(), y_col: "Segmento"},
                )
                st.plotly_chart(fig, width="stretch")

        item_df = attach_catalog_labels(_cached_eval_item())
        st.markdown("**Performance por producto**")
        if item_df.empty:
            st.info("No hay evaluación por producto disponible.")
        else:
            friendly_table(item_df, max_rows=500)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"No se pudo cargar evaluación ModelOps: {exc}")


def render_kpi_blocks() -> None:
    """Render KPI blocks expected by the current app.

    This restores the function imported by `frontend/app.py`. It uses ModelOps
    evaluation outputs and keeps labels readable for business users.
    """
    st.subheader("KPIs por tienda, producto y segmento")
    try:
        segment_df = attach_catalog_labels(_cached_eval_segment())
        item_df = attach_catalog_labels(_cached_eval_item())

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Segmentos/categorías con mayor error**")
            if segment_df.empty:
                st.info("No hay KPIs por segmento disponibles.")
            else:
                metric_col = _first_existing_col(segment_df, ["rmse", "mae", "abs_error"])
                label_col = _first_existing_col(segment_df, ["segment_name", "segment_key", "item_category_name"])
                plot_df = segment_df.sort_values(metric_col, ascending=False).head(15)
                fig = px.bar(
                    plot_df,
                    x=metric_col,
                    y=label_col,
                    orientation="h",
                    title=f"Top segmentos por {metric_col.upper()}",
                )
                st.plotly_chart(fig, width="stretch")
                st.dataframe(segment_df.head(200), width="stretch")

        with col2:
            st.markdown("**Productos con mayor error**")
            if item_df.empty:
                st.info("No hay KPIs por producto disponibles.")
            else:
                metric_col = _first_existing_col(item_df, ["rmse", "mae", "abs_error"])
                label_col = _first_existing_col(item_df, ["item_label", "item_name", "item_id"])
                plot_df = item_df.sort_values(metric_col, ascending=False).head(15)
                fig = px.bar(
                    plot_df,
                    x=metric_col,
                    y=label_col,
                    orientation="h",
                    title=f"Top productos por {metric_col.upper()}",
                )
                st.plotly_chart(fig, width="stretch")
                friendly_table(item_df, max_rows=200)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"No se pudieron cargar KPIs ModelOps: {exc}")


def render_model_registry_block() -> None:
    """Render model registry/champion information ranked by RMSE."""
    st.subheader("Model registry")
    try:
        champion = load_champion()
        runs = _cached_model_runs()
        champion_id = champion.get("model_id") if champion else None

        if champion:
            st.markdown("**Champion actual**")
            col1, col2, col3 = st.columns(3)
            col1.metric("Modelo", champion.get("model_name") or champion.get("model_id", "N/D"))
            col2.metric("RMSE", _metric(champion.get("rmse")))
            col3.metric("MAE", _metric(champion.get("mae")))
            st.caption("El ranking usa RMSE como primer criterio y MAE como desempate.")
            render_pretty_json("Ver champion.json técnico", champion)
        else:
            st.info("No existe champion.json todavía.")

        if not runs.empty:
            st.markdown("**Historial de modelos/corridas**")
            render_model_ranking_by_rmse(runs, champion_id=champion_id)
        else:
            st.info("No existe model_runs.csv todavía.")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"No se pudo cargar registry ModelOps: {exc}")


def render_model_scope_help() -> None:
    """Compatibility wrapper for app import."""
    _render_model_scope_help()


def render_review_suggestions_block(limit: int = 100) -> pd.DataFrame:
    """Render review suggestions and return the dataframe for feedback selectors."""
    st.subheader("Registros sugeridos para revisión")
    try:
        suggestions = attach_catalog_labels(_cached_review_suggestions())
        if suggestions.empty:
            st.info("No hay sugerencias de revisión en ModelOps todavía.")
            return suggestions

        metric_col = _first_existing_col(suggestions, ["abs_error", "rmse", "mae"])
        if metric_col:
            suggestions = suggestions.sort_values(metric_col, ascending=False)
        friendly_table(suggestions, max_rows=limit)
        return suggestions.head(limit)
    except Exception as exc:  # noqa: BLE001
        st.info(f"No se pudieron cargar sugerencias de revisión: {exc}")
        return pd.DataFrame()


def render_modelops_debug_block() -> None:
    """Render storage healthcheck for debugging."""
    try:
        render_pretty_json("Healthcheck ModelOps", modelops_healthcheck())
    except Exception as exc:  # noqa: BLE001
        st.error(f"No se pudo ejecutar healthcheck ModelOps: {exc}")


def render_modelops_forecast_summaries() -> None:
    """Optional block for category/shop segment summaries."""
    try:
        by_category = attach_catalog_labels(load_forecast_summary_by_category())
        by_shop_segment = attach_catalog_labels(load_forecast_summary_by_shop_segment())
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Forecast por categoría**")
            friendly_table(by_category, max_rows=200)
        with col2:
            st.markdown("**Forecast por tienda/segmento**")
            friendly_table(by_shop_segment, max_rows=200)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("action=render_forecast_summaries status=skipped error=%s", exc)


def _metric(value: object) -> str:
    try:
        if value is None or pd.isna(value):
            return "N/D"
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None
