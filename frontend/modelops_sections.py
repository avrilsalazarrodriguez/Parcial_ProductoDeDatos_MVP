"""Reusable Streamlit sections for ModelOps S3 outputs."""

from __future__ import annotations

import logging

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
    modelops_healthcheck,
)

LOGGER = logging.getLogger(__name__)


@st.cache_data(ttl=600)
def _cached_forecast_detail() -> pd.DataFrame:
    return load_forecast_detail()


@st.cache_data(ttl=600)
def _cached_eval_segment() -> pd.DataFrame:
    return load_evaluation_by_segment()


@st.cache_data(ttl=600)
def _cached_eval_item() -> pd.DataFrame:
    return load_evaluation_by_item()


@st.cache_data(ttl=600)
def _cached_model_metrics() -> dict:
    return load_model_metrics()


def get_batch_df_with_modelops_fallback(local_batch_df: pd.DataFrame) -> pd.DataFrame:
    """Return ModelOps forecast table if available; otherwise local fallback."""
    try:
        forecast = _cached_forecast_detail()
        if forecast.empty:
            st.warning("ModelOps S3 no devolvió registros. Uso predicciones locales empaquetadas.")
            return local_batch_df
        return forecast
    except Exception as exc:  # noqa: BLE001 - surface fallback in UI
        LOGGER.warning("action=modelops_forecast_load status=fallback error=%s", exc)
        st.warning("No puede cargar ModelOps S3. Uso predicciones locales empaquetadas.")
        return local_batch_df


def render_modelops_summary_block() -> None:
    """Render executive ModelOps summary."""
    st.subheader("Resumen ModelOps desde S3")
    try:
        metrics = _cached_model_metrics()
        champion = load_champion()
        col1, col2, col3 = st.columns(3)
        col1.metric("Champion run", champion.get("model_run_id", "sin registry"))
        col2.metric("Modelo final", str(metrics.get("final", "disponible"))[:30])
        col3.metric("Modelos segmentados", str(metrics.get("n_segment_models", "N/D")))
        with st.expander("Ver métricas globales"):
            st.json(metrics)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"No se pudo cargar resumen ModelOps: {exc}")


def render_modelops_evaluation_block() -> None:
    """Render ModelOps evaluation tables and charts."""
    st.subheader("Evaluación ModelOps desde S3")
    try:
        metrics = _cached_model_metrics()
        st.markdown("**Métricas globales**")
        st.json(metrics)

        segment_df = _cached_eval_segment()
        st.markdown("**Performance por segmento/categoría**")
        st.dataframe(segment_df, width="stretch")

        if not segment_df.empty and "mae" in segment_df.columns:
            plot_df = segment_df.sort_values("mae", ascending=False).head(20).copy()
            y_col = "segment_key" if "segment_key" in plot_df.columns else plot_df.columns[0]
            fig = px.bar(plot_df, x="mae", y=y_col, orientation="h", title="Segmentos con mayor MAE")
            st.plotly_chart(fig, width="stretch")

        item_df = _cached_eval_item()
        st.markdown("**Performance por producto**")
        st.dataframe(item_df.head(500), width="stretch")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"No se pudo cargar evaluación ModelOps desde S3: {exc}")


def render_modelops_registry_block() -> None:
    """Render model registry/champion information."""
    st.subheader("Model registry")
    try:
        champion = load_champion()
        runs = load_model_runs()
        if champion:
            st.markdown("**Champion actual**")
            st.json(champion)
        else:
            st.info("No existe champion.json todavía.")

        if not runs.empty:
            st.markdown("**Historial de corridas**")
            st.dataframe(runs, width="stretch")
        else:
            st.info("No existe model_runs.csv todavía.")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"No se pudo cargar registry ModelOps: {exc}")


def render_modelops_debug_block() -> None:
    """Render S3 object healthcheck for debugging."""
    try:
        st.json(modelops_healthcheck())
    except Exception as exc:  # noqa: BLE001
        st.error(f"No se pudo ejecutar healthcheck ModelOps: {exc}")
