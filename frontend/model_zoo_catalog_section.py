"""Streamlit section for the friendly model zoo catalog."""

from __future__ import annotations

import streamlit as st

from backend.modelops_s3 import load_champion, load_model_runs


def render_model_zoo_catalog() -> None:
    """Render champion/challenger model catalog in the dashboard."""
    st.subheader("Catálogo de modelos entrenados")
    try:
        champion = load_champion()
        model_runs = load_model_runs()
    except Exception as exc:  # noqa: BLE001
        st.warning(f"No se pudo cargar el catálogo de modelos desde S3: {exc}")
        return

    champion_id = champion.get("model_id") or champion.get("champion_model_id")
    st.success(f"Champion actual: {champion_id}")

    cols = [
        "run_id",
        "model_id",
        "model_family",
        "status",
        "rmse",
        "mae",
        "wape",
        "bias",
        "nonzero_recall",
        "beats_naive_wape",
        "promotion_reason",
        "description",
    ]
    visible = [c for c in cols if c in model_runs.columns]
    if not visible:
        st.dataframe(model_runs, width="stretch")
        return

    sorted_runs = model_runs.copy()
    if "wape" in sorted_runs.columns:
        sorted_runs = sorted_runs.sort_values(["status", "wape"], ascending=[True, True])
    st.dataframe(sorted_runs[visible], width="stretch")

    if {"model_id", "wape"}.issubset(model_runs.columns):
        st.caption("WAPE menor es mejor. El champion se elige con WAPE/RMSE y comparación contra naive.")
        st.bar_chart(model_runs.set_index("model_id")["wape"])
