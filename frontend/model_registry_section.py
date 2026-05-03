"""Streamlit section for model registry and model_id selection.

Copy or import this module if the app needs an explicit model catalog view. It
reads the S3 registry produced by scripts/register_and_promote_model.py.
"""

from __future__ import annotations

import json
import os
from io import BytesIO

import boto3
import pandas as pd
import streamlit as st


def _bucket() -> str:
    bucket = os.getenv("MODEL_BUCKET")
    if not bucket:
        raise RuntimeError("MODEL_BUCKET is not set")
    return bucket


def _read_s3_bytes(key: str) -> bytes:
    obj = boto3.client("s3").get_object(Bucket=_bucket(), Key=key)
    return obj["Body"].read()


@st.cache_data(ttl=300)
def load_model_runs() -> pd.DataFrame:
    """Load model registry model_runs.csv from S3."""
    return pd.read_csv(BytesIO(_read_s3_bytes("modelops/registry/model_runs.csv")))


@st.cache_data(ttl=300)
def load_champion() -> dict:
    """Load champion metadata from S3."""
    return json.loads(_read_s3_bytes("modelops/registry/champion.json").decode("utf-8"))


@st.cache_data(ttl=300)
def load_segment_metrics_for_run(model_run_id: str) -> pd.DataFrame:
    """Load segment-level metrics for a selected run."""
    return pd.read_parquet(
        BytesIO(_read_s3_bytes(f"modelops/registry/segment_metrics/{model_run_id}.parquet"))
    )


def render_model_registry_selector() -> str | None:
    """Render a technical model_id selector and return the selected model_run_id.

    The selector is meant for transparency and debugging. The business-facing app
    should keep using modelops/latest/ as the champion output.
    """
    st.subheader("Catálogo de modelos")
    try:
        champion = load_champion()
        runs = load_model_runs()
    except Exception as exc:
        st.warning(f"No se pudo cargar el registro de modelos: {exc}")
        return None

    st.success(f"Champion actual: {champion.get('model_run_id')}")
    st.dataframe(runs.sort_values("created_at", ascending=False), width="stretch")

    run_ids = runs["model_run_id"].dropna().astype(str).tolist()
    if not run_ids:
        return None

    selected_run_id = st.selectbox("Seleccionar model_id para inspección técnica", run_ids)
    if selected_run_id:
        try:
            segment_metrics = load_segment_metrics_for_run(selected_run_id)
            st.write("Performance por segmento del model_id seleccionado")
            st.dataframe(segment_metrics, width="stretch")
        except Exception as exc:
            st.info(f"No hay métricas segmentadas para {selected_run_id}: {exc}")

    return selected_run_id
