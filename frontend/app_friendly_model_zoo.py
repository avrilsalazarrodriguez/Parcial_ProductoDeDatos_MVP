"""Streamlit app with friendly names, clean ModelOps outputs and model catalog.

This replacement app keeps the same MVP functions as the current app, but adds:

* shop labels such as `2 — Store name` in filters;
* item/category names in forecast and KPI tables;
* cleaned null-like metadata and recency labels;
* model-zoo catalog view with champion/challenger models;
* safe local RDS behavior through `DISABLE_RDS_WRITES=true`.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from backend.modelops_s3 import (  # noqa: E402
    load_evaluation_by_item,
    load_evaluation_by_segment,
    load_evaluation_detail,
    load_forecast_detail,
    load_model_metrics,
    modelops_healthcheck,
)
from backend.storage import upload_batch_dataframe_to_s3  # noqa: E402
from backend.db import (  # noqa: E402
    insert_batch_export as _insert_batch_export,
    insert_business_feedback as _insert_business_feedback,
    insert_usage_event as _insert_usage_event,
    read_batch_exports as _read_batch_exports,
    read_business_feedback as _read_business_feedback,
    read_problem_products as _read_problem_products,
)
from frontend.friendly_display import (  # noqa: E402
    ensure_friendly_columns,
    first_int_from_label,
    friendly_forecast_columns,
    make_options,
    render_forecast_filters,
    render_table,
)
from frontend.model_zoo_catalog_section import render_model_zoo_catalog  # noqa: E402

LOGGER = logging.getLogger(__name__)

MODEL_PATH = Path("artifacts/model.joblib")
VALID_PATH = Path("data/prep/valid.parquet")
TEST_FEATURES_PATH = Path("data/prep/test_features.parquet")
TEST_PAIRS_PATH = Path("data/prep/test_pairs.parquet")
SUBMISSION_PATH = Path("data/predictions/submission.csv")

USE_MODELOPS_S3 = os.getenv("USE_MODELOPS_S3", "true").lower() == "true"
DISABLE_RDS_WRITES = os.getenv("DISABLE_RDS_WRITES", "false").lower() == "true"


def _log_nonfatal(action: str, exc: Exception) -> None:
    LOGGER.warning(
        "action=%s status=skipped error_type=%s error_message=%s",
        action,
        type(exc).__name__,
        str(exc)[:250],
    )


def safe_insert_usage_event(**kwargs) -> bool:
    if DISABLE_RDS_WRITES:
        return False
    try:
        _insert_usage_event(**kwargs)
        return True
    except Exception as exc:  # noqa: BLE001
        _log_nonfatal("insert_usage_event", exc)
        return False


def safe_insert_batch_export(**kwargs) -> bool:
    if DISABLE_RDS_WRITES:
        return False
    try:
        _insert_batch_export(**kwargs)
        return True
    except Exception as exc:  # noqa: BLE001
        _log_nonfatal("insert_batch_export", exc)
        return False


def safe_read_batch_exports(limit: int = 20) -> pd.DataFrame:
    if DISABLE_RDS_WRITES:
        return pd.DataFrame()
    try:
        return _read_batch_exports(limit=limit)
    except Exception as exc:  # noqa: BLE001
        _log_nonfatal("read_batch_exports", exc)
        return pd.DataFrame()


def safe_insert_feedback(**kwargs) -> bool:
    if DISABLE_RDS_WRITES:
        raise RuntimeError("RDS está desactivado en local. Prueba feedback real en ECS/Fargate.")
    _insert_business_feedback(**kwargs)
    return True


def safe_read_feedback(limit: int = 100) -> pd.DataFrame:
    if DISABLE_RDS_WRITES:
        return pd.DataFrame()
    try:
        return _read_business_feedback(limit=limit)
    except Exception as exc:  # noqa: BLE001
        _log_nonfatal("read_business_feedback", exc)
        return pd.DataFrame()


def safe_read_problem_products(limit: int = 100) -> pd.DataFrame:
    if DISABLE_RDS_WRITES:
        return pd.DataFrame()
    try:
        return _read_problem_products(limit=limit)
    except Exception as exc:  # noqa: BLE001
        _log_nonfatal("read_problem_products", exc)
        return pd.DataFrame()


@st.cache_data(ttl=600)
def load_valid_data() -> pd.DataFrame:
    return pd.read_parquet(VALID_PATH)


@st.cache_data(ttl=600)
def load_test_features() -> pd.DataFrame:
    return pd.read_parquet(TEST_FEATURES_PATH)


@st.cache_data(ttl=600)
def load_test_pairs() -> pd.DataFrame:
    return pd.read_parquet(TEST_PAIRS_PATH)


@st.cache_data(ttl=600)
def load_submission() -> pd.DataFrame:
    return pd.read_csv(SUBMISSION_PATH)


@st.cache_resource
def load_model() -> dict:
    return joblib.load(MODEL_PATH)


@st.cache_data(ttl=600)
def load_batch_forecast() -> pd.DataFrame:
    """Load forecast from S3 when available; fall back to local submission."""
    if USE_MODELOPS_S3:
        try:
            return ensure_friendly_columns(load_forecast_detail())
        except Exception as exc:  # noqa: BLE001
            st.sidebar.warning("No pude cargar ModelOps S3. Uso predicciones locales.")
            LOGGER.warning("action=load_forecast_detail status=fallback error=%s", str(exc)[:250])
    pairs = load_test_pairs().copy()
    submission = load_submission()
    pairs["prediction"] = submission.iloc[:, -1].to_numpy()
    return ensure_friendly_columns(pairs)


@st.cache_data(ttl=600)
def load_remote_evaluation_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    if not USE_MODELOPS_S3:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
    try:
        return (
            load_evaluation_detail(),
            load_evaluation_by_segment(),
            load_evaluation_by_item(),
            load_model_metrics(),
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("action=load_remote_eval status=fallback error=%s", str(exc)[:250])
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}


def predict_with_model(model_payload: dict, features: pd.DataFrame) -> np.ndarray:
    bundle = model_payload["bundle"]
    feature_cols = bundle["feature_cols"]
    x_test = features[feature_cols]
    prob = bundle["clf"].predict_proba(x_test)[:, 1].astype(np.float32)
    mu = bundle["reg"].predict(x_test).astype(np.float32)
    return np.clip(prob * mu, 0, 20)


def build_evaluation_sample(valid_df: pd.DataFrame, model_payload: dict, forecast_df: pd.DataFrame) -> pd.DataFrame:
    sample = valid_df.sample(min(5000, len(valid_df)), random_state=42).copy()
    sample["prediction"] = predict_with_model(model_payload, sample)
    sample["naive_prediction"] = sample["cnt_lag_1"].clip(0, 20) if "cnt_lag_1" in sample.columns else 0.0
    sample["error"] = sample["prediction"] - sample["y"]
    sample["abs_error"] = sample["error"].abs()
    sample["low_activity_flag"] = sample["dead_6"].astype(int) if "dead_6" in sample.columns else 0

    if {"shop_id", "item_id"}.issubset(sample.columns):
        dim_cols = [
            col
            for col in [
                "shop_id",
                "shop_name",
                "shop_label",
                "item_id",
                "item_name",
                "item_label",
                "item_category_id",
                "item_category_name",
                "category_group",
                "category_label",
            ]
            if col in forecast_df.columns
        ]
        if dim_cols:
            dims = forecast_df[dim_cols].drop_duplicates(["shop_id", "item_id"])
            sample = sample.merge(dims, on=["shop_id", "item_id"], how="left")
    return ensure_friendly_columns(sample)


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def build_uploaded_batch(uploaded_df: pd.DataFrame, model_payload: dict) -> pd.DataFrame:
    bundle = model_payload["bundle"]
    feature_cols = bundle["feature_cols"]
    missing = [col for col in feature_cols if col not in uploaded_df.columns]
    if missing:
        raise ValueError("El archivo no tiene columnas requeridas: " + ", ".join(missing[:10]))
    result = uploaded_df.copy()
    result["prediction"] = predict_with_model(model_payload, result)
    return result


st.set_page_config(page_title="1C Company - Pronóstico de Ventas", page_icon="📦", layout="wide")
st.title("📦 Producto de Datos — Pronóstico de Ventas")
st.caption("MVP Streamlit para consultar pronósticos, evaluar modelos y capturar feedback.")

with st.sidebar:
    st.subheader("Estado de conexiones")
    st.write("ModelOps S3:", "activo" if USE_MODELOPS_S3 else "inactivo")
    st.write("RDS:", "desactivado localmente" if DISABLE_RDS_WRITES else "activo")

valid_df = load_valid_data()
test_features = load_test_features()
test_pairs = load_test_pairs()
model_payload = load_model()
batch_df = load_batch_forecast()
eval_df = build_evaluation_sample(valid_df, model_payload, batch_df)
remote_eval_detail, remote_by_segment, remote_by_item, remote_metrics = load_remote_evaluation_tables()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Resumen", "Inferencia individual", "Batch CFO", "Evaluación", "KPIs", "Feedback"]
)

with tab1:
    st.header("Resumen ejecutivo")
    st.write(
        "La app muestra pronósticos mensuales, métricas de evaluación y feedback operativo. "
        "Las tablas usan nombres de tienda/categoría para facilitar rastreo de negocio."
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Filas de validación", f"{len(valid_df):,}")
    col2.metric("Pares tienda-producto", f"{len(test_pairs):,}")
    col3.metric("Predicciones batch", f"{len(batch_df):,}")
    champion = remote_metrics.get("champion_model_id") or remote_metrics.get("global", {}).get("model_id", "modelo local")
    col4.metric("Modelo en uso", str(champion))

    st.subheader("Muestra de pronósticos disponibles")
    render_table(batch_df.head(100), max_rows=100)

    st.subheader("Dashboard rápido")
    top_shops = (
        batch_df.groupby(["shop_id", "shop_name"], as_index=False)
        .agg(total_forecast=("prediction", "sum"))
        .sort_values("total_forecast", ascending=False)
        .head(15)
    )
    top_shops["shop_label"] = top_shops["shop_id"].astype(str) + " — " + top_shops["shop_name"]
    c1, c2 = st.columns(2)
    c1.plotly_chart(
        px.bar(top_shops, x="shop_label", y="total_forecast", title="Tiendas con mayor pronóstico"),
        width="stretch",
    )
    c2.plotly_chart(
        px.histogram(batch_df.sample(min(10000, len(batch_df)), random_state=42), x="prediction", nbins=30),
        width="stretch",
    )

with tab2:
    st.header("Inferencia individual")
    c1, c2 = st.columns(2)
    with c1:
        selected_shop_label = st.selectbox("Tienda", make_options(batch_df, "shop_label", include_all=False))
        selected_shop = first_int_from_label(selected_shop_label)
    item_options = batch_df.query("shop_id == @selected_shop")["item_label"].dropna().sort_values().unique().tolist()
    with c2:
        selected_item_label = st.selectbox("Producto", item_options)
        selected_item = first_int_from_label(selected_item_label)

    selected_rows = test_pairs.query("shop_id == @selected_shop and item_id == @selected_item")
    if selected_rows.empty:
        st.warning("No se encontró ese par tienda-producto en el conjunto futuro.")
    else:
        selected_features = test_features.iloc[[selected_rows.index[0]]]
        pred = predict_with_model(model_payload, selected_features)[0]
        safe_insert_usage_event(
            event_type="single_inference",
            shop_id=int(selected_shop),
            item_id=int(selected_item),
            records_count=1,
            status="success",
            message="Inferencia individual ejecutada.",
        )
        st.metric("Pronóstico próximo mes", f"{pred:.2f} unidades")
        with st.expander("Ver features usadas"):
            st.dataframe(selected_features, width="stretch")

with tab3:
    st.header("Batch CFO")
    scope = st.radio("Alcance", ["Todos los productos de una tienda", "Catálogo completo"], horizontal=True)
    if scope == "Todos los productos de una tienda":
        shop_label = st.selectbox("Selecciona tienda", make_options(batch_df, "shop_label", include_all=False))
        selected_shop = first_int_from_label(shop_label)
        filtered_batch = batch_df.query("shop_id == @selected_shop").copy()
        st.caption(f"Mostrando pronósticos de {shop_label}.")
    else:
        selected_shop = None
        filtered_batch = render_forecast_filters(batch_df, key_prefix="cfo")

    c1, c2, c3 = st.columns(3)
    c1.metric("Registros", f"{len(filtered_batch):,}")
    c2.metric("Pronóstico total", f"{filtered_batch['prediction'].sum():,.1f}")
    c3.metric("Promedio", f"{filtered_batch['prediction'].mean():.3f}")
    render_table(filtered_batch, max_rows=1000)

    csv = friendly_forecast_columns(filtered_batch).to_csv(index=False).encode("utf-8")
    if st.button("Generar archivo CFO y guardar en S3"):
        s3_uri = upload_batch_dataframe_to_s3(filtered_batch, scope=scope, shop_id=selected_shop)
        registered = safe_insert_batch_export(
            scope=scope,
            shop_id=selected_shop,
            records_count=len(filtered_batch),
            total_prediction=float(filtered_batch["prediction"].sum()),
            s3_uri=s3_uri,
        )
        st.success("Archivo CFO generado y guardado en S3" + (" y registrado en RDS." if registered else "."))
        st.code(s3_uri)
    st.download_button("Descargar archivo CFO", data=csv, file_name="forecast_cfo_next_month.csv", mime="text/csv")

    st.subheader("Historial de archivos generados")
    history = safe_read_batch_exports(limit=20)
    st.dataframe(history, width="stretch") if not history.empty else st.info("Sin historial disponible en esta sesión.")

    st.subheader("Batch por archivo cargado")
    uploaded_file = st.file_uploader("Subir CSV con features", type=["csv"])
    if uploaded_file is not None:
        uploaded_df = pd.read_csv(uploaded_file)
        st.dataframe(uploaded_df.head(), width="stretch")
        if st.button("Predecir archivo cargado"):
            try:
                uploaded_predictions = build_uploaded_batch(uploaded_df, model_payload)
                safe_insert_usage_event(
                    event_type="uploaded_batch_inference",
                    records_count=len(uploaded_predictions),
                    status="success",
                    message="Archivo cargado predicho correctamente.",
                )
                st.dataframe(uploaded_predictions.head(100), width="stretch")
            except ValueError as exc:
                st.error(str(exc))

with tab4:
    st.header("Evaluación vs ground truth")
    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE modelo local", f"{rmse(eval_df['y'], eval_df['prediction']):.4f}")
    col2.metric("RMSE naive", f"{rmse(eval_df['y'], eval_df['naive_prediction']):.4f}")
    col3.metric("MAE modelo local", f"{mae(eval_df['y'], eval_df['prediction']):.4f}")

    if remote_metrics:
        st.subheader("Métricas ModelOps / Model Zoo")
        st.json(remote_metrics.get("global", remote_metrics))
    if not remote_by_segment.empty:
        st.subheader("Performance por segmento")
        st.dataframe(remote_by_segment.head(500), width="stretch")
    if not remote_by_item.empty:
        st.subheader("Performance por producto")
        st.dataframe(remote_by_item.head(500), width="stretch")

    st.subheader("Muestra de errores local enriquecida")
    cols = [c for c in ["shop_id", "shop_name", "item_id", "item_name", "item_category_name", "y", "prediction", "naive_prediction", "abs_error"] if c in eval_df.columns]
    st.dataframe(eval_df[cols].head(500), width="stretch")

with tab5:
    st.header("KPIs por tienda, producto y modelo")
    shop_cols = ["shop_id"] + (["shop_name"] if "shop_name" in eval_df.columns else [])
    by_shop = eval_df.groupby(shop_cols, as_index=False).agg(n=("y", "size"), mae=("abs_error", "mean"), pred_mean=("prediction", "mean"))
    by_shop["shop_label"] = by_shop["shop_id"].astype(str) + " — " + by_shop.get("shop_name", "")
    item_cols = ["item_id"] + [c for c in ["item_name", "item_category_name", "category_group"] if c in eval_df.columns]
    by_item = eval_df.groupby(item_cols, as_index=False).agg(n=("y", "size"), mae=("abs_error", "mean"), pred_mean=("prediction", "mean"))

    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(by_shop.sort_values("mae", ascending=False).head(15), x="mae", y="shop_label", orientation="h"), width="stretch")
    item_plot = by_item.sort_values("mae", ascending=False).head(15).copy()
    item_plot["item_label"] = item_plot["item_id"].astype(str) + " — " + item_plot.get("item_name", "")
    c2.plotly_chart(px.bar(item_plot, x="mae", y="item_label", orientation="h"), width="stretch")

    st.subheader("Error por tienda")
    st.dataframe(by_shop.sort_values("mae", ascending=False), width="stretch")
    st.subheader("Productos con mayor error")
    st.dataframe(by_item.sort_values("mae", ascending=False).head(100), width="stretch")

    with st.expander("Catálogo de modelos entrenados"):
        render_model_zoo_catalog()
    with st.expander("Debug ModelOps S3"):
        st.json(modelops_healthcheck())

with tab6:
    st.header("Feedback de negocio")
    c1, c2 = st.columns(2)
    with c1:
        feedback_shop_label = st.selectbox("Tienda", make_options(batch_df, "shop_label", include_all=False), key="feedback_shop")
        feedback_shop = first_int_from_label(feedback_shop_label) or 0
    with c2:
        item_options = batch_df.query("shop_id == @feedback_shop")["item_label"].dropna().sort_values().unique().tolist()
        feedback_item_label = st.selectbox("Producto", item_options, key="feedback_item")
        feedback_item = first_int_from_label(feedback_item_label) or 0

    issue_type = st.selectbox("Tipo de observación", ["Predicción muy alta", "Predicción muy baja", "Producto descontinuado", "Otro"])
    analyst_name = st.text_input("Nombre del analista", value="")
    comment = st.text_area("Comentario")

    if st.button("Guardar feedback en RDS"):
        try:
            safe_insert_feedback(
                shop_id=int(feedback_shop),
                item_id=int(feedback_item),
                issue_type=issue_type,
                comment=comment,
                analyst_name=analyst_name or None,
            )
            st.success("Feedback guardado correctamente.")
        except RuntimeError as exc:
            st.warning(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.error(f"No se pudo guardar feedback: {exc}")

    st.subheader("Feedback capturado")
    feedback = safe_read_feedback(limit=100)
    st.dataframe(feedback, width="stretch") if not feedback.empty else st.info("Todavía no hay feedback visible.")

    st.subheader("Productos sugeridos para revisión")
    problem_products = safe_read_problem_products(limit=100)
    st.dataframe(problem_products, width="stretch") if not problem_products.empty else st.info("Sin productos problemáticos visibles.")
