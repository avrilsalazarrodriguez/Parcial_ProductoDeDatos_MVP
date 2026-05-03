"""Aplicación Streamlit del MVP 1C Company con ModelOps v2.

Cambios incluidos:
- Usa ModelOps S3/local para Batch CFO y evaluación.
- Muestra nombres de tienda, producto, categoría y segmento.
- Evita JSON crudo: lo transforma en tarjetas y tablas.
- Explica ``model_scope`` para usuarios de negocio.
- Reincorpora registros sugeridos para revisión.
- Mantiene inferencia individual con ``artifacts/model.joblib``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from backend.storage import upload_batch_dataframe_to_s3
from frontend.modelops_sections import (
    get_batch_df_with_modelops_fallback,
    render_kpi_blocks,
    render_model_registry_block,
    render_model_scope_help,
    render_modelops_debug_block,
    render_modelops_evaluation_block,
    render_modelops_summary_block,
    render_review_suggestions_block,
)
from src.modelops_v2.catalogs import add_catalog_metadata, preferred_forecast_columns

try:
    from backend.db import (
        insert_batch_export,
        insert_business_feedback,
        insert_usage_event,
        read_batch_exports,
        read_business_feedback,
        read_problem_products,
    )
except Exception:  # noqa: BLE001
    insert_batch_export = None
    insert_business_feedback = None
    insert_usage_event = None
    read_batch_exports = None
    read_business_feedback = None
    read_problem_products = None


MODEL_PATH = Path("artifacts/model.joblib")
VALID_PATH = Path("data/prep/valid.parquet")
TEST_FEATURES_PATH = Path("data/prep/test_features.parquet")
TEST_PAIRS_PATH = Path("data/prep/test_pairs.parquet")
SUBMISSION_PATH = Path("data/predictions/submission.csv")
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
USE_MODELOPS_S3 = os.getenv("USE_MODELOPS_S3", "true").lower() == "true"
DISABLE_RDS_WRITES = os.getenv("DISABLE_RDS_WRITES", "false").lower() == "true"


st.set_page_config(
    page_title="1C Company - Pronóstico de Ventas",
    page_icon="📦",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_valid_data() -> pd.DataFrame:
    """Carga validación con ground truth real."""
    return pd.read_parquet(VALID_PATH)


@st.cache_data(show_spinner=False)
def load_test_features() -> pd.DataFrame:
    """Carga features del mes futuro."""
    return pd.read_parquet(TEST_FEATURES_PATH)


@st.cache_data(show_spinner=False)
def load_test_pairs() -> pd.DataFrame:
    """Carga pares tienda-producto del test."""
    return pd.read_parquet(TEST_PAIRS_PATH)


@st.cache_data(show_spinner=False)
def load_submission() -> pd.DataFrame:
    """Carga predicciones locales empaquetadas."""
    return pd.read_csv(SUBMISSION_PATH)


@st.cache_resource(show_spinner=False)
def load_model() -> dict[str, Any]:
    """Carga modelo local para inferencia individual."""
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def enrich_for_ui(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega nombres y etiquetas amigables."""
    return add_catalog_metadata(df, data_dir=DATA_DIR)


def safe_rds_write(fn: Any, **kwargs: Any) -> bool:
    """Escribe en RDS si está disponible; si no, no rompe la UI."""
    if DISABLE_RDS_WRITES or fn is None:
        return False
    try:
        fn(**kwargs)
        return True
    except Exception as exc:  # noqa: BLE001
        st.warning(f"No se pudo escribir en RDS. La app sigue funcionando. Detalle: {exc}")
        return False


def safe_rds_read(fn: Any, limit: int = 100) -> pd.DataFrame:
    """Lee de RDS si está disponible; si no, regresa tabla vacía."""
    if fn is None:
        return pd.DataFrame()
    try:
        return fn(limit=limit)
    except Exception as exc:  # noqa: BLE001
        st.caption(f"RDS no disponible para esta lectura: {exc}")
        return pd.DataFrame()


def predict_with_model(model_payload: dict[str, Any], features: pd.DataFrame) -> np.ndarray:
    """Inferencia individual compatible con el modelo original de dos etapas."""
    bundle = model_payload["bundle"]
    feature_cols = bundle["feature_cols"]
    x_test = features[feature_cols]
    prob = bundle["clf"].predict_proba(x_test)[:, 1].astype(np.float32)
    mu = bundle["reg"].predict(x_test).astype(np.float32)
    return np.clip((prob**0.90) * mu, 0, 20)


def build_evaluation_sample(valid_df: pd.DataFrame, model_payload: dict[str, Any], n: int = 5000) -> pd.DataFrame:
    """Muestra local de evaluación cuando ModelOps no está disponible."""
    sample = valid_df.sample(min(n, len(valid_df)), random_state=42).copy()
    sample["prediction"] = predict_with_model(model_payload, sample)
    sample["y_true"] = sample["y"]
    sample["error"] = sample["prediction"] - sample["y_true"]
    sample["abs_error"] = sample["error"].abs()
    if "cnt_lag_1" in sample.columns:
        sample["naive_prediction"] = sample["cnt_lag_1"].clip(0, 20)
    else:
        sample["naive_prediction"] = 0.0
    sample["model_scope"] = "global"
    return enrich_for_ui(sample)


def build_batch_table(test_pairs: pd.DataFrame, submission: pd.DataFrame, test_features: pd.DataFrame) -> pd.DataFrame:
    """Une pares tienda-producto con predicciones locales."""
    batch_df = test_pairs.copy()
    batch_df["prediction"] = submission.iloc[:, -1].values
    batch_df["prediction"] = batch_df["prediction"].clip(0, 20)
    batch_df["model_scope"] = "global"
    for col in ["recency", "months_active", "demand_tier", "price_tier", "item_category_id"]:
        if col in test_features.columns and col not in batch_df.columns:
            batch_df[col] = test_features[col].values
    return enrich_for_ui(batch_df)


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """RMSE local."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    """MAE local."""
    return float(np.mean(np.abs(y_true - y_pred)))


def select_shop_from_label(df: pd.DataFrame, label: str | None) -> int | None:
    """Extrae shop_id de un label seleccionado."""
    if label is None:
        return None
    row = df.loc[df["shop_label"] == label]
    if row.empty:
        return None
    return int(row.iloc[0]["shop_id"])


def select_item_from_label(df: pd.DataFrame, label: str | None) -> int | None:
    """Extrae item_id de un label seleccionado."""
    if label is None:
        return None
    row = df.loc[df["item_label"] == label]
    if row.empty:
        return None
    return int(row.iloc[0]["item_id"])


def render_sidebar() -> None:
    """Estado de conexiones."""
    st.sidebar.header("Estado de conexiones")
    st.sidebar.write(f"ModelOps S3: {'activo' if USE_MODELOPS_S3 else 'desactivado'}")
    st.sidebar.write(f"RDS writes: {'desactivado' if DISABLE_RDS_WRITES else 'activo si stack existe'}")
    if os.getenv("MODEL_BUCKET"):
        st.sidebar.caption(f"Bucket: {os.getenv('MODEL_BUCKET')}")
    if os.getenv("MODELOPS_PREFIX"):
        st.sidebar.caption(f"Prefix: {os.getenv('MODELOPS_PREFIX')}")


def render_pretty_forecast_table(df: pd.DataFrame, rows: int = 1000) -> None:
    """Muestra forecast con columnas amigables."""
    cols = preferred_forecast_columns(df)
    st.dataframe(df[cols].head(rows), width="stretch")


render_sidebar()

st.title("📦 Producto de Datos — Pronóstico de Ventas")
st.caption(
    "MVP Streamlit para consultar pronósticos mensuales, evaluar modelos, generar archivos CFO "
    "y capturar feedback operativo."
)

try:
    valid_df = load_valid_data()
    test_features = load_test_features()
    test_pairs = load_test_pairs()
    submission = load_submission()
    model_payload = load_model()
except Exception as exc:  # noqa: BLE001
    st.error(f"No pude cargar artefactos locales requeridos: {exc}")
    st.stop()

local_batch_df = build_batch_table(test_pairs, submission, test_features)
batch_df = get_batch_df_with_modelops_fallback(local_batch_df) if USE_MODELOPS_S3 else local_batch_df
batch_df = enrich_for_ui(batch_df)
eval_df = build_evaluation_sample(valid_df, model_payload)

TAB_NAMES = [
    "Resumen",
    "Inferencia individual",
    "Batch CFO",
    "Evaluación",
    "KPIs",
    "Feedback",
    "Model Registry",
]
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(TAB_NAMES)

with tab1:
    st.header("Resumen ejecutivo")
    st.write(
        "La aplicación permite consultar pronósticos mensuales por tienda y producto, "
        "revisar errores contra ground truth, generar archivos para finanzas y capturar "
        "observaciones del negocio para retraining o análisis posterior."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Filas de validación", f"{len(valid_df):,}")
    col2.metric("Pares tienda-producto", f"{len(test_pairs):,}")
    col3.metric("Predicciones batch", f"{len(batch_df):,}")
    col4.metric("Modelo en uso", "Champion ModelOps")

    st.subheader("Muestra de pronósticos disponibles")
    render_pretty_forecast_table(batch_df.head(20), rows=20)

    if USE_MODELOPS_S3:
        st.divider()
        render_modelops_summary_block()

    st.subheader("Dashboard rápido")
    col_a, col_b = st.columns(2)
    shop_label_col = "shop_label" if "shop_label" in batch_df.columns else "shop_id"
    top_shops = (
        batch_df.groupby(shop_label_col, as_index=False)
        .agg(total_forecast=("prediction", "sum"))
        .sort_values("total_forecast", ascending=False)
        .head(15)
    )
    fig_shop = px.bar(
        top_shops,
        x=shop_label_col,
        y="total_forecast",
        title="Tiendas con mayor pronóstico total",
        labels={shop_label_col: "Tienda", "total_forecast": "Unidades pronosticadas"},
    )
    col_a.plotly_chart(fig_shop, width="stretch")

    fig_dist = px.histogram(
        batch_df.sample(min(10000, len(batch_df)), random_state=42),
        x="prediction",
        nbins=30,
        title="Distribución de pronósticos",
        labels={"prediction": "Unidades pronosticadas"},
    )
    col_b.plotly_chart(fig_dist, width="stretch")

with tab2:
    st.header("Inferencia individual")
    st.write(
        "Selecciona tienda y producto con nombres legibles. La app carga `model.joblib` "
        "con cache y calcula el pronóstico en el momento."
    )

    pair_ui = enrich_for_ui(test_pairs.copy())
    shop_options = sorted(pair_ui["shop_label"].dropna().unique().tolist())
    selected_shop_label = st.selectbox("Tienda", options=shop_options)
    selected_shop = select_shop_from_label(pair_ui, selected_shop_label)

    available_items_df = pair_ui.query("shop_id == @selected_shop").copy()
    item_options = sorted(available_items_df["item_label"].dropna().unique().tolist())
    selected_item_label = st.selectbox("Producto", options=item_options)
    selected_item = select_item_from_label(available_items_df, selected_item_label)

    selected_rows = pair_ui.query("shop_id == @selected_shop and item_id == @selected_item")
    if selected_rows.empty:
        st.warning("No se encontró ese par tienda-producto en el conjunto futuro.")
    else:
        selected_index = selected_rows.index[0]
        selected_features = test_features.iloc[[selected_index]]
        pred = predict_with_model(model_payload, selected_features)[0]
        safe_rds_write(
            insert_usage_event,
            event_type="single_inference",
            shop_id=int(selected_shop),
            item_id=int(selected_item),
            records_count=1,
            status="success",
            message="Inferencia individual ejecutada correctamente.",
        )
        st.metric("Pronóstico próximo mes", f"{pred:.2f} unidades")
        st.table(
            pd.DataFrame(
                [
                    {"Campo": "Tienda", "Valor": selected_shop_label},
                    {"Campo": "Producto", "Valor": selected_item_label},
                ]
            )
        )
        with st.expander("Ver features usadas por el modelo"):
            st.dataframe(selected_features, width="stretch")

with tab3:
    st.header("Batch CFO")
    st.write(
        "Genera un archivo de pronósticos para una tienda, categoría o catálogo completo. "
        "La tabla incluye nombres para que finanzas pueda rastrear cada registro sin buscar IDs manualmente."
    )
    render_model_scope_help()

    scope = st.radio(
        "Alcance del archivo",
        ["Todos los productos de una tienda", "Todos los productos de una categoría", "Catálogo completo"],
        horizontal=True,
    )

    filtered_batch = batch_df.copy()
    selected_shop_batch: int | None = None
    selected_category_batch: str | None = None

    if scope == "Todos los productos de una tienda":
        shop_options = sorted(batch_df["shop_label"].dropna().unique().tolist())
        selected_shop_label_batch = st.selectbox("Selecciona tienda", options=shop_options, key="batch_shop")
        selected_shop_batch = select_shop_from_label(batch_df, selected_shop_label_batch)
        filtered_batch = batch_df.query("shop_id == @selected_shop_batch").copy()
        st.caption(f"Mostrando pronósticos de {selected_shop_label_batch}.")
    elif scope == "Todos los productos de una categoría":
        category_col = "item_category_name" if "item_category_name" in batch_df.columns else "category_group"
        category_options = sorted(batch_df[category_col].dropna().unique().tolist())
        selected_category_batch = st.selectbox("Selecciona categoría", options=category_options)
        filtered_batch = batch_df.query(f"{category_col} == @selected_category_batch").copy()
        st.caption(f"Mostrando pronósticos de la categoría {selected_category_batch}.")
    else:
        st.caption("Mostrando pronósticos del catálogo completo.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Registros del archivo", f"{len(filtered_batch):,}")
    col2.metric("Pronóstico total", f"{filtered_batch['prediction'].sum():,.1f}")
    col3.metric("Promedio por producto", f"{filtered_batch['prediction'].mean():.3f}")

    render_pretty_forecast_table(filtered_batch)
    csv = filtered_batch.to_csv(index=False).encode("utf-8")

    if st.button("Generar archivo CFO y guardar en S3"):
        try:
            s3_uri = upload_batch_dataframe_to_s3(
                df=filtered_batch,
                scope=scope,
                shop_id=selected_shop_batch,
            )
            safe_rds_write(
                insert_batch_export,
                scope=scope,
                shop_id=selected_shop_batch,
                records_count=len(filtered_batch),
                total_prediction=float(filtered_batch["prediction"].sum()),
                s3_uri=s3_uri,
            )
            st.success("Archivo CFO generado y guardado en S3.")
            st.code(s3_uri)
        except Exception as exc:  # noqa: BLE001
            st.error(f"No se pudo guardar el archivo CFO en S3: {exc}")

    st.download_button(
        "Descargar archivo CFO",
        data=csv,
        file_name="forecast_cfo_next_month.csv",
        mime="text/csv",
    )

    st.subheader("Historial de archivos generados")
    batch_exports = safe_rds_read(read_batch_exports, limit=20)
    if batch_exports.empty:
        st.info("No hay historial de exports en RDS o RDS no está disponible localmente.")
    else:
        st.dataframe(batch_exports, width="stretch")

with tab4:
    st.header("Evaluación vs ground truth")
    st.write(
        "Comparamos el champion contra el valor real del mes de validación y contra un baseline naive."
    )
    if USE_MODELOPS_S3:
        render_modelops_evaluation_block()
    else:
        model_rmse = rmse(eval_df["y_true"], eval_df["prediction"])
        naive_rmse = rmse(eval_df["y_true"], eval_df["naive_prediction"])
        model_mae = mae(eval_df["y_true"], eval_df["prediction"])
        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE modelo", f"{model_rmse:.4f}")
        col2.metric("RMSE naive", f"{naive_rmse:.4f}")
        col3.metric("MAE modelo", f"{model_mae:.4f}")
        render_pretty_forecast_table(eval_df, rows=500)

with tab5:
    st.header("KPIs por tienda, producto y segmento")
    st.write(
        "Estas vistas ayudan a identificar dónde falla más el modelo. Incluyen nombres legibles "
        "para que negocio pueda investigar cada punto."
    )
    if USE_MODELOPS_S3:
        render_kpi_blocks()
    else:
        by_shop = eval_df.groupby(["shop_id", "shop_name", "shop_label"], as_index=False).agg(
            n=("y_true", "size"),
            y_mean=("y_true", "mean"),
            pred_mean=("prediction", "mean"),
            mae=("abs_error", "mean"),
        ).sort_values("mae", ascending=False)
        by_item = eval_df.groupby(["item_id", "item_name", "item_label"], as_index=False).agg(
            n=("y_true", "size"),
            y_mean=("y_true", "mean"),
            pred_mean=("prediction", "mean"),
            mae=("abs_error", "mean"),
        ).sort_values("mae", ascending=False)
        st.dataframe(by_shop.head(100), width="stretch")
        st.dataframe(by_item.head(100), width="stretch")

with tab6:
    st.header("Feedback de negocio")
    st.write(
        "Captura observaciones del negocio y muestra registros sugeridos para revisión del equipo ML."
    )

    suggestions = render_review_suggestions_block(limit=100)
    st.divider()

    st.subheader("Registrar nueva observación")
    feedback_source = suggestions if not suggestions.empty else batch_df
    shop_options = sorted(feedback_source["shop_label"].dropna().unique().tolist()) if "shop_label" in feedback_source.columns else []

    selected_shop_feedback = st.selectbox("Tienda", options=shop_options) if shop_options else None
    selected_shop_id = select_shop_from_label(feedback_source, selected_shop_feedback) if selected_shop_feedback else 0

    item_df = feedback_source.query("shop_id == @selected_shop_id") if selected_shop_feedback and "shop_id" in feedback_source.columns else feedback_source
    item_options = sorted(item_df["item_label"].dropna().unique().tolist()) if "item_label" in item_df.columns else []
    selected_item_feedback = st.selectbox("Producto", options=item_options) if item_options else None
    selected_item_id = select_item_from_label(item_df, selected_item_feedback) if selected_item_feedback else 0

    issue_type = st.selectbox(
        "Tipo de observación",
        ["Predicción muy alta", "Predicción muy baja", "Producto descontinuado", "Baseline naive mejor", "Otro"],
    )
    analyst_name = st.text_input("Nombre del analista", value="")
    comment = st.text_area("Comentario del analista")

    if st.button("Guardar feedback en RDS"):
        ok = safe_rds_write(
            insert_business_feedback,
            shop_id=int(selected_shop_id),
            item_id=int(selected_item_id),
            issue_type=issue_type,
            comment=comment,
            analyst_name=analyst_name or None,
        )
        if ok:
            st.success("Feedback guardado correctamente en RDS.")
        else:
            st.info("Feedback no se guardó porque RDS no está disponible o está desactivado en este ambiente.")

    st.divider()
    st.subheader("Feedback capturado en RDS")
    feedback_df = safe_rds_read(read_business_feedback, limit=100)
    if feedback_df.empty:
        st.info("Todavía no hay observaciones guardadas o RDS no está disponible localmente.")
    else:
        st.dataframe(feedback_df, width="stretch")

    st.subheader("Productos problemáticos desde RDS")
    problem_df = safe_rds_read(read_problem_products, limit=100)
    if problem_df.empty:
        st.info("No hay registros en problem_products de RDS. Usa la lista sugerida de ModelOps arriba.")
    else:
        st.dataframe(problem_df, width="stretch")

with tab7:
    st.header("Model Registry")
    st.write(
        "Todos los modelos entrenados se muestran ordenados desde el champion hasta el menor performance."
    )
    render_model_registry_block()
    with st.expander("Debug ModelOps S3"):
        render_modelops_debug_block()
