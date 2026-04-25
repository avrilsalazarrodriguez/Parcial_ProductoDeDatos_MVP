"""
Aplicación Streamlit para el MVP de pronóstico de ventas de 1C Company.

Esta app reutiliza los artefactos construidos en las tareas 1 a 7:
- artifacts/model.joblib
- data/prep/valid.parquet
- data/prep/test_features.parquet
- data/prep/test_pairs.parquet
- data/predictions/submission.csv

La primera versión corre con archivos locales para validar la UI.

Después se conectará a RDS y Secrets Manager para cumplir la arquitectura final.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# Aqui definimos rutas reales del repositorio.
MODEL_PATH = Path("artifacts/model.joblib")
VALID_PATH = Path("data/prep/valid.parquet")
TEST_FEATURES_PATH = Path("data/prep/test_features.parquet")
TEST_PAIRS_PATH = Path("data/prep/test_pairs.parquet")
SUBMISSION_PATH = Path("data/predictions/submission.csv")


@st.cache_data
def load_valid_data() -> pd.DataFrame:
    """Aqui cargamos el conjunto de validación con ground truth real."""
    return pd.read_parquet(VALID_PATH)


@st.cache_data
def load_test_features() -> pd.DataFrame:
    """Aqui cargamos features del mes futuro que usará la app."""
    return pd.read_parquet(TEST_FEATURES_PATH)


@st.cache_data
def load_test_pairs() -> pd.DataFrame:
    """Aqui cargamos los pares tienda-producto asociados al test."""
    return pd.read_parquet(TEST_PAIRS_PATH)


@st.cache_data
def load_submission() -> pd.DataFrame:
    """Aqui cargamos las predicciones batch ya generadas por el pipeline previo."""
    return pd.read_csv(SUBMISSION_PATH)


@st.cache_resource
def load_model() -> dict:
    """Aqui cargamos el modelo entrenado una sola vez para inferencia individual."""
    return joblib.load(MODEL_PATH)


def predict_with_model(model_payload: dict, features: pd.DataFrame) -> np.ndarray:
    """
    Aqui ejecutamos inferencia usando el modelo existente de dos etapas.
    
    - clasificador para probabilidad de venta
    - regresor para unidades condicionadas a venta
    """
    bundle = model_payload["bundle"]
    feature_cols = bundle["feature_cols"]

    x_test = features[feature_cols]
    prob = bundle["clf"].predict_proba(x_test)[:, 1].astype(np.float32)
    mu = bundle["reg"].predict(x_test).astype(np.float32)

    # Aqui limitamos al rango usado en el entrenamiento del proyecto.
    return np.clip(prob * mu, 0, 20)


def build_evaluation_sample(valid_df: pd.DataFrame, model_payload: dict, n: int = 3000) -> pd.DataFrame:
    """
    Generamos una muestra de evaluación para comparar predicción vs ground truth.
    
    Usamos una muestra para que la app responda rápido.
    """
    sample = valid_df.sample(min(n, len(valid_df)), random_state=42).copy()
    sample["prediction"] = predict_with_model(model_payload, sample)
    sample["error"] = sample["prediction"] - sample["y"]
    sample["abs_error"] = sample["error"].abs()
    return sample


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Calculamos RMSE para reportar el error del modelo."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def naive_prediction(valid_df: pd.DataFrame) -> pd.Series:
    """
    Construimos un baseline naive simple.

    Para este MVP usamos cnt_lag_1 como predicción naive:
    lo que se vendió el mes anterior se usa como pronóstico del mes actual.
    """
    if "cnt_lag_1" not in valid_df.columns:
        return pd.Series(np.zeros(len(valid_df)), index=valid_df.index)
    return valid_df["cnt_lag_1"].clip(0, 20)


st.set_page_config(
    page_title="1C Company - Pronóstico de Ventas",
    page_icon="📦",
    layout="wide",
)

st.title("📦 Producto de Datos — Pronóstico de Ventas")
st.caption("MVP Streamlit para consultar pronósticos, evaluación del modelo y feedback de negocio.")

# Aqui cargamos los datos principales de la app.
valid_df = load_valid_data()
test_features = load_test_features()
test_pairs = load_test_pairs()
submission = load_submission()
model_payload = load_model()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Resumen",
        "Inferencia individual",
        "Batch",
        "Evaluación",
        "KPIs",
        "Feedback",
    ]
)

with tab1:
    st.header("Resumen ejecutivo")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Filas validación", f"{len(valid_df):,}")
    col2.metric("Pares test", f"{len(test_pairs):,}")
    col3.metric("Predicciones batch", f"{len(submission):,}")
    col4.metric("Modelo", "LightGBM 2 etapas")

    st.write("Esta vista resume el producto mínimo viable para que negocio consulte pronósticos sin abrir notebooks.")

    st.dataframe(submission.head(20), use_container_width=True)

with tab2:
    st.header("Inferencia individual")

    st.write("Selecciona una tienda y producto existentes en el conjunto de test.")

    pairs_preview = test_pairs.copy()
    pairs_preview["pair_label"] = (
        "shop_id=" + pairs_preview["shop_id"].astype(str)
        + " | item_id=" + pairs_preview["item_id"].astype(str)
    )

    selected_label = st.selectbox(
        "Par tienda-producto",
        options=pairs_preview["pair_label"].head(5000).tolist(),
    )

    selected_index = pairs_preview.index[pairs_preview["pair_label"] == selected_label][0]
    selected_features = test_features.iloc[[selected_index]]

    if st.button("Ejecutar inferencia individual"):
        pred = predict_with_model(model_payload, selected_features)[0]
        st.metric("Pronóstico próximo mes", f"{pred:.2f} unidades")

        st.write("Features usadas para esta predicción:")
        st.dataframe(selected_features, use_container_width=True)

with tab3:
    st.header("Inferencia batch")

    st.write("Para el MVP, el batch se resuelve con predicciones precomputadas por el pipeline previo.")

    col1, col2 = st.columns(2)

    with col1:
        selected_shop = st.selectbox(
            "Filtrar por tienda",
            options=["Todas"] + sorted(test_pairs["shop_id"].unique().tolist()),
        )

    batch_df = test_pairs.copy()
    batch_df["prediction"] = submission.iloc[:, -1].values

    if selected_shop != "Todas":
        batch_df = batch_df.query("shop_id == @selected_shop")

    st.dataframe(batch_df.head(1000), use_container_width=True)

    csv = batch_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar pronóstico batch",
        data=csv,
        file_name="forecast_batch.csv",
        mime="text/csv",
    )

with tab4:
    st.header("Evaluación vs ground truth")

    eval_df = build_evaluation_sample(valid_df, model_payload)

    model_rmse = rmse(eval_df["y"], eval_df["prediction"])
    naive_rmse = rmse(eval_df["y"], naive_prediction(eval_df))

    col1, col2 = st.columns(2)
    col1.metric("RMSE modelo", f"{model_rmse:.4f}")
    col2.metric("RMSE naive", f"{naive_rmse:.4f}")

    fig = px.scatter(
        eval_df,
        x="y",
        y="prediction",
        hover_data=["shop_id", "item_id"],
        title="Predicción vs Ground Truth",
        labels={"y": "Ground truth", "prediction": "Predicción"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        eval_df[["shop_id", "item_id", "y", "prediction", "error", "abs_error"]].head(500),
        use_container_width=True,
    )

with tab5:
    st.header("KPIs por tienda y producto")

    eval_df = build_evaluation_sample(valid_df, model_payload)

    by_shop = (
        eval_df.groupby("shop_id", as_index=False)
        .agg(
            n=("y", "size"),
            y_mean=("y", "mean"),
            pred_mean=("prediction", "mean"),
            mae=("abs_error", "mean"),
        )
        .sort_values("mae", ascending=False)
    )

    by_item = (
        eval_df.groupby("item_id", as_index=False)
        .agg(
            n=("y", "size"),
            y_mean=("y", "mean"),
            pred_mean=("prediction", "mean"),
            mae=("abs_error", "mean"),
        )
        .sort_values("mae", ascending=False)
    )

    st.subheader("Error por tienda")
    st.dataframe(by_shop, use_container_width=True)

    st.subheader("Productos con mayor error")
    st.dataframe(by_item.head(50), use_container_width=True)

with tab6:
    st.header("Feedback de negocio")

    st.write("Esta versión local captura feedback en pantalla. En la versión AWS se guardará en RDS.")

    shop_id = st.number_input("shop_id", min_value=0, step=1)
    item_id = st.number_input("item_id", min_value=0, step=1)
    issue_type = st.selectbox(
        "Tipo de observación",
        ["Predicción muy alta", "Predicción muy baja", "Producto descontinuado", "Otro"],
    )
    comment = st.text_area("Comentario del analista")

    if st.button("Registrar observación"):
        st.success("Observación capturada en la UI. Siguiente paso: persistir en RDS.")
        st.write(
            {
                "shop_id": shop_id,
                "item_id": item_id,
                "issue_type": issue_type,
                "comment": comment,
            }
        )
