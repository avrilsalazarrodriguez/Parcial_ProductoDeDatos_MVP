"""Clean Streamlit MVP for sales forecasting.

This is the stable branch version:
- no dependency on frontend/modelops_sections.py;
- no pandas `.query()` with unsupported expressions;
- no `height=None` passed to Streamlit;
- RMSE is the model selection metric, MAE is the tie breaker;
- tables show shop_id/shop_name and item_id/item_name, not extra *_label columns;
- filters still allow dropdowns with "id — name" plus manual IDs;
- Model Registry, feedback, overestimated and underestimated products are displayed.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.batch_upload_inference import render_uploaded_batch_inference
from frontend.low_activity_kpis import render_low_activity_products_table

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from backend.modelops_s3 import (  # noqa: E402
    load_overestimated_products,
    load_underestimated_products,
    load_champion,
    load_evaluation_by_item,
    load_evaluation_by_segment,
    load_evaluation_curves_by_model,
    load_evaluation_detail,
    load_forecast_detail,
    load_forecast_summary_by_category,
    load_forecast_summary_by_shop_segment,
    load_model_metrics,
    load_model_runs,
    load_review_suggestions,
    modelops_healthcheck,
)

try:  # noqa: E402
    from backend.storage import upload_batch_dataframe_to_s3
except Exception:  # noqa: BLE001
    upload_batch_dataframe_to_s3 = None

try:  # noqa: E402
    from frontend.cfo_s3_exports import upload_cfo_dataframe_partitioned_to_s3
except Exception:  # noqa: BLE001
    upload_cfo_dataframe_partitioned_to_s3 = None

try:  # noqa: E402
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

LOGGER = logging.getLogger(__name__)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
VALID_PATH = Path(os.getenv("VALID_PATH", "data/prep/valid.parquet"))
TEST_FEATURES_PATH = Path(os.getenv("TEST_FEATURES_PATH", "data/prep/test_features.parquet"))
TEST_PAIRS_PATH = Path(os.getenv("TEST_PAIRS_PATH", "data/prep/test_pairs.parquet"))
SUBMISSION_PATH = Path(os.getenv("SUBMISSION_PATH", "data/predictions/submission.csv"))
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
USE_MODELOPS_S3 = os.getenv("USE_MODELOPS_S3", "true").lower() == "true"
DISABLE_RDS_WRITES = os.getenv("DISABLE_RDS_WRITES", "false").lower() == "true"

UNKNOWN_TEXT = {"", "none", "nan", "null", "unknown", "noce", "n/a", "na"}

DEFAULT_SHOP_NAMES = {
    0: "!Yakutsk Ordzhonikidze, 56 fran", 1: "!Yakutsk TC Central fran", 2: "Adygea TC Mega",
    3: "Balashikha TRC October-Kinomir", 4: "Volzhsky TC Volga Mall", 5: "Vologda SEC Marmelad",
    6: "Voronezh Plekhanovskaya 13", 7: "Voronezh TRC Maksimir", 8: "Voronezh TRC City-Park Grad",
    9: "Outbound Trade", 10: "Zhukovsky st. Chkalov 39m", 11: "Zhukovsky st. Chkalov 39m²",
    12: "Online shop emergencies", 13: "Kazan TC Behetle", 14: "Kazan TC ParkHouse II",
    15: "Kaluga TRC XXI century", 16: "Kolomna TC Rio", 17: "Krasnoyarsk TC Vzletka Plaza",
    18: "Krasnoyarsk TC June", 19: "Kursk TC Pushkinsky", 20: "Moscow Sale", 21: "Moscow MTRC Afi Mall",
    22: "Moscow Shop C21", 23: "Moscow TC Budenovskiy pav. A2", 24: "Moscow TC Budenovskiy pav. K7",
    25: "Moscow TRC Atrium", 26: "Moscow TC Areal Belyaevo", 27: "Moscow TC MEGA Belaya Dacha II",
    28: "Moscow TC MEGA Teply Stan II", 29: "Moscow TC New Century Novokosino", 30: "Moscow TC Perlovskiy",
    31: "Moscow TC Semenovskiy", 32: "Moscow TC Serebryany Dom", 33: "Mytishchi TRK XL-3",
    34: "N. Novgorod TRC RIO", 35: "N. Novgorod TRC Fantasy", 36: "Novosibirsk SEC Gallery Novosibirsk",
    37: "Novosibirsk TC Mega", 38: "Omsk TC Mega", 39: "Rostov-on-Don TRK Megacenter Horizont",
    40: "Rostov-on-Don TRK Megacenter Horizont Ostrovnoy", 41: "Rostov-on-Don TC Mega",
    42: "SPb TC Nevsky Center", 43: "SPb TK Sennaya", 44: "Samara TC Melody", 45: "Samara TC ParkHouse",
    46: "Sergiev Posad TC 7Ya", 47: "Surgut SEC City Mall", 48: "Tomsk SEC Emerald City", 49: "Tyumen SEC Crystal",
    50: "Tyumen TC Goodwin", 51: "Tyumen TC Green Coast", 52: "Ufa TC Central", 53: "Ufa TC Family 2",
    54: "Khimki TC Mega", 55: "Digital warehouse 1C-Online", 56: "Chekhov SEC Carnival",
    57: "Yakutsk Ordzhonikidze, 56", 58: "Yakutsk TC Central", 59: "Yaroslavl TC Altair",
}

MODEL_DESCRIPTIONS = {
    "hybrid_router_v1": (
        "Router híbrido final: no es un único modelo, sino una política de enrutamiento. "
        "Usa reglas de inactividad, naive para demanda reciente, especialista recurrente y Hurdle HGB según features históricas."
    ),
    "hurdle_hgb": (
        "Hurdle HGB de dos etapas: primero estima probabilidad de venta y después unidades esperadas. "
        "Es útil para demanda intermitente con muchos ceros."
    ),
    "incumbent_two_stage": (
        "LightGBM original de dos etapas: modelo original del proyecto con tratamientos y transformaciones previas. "
        "Se conserva como incumbent para comparar contra los challengers."
    ),
    "original_lightgbm_two_stage": (
        "LightGBM original de dos etapas: clasificador venta/no venta + regresor de unidades, con el pipeline original de features."
    ),
    "hgb_poisson": (
        "HistGradientBoosting con pérdida Poisson: modelo para conteos no negativos; útil cuando la variable objetivo es número de unidades."
    ),
    "poisson_hgb": (
        "HistGradientBoosting con pérdida Poisson: alternativa de conteo para demanda no negativa."
    ),
    "specialist_recurrent": (
        "Especialista de demanda recurrente: entrenado para producto-tienda con señales históricas de venta reciente o recurrente."
    ),
    "rolling_mean_3": (
        "Promedio móvil de los últimos 3 rezagos: baseline fuerte para productos donde la señal reciente importa."
    ),
    "item_mean_fallback": (
        "Promedio histórico por producto: baseline/fallback cuando el historial de tienda-producto es insuficiente."
    ),
    "naive_lag1": (
        "Naive último periodo: usa la venta del mes anterior. Es baseline obligatorio para demostrar mejora real."
    ),
    "naive_last_observed": (
        "Naive último periodo observado: baseline simple que conserva la señal de ventas recientes."
    ),
}



# ---------- Generic helpers ----------

def clean_text(value: object, fallback: str = "Sin información") -> str:
    if value is None or pd.isna(value):
        return fallback
    text = str(value).strip()
    if text.lower() in UNKNOWN_TEXT:
        return fallback
    return text


def is_generic_shop_name(value: object, shop_id: object | None = None) -> bool:
    text = clean_text(value, "").lower()
    if not text:
        return True
    if shop_id is not None and pd.notna(shop_id):
        return text in {f"tienda {int(shop_id)}", f"shop {int(shop_id)}"}
    return text.startswith("tienda ") or text.startswith("shop ")


def safe_float(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def metric_text(value: object, digits: int = 4) -> str:
    number = safe_float(value)
    if number is None:
        return "N/D"
    return f"{number:.{digits}f}"


def render_dataframe(df: pd.DataFrame, title: str | None = None, max_rows: int | None = None, height: int | None = None) -> None:
    if title:
        st.subheader(title)
    if df is None or df.empty:
        st.info("No hay datos disponibles.")
        return
    display = df.copy()
    if max_rows is not None:
        display = display.head(max_rows)
    kwargs: dict[str, Any] = {"width": "stretch"}
    if height is not None:
        kwargs["height"] = height
    st.dataframe(display, **kwargs)


def make_id_label(df: pd.DataFrame, id_col: str, name_col: str) -> pd.Series:
    return df.apply(lambda row: f"{int(row[id_col])} — {row[name_col]}" if pd.notna(row[id_col]) else str(row[name_col]), axis=1)


def parse_id_from_label(label: str | None) -> int | None:
    if label is None:
        return None
    token = str(label).split("—", 1)[0].strip().split()[-1]
    try:
        return int(token)
    except Exception:
        return None


def choose_id_with_dropdown_and_text(df: pd.DataFrame, id_col: str, name_col: str, title: str, key: str) -> int | None:
    if df.empty or id_col not in df.columns:
        return None
    source = df[[id_col, name_col]].dropna(subset=[id_col]).drop_duplicates().sort_values(id_col).copy()
    source["_option"] = make_id_label(source, id_col, name_col)
    selected_label = st.selectbox(title, source["_option"].tolist(), key=f"{key}_select")
    typed_id = st.text_input(f"O escribe {id_col}", value="", placeholder="Ej. 2", key=f"{key}_text")
    if typed_id.strip():
        try:
            return int(typed_id.strip())
        except ValueError:
            st.error(f"{id_col} debe ser numérico. Uso el selector.")
    return parse_id_from_label(selected_label)


# ---------- Data loading and enrichment ----------

@st.cache_data(ttl=900, show_spinner=False)
def load_catalogs() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = DATA_DIR / "raw"
    shop_paths = [raw_dir / "shops_en.csv", raw_dir / "shops.csv", DATA_DIR / "shops_en.csv", DATA_DIR / "shops.csv"]
    item_paths = [raw_dir / "items_en.csv", raw_dir / "items.csv", DATA_DIR / "items_en.csv", DATA_DIR / "items.csv"]
    cat_paths = [raw_dir / "item_categories_en.csv", raw_dir / "item_categories.csv", DATA_DIR / "item_categories_en.csv", DATA_DIR / "item_categories.csv"]

    shops = pd.DataFrame(columns=["shop_id", "shop_name"])
    items = pd.DataFrame(columns=["item_id", "item_name", "item_category_id", "item_category_name"])

    for path in shop_paths:
        if path.exists():
            shops = pd.read_csv(path)
            break
    if not shops.empty:
        name_col = "shop_name" if "shop_name" in shops.columns else next((c for c in shops.columns if "shop" in c.lower() and c != "shop_id"), None)
        if name_col is None:
            shops["shop_name"] = shops["shop_id"].map(lambda x: DEFAULT_SHOP_NAMES.get(int(x), f"Tienda {int(x)}") if pd.notna(x) else "Tienda sin ID")
        else:
            shops["shop_name"] = shops.apply(
                lambda row: DEFAULT_SHOP_NAMES.get(int(row["shop_id"]), clean_text(row[name_col], "Tienda sin nombre"))
                if pd.notna(row.get("shop_id")) and is_generic_shop_name(row[name_col], row.get("shop_id"))
                else clean_text(row[name_col], "Tienda sin nombre"),
                axis=1,
            )
        shops["shop_id"] = pd.to_numeric(shops["shop_id"], errors="coerce").astype("Int64")
        shops = shops[["shop_id", "shop_name"]].drop_duplicates("shop_id")

    for path in item_paths:
        if path.exists():
            items = pd.read_csv(path)
            break
    if not items.empty:
        name_col = "item_name" if "item_name" in items.columns else next((c for c in items.columns if "item" in c.lower() and c != "item_id"), None)
        items["item_name"] = items[name_col].map(lambda v: clean_text(v, "Producto sin nombre")) if name_col else items["item_id"].map(lambda x: f"Producto {x}")
        items["item_id"] = pd.to_numeric(items["item_id"], errors="coerce").astype("Int64")
        if "item_category_id" not in items.columns:
            items["item_category_id"] = pd.NA

        categories = pd.DataFrame()
        for path in cat_paths:
            if path.exists():
                categories = pd.read_csv(path)
                break
        if not categories.empty and "item_category_id" in categories.columns:
            cat_name_col = "item_category_name" if "item_category_name" in categories.columns else next((c for c in categories.columns if "category" in c.lower() and c != "item_category_id"), None)
            categories["item_category_name"] = categories[cat_name_col].map(lambda v: clean_text(v, "Categoría sin nombre")) if cat_name_col else categories["item_category_id"].map(lambda x: f"Categoría {x}")
            categories["item_category_id"] = pd.to_numeric(categories["item_category_id"], errors="coerce").astype("Int64")
            items = items.merge(categories[["item_category_id", "item_category_name"]], on="item_category_id", how="left")
        if "item_category_name" not in items.columns:
            items["item_category_name"] = "Sin categoría"
        items = items[["item_id", "item_name", "item_category_id", "item_category_name"]].drop_duplicates("item_id")

    return shops, items


def enrich_business_names(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()
    out = df.copy()
    shops, items = load_catalogs()

    # Remove label columns from visible tables. Selectboxes create labels on the fly.
    out = out.drop(columns=[c for c in ["shop_label", "item_label", "category_label", "segment_label"] if c in out.columns], errors="ignore")

    if "shop_id" in out.columns:
        out["shop_id"] = pd.to_numeric(out["shop_id"], errors="coerce").astype("Int64")
        if "shop_name" in out.columns:
            out = out.drop(columns=["shop_name"], errors="ignore")
        if not shops.empty:
            out = out.merge(shops, on="shop_id", how="left")
        if "shop_name" not in out.columns:
            out["shop_name"] = pd.NA
        out["shop_name"] = out.apply(
            lambda row: DEFAULT_SHOP_NAMES.get(int(row["shop_id"]), f"Tienda {int(row['shop_id'])}")
            if pd.notna(row.get("shop_id")) and is_generic_shop_name(row.get("shop_name"), row.get("shop_id"))
            else clean_text(row.get("shop_name"), "Tienda sin nombre"),
            axis=1,
        )

    if "item_id" in out.columns:
        out["item_id"] = pd.to_numeric(out["item_id"], errors="coerce").astype("Int64")
        if "item_category_id" in out.columns:
            out["__existing_item_category_id"] = out["item_category_id"]
        out = out.drop(
            columns=[c for c in ["item_name", "item_category_name", "item_category_id"] if c in out.columns],
            errors="ignore",
        )
        if not items.empty:
            out = out.merge(items, on="item_id", how="left")
        if "__existing_item_category_id" in out.columns:
            if "item_category_id" in out.columns:
                out["item_category_id"] = out["item_category_id"].fillna(out["__existing_item_category_id"])
            else:
                out["item_category_id"] = out["__existing_item_category_id"]
            out = out.drop(columns=["__existing_item_category_id"], errors="ignore")
        if "item_name" not in out.columns:
            out["item_name"] = pd.NA
        if "item_category_name" not in out.columns:
            out["item_category_name"] = "Sin categoría"
        out["item_name"] = out.apply(lambda row: clean_text(row.get("item_name"), f"Producto {row.get('item_id')}"), axis=1)
        out["item_category_name"] = out["item_category_name"].map(lambda v: clean_text(v, "Sin categoría"))

    if "category_group" not in out.columns and "item_category_name" in out.columns:
        out["category_group"] = out["item_category_name"].astype(str).str.split(" - ").str[0].replace("nan", "Sin grupo")
    if "recency" in out.columns and "recency_label" not in out.columns:
        rec = pd.to_numeric(out["recency"], errors="coerce")
        out["recency_label"] = rec.map(lambda x: "Sin ventas recientes" if pd.isna(x) or x >= 99 else f"{int(x)} meses")
    return out


def visible_forecast_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Columns shown in business-facing forecast tables.

    Decision columns are kept near the prediction so users can understand why a
    row used a particular component of the hybrid router.
    """
    out = add_intuitive_columns(df) if df is not None and not df.empty else df
    preferred = [
        "shop_id",
        "shop_name",
        "item_id",
        "item_name",
        "prediction",
        "decision_recommendation",
        "model_scope",
        "routing_reason",
        "model_id",
        "model_name",
        "item_category_id",
        "item_category_name",
        "category_group",
        "segment_key",
        "segment_name",
        "recency_label",
        "recency",
        "pred_champion",
        "pred_naive_lag1",
        "pred_rolling_mean_3",
        "pred_specialist_recurrent",
    ]
    cols = [c for c in preferred if c in out.columns]
    cols.extend([c for c in out.columns if c not in cols and not c.endswith("_label")])
    return out[cols]



@st.cache_data(ttl=600, show_spinner=False)
def load_valid_data() -> pd.DataFrame:
    return pd.read_parquet(VALID_PATH)


@st.cache_data(ttl=600, show_spinner=False)
def load_test_features() -> pd.DataFrame:
    return pd.read_parquet(TEST_FEATURES_PATH)


@st.cache_data(ttl=600, show_spinner=False)
def load_test_pairs() -> pd.DataFrame:
    return pd.read_parquet(TEST_PAIRS_PATH)


@st.cache_data(ttl=600, show_spinner=False)
def load_submission() -> pd.DataFrame:
    return pd.read_csv(SUBMISSION_PATH)


@st.cache_resource(show_spinner=False)
def load_model() -> dict:
    return joblib.load(MODEL_PATH)


@st.cache_data(ttl=600, show_spinner=False)
def load_batch_forecast() -> pd.DataFrame:
    try:
        return enrich_business_names(load_forecast_detail())
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("action=load_forecast_detail status=fallback error=%s", exc)
        pairs = load_test_pairs().copy()
        submission = load_submission()
        pairs["prediction"] = submission.iloc[:, -1].to_numpy()
        pairs["model_name"] = "Modelo local"
        pairs["model_scope"] = "original_two_stage"
        return enrich_business_names(pairs)


def predict_with_model(model_payload: dict, features: pd.DataFrame) -> np.ndarray:
    bundle = model_payload.get("bundle", model_payload)
    feature_cols = bundle["feature_cols"]
    x_test = features[feature_cols]
    prob = bundle["clf"].predict_proba(x_test)[:, 1].astype(np.float32)
    mu = bundle["reg"].predict(x_test).astype(np.float32)
    return np.clip(prob * mu, 0, 20)


def build_local_eval_sample(valid_df: pd.DataFrame, model_payload: dict, batch_df: pd.DataFrame) -> pd.DataFrame:
    """Build local evaluation sample, robust to missing metadata columns."""
    sample = valid_df.sample(min(5000, len(valid_df)), random_state=42).copy()
    sample["prediction"] = predict_with_model(model_payload, sample)
    sample["naive_prediction"] = sample["cnt_lag_1"].clip(0, 20) if "cnt_lag_1" in sample.columns else 0.0
    sample["abs_error"] = (sample["prediction"] - sample["y"]).abs()

    if {"shop_id", "item_id"}.issubset(sample.columns):
        dims_enriched = enrich_business_names(batch_df).copy()
        wanted_cols = [
            "shop_id",
            "shop_name",
            "item_id",
            "item_name",
            "item_category_id",
            "item_category_name",
            "category_group",
            "model_scope",
        ]
        existing_cols = [col for col in wanted_cols if col in dims_enriched.columns]
        existing_cols = list(dict.fromkeys(["shop_id", "item_id"] + existing_cols))
        dims = dims_enriched[existing_cols].drop_duplicates(["shop_id", "item_id"])
        sample = sample.merge(dims, on=["shop_id", "item_id"], how="left", suffixes=("", "_dim"))

    return enrich_business_names(sample)


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))



@st.cache_data(ttl=600, show_spinner=False)
def get_eval_df_cached() -> pd.DataFrame:
    """Cached local evaluation sample.

    This avoids rebuilding local predictions on every widget interaction.
    The cache is short-lived so new ModelOps/local files can still be picked up.
    """
    return build_local_eval_sample(load_valid_data(), load_model(), load_batch_forecast())

def current_champion() -> dict[str, Any]:
    return load_champion()


def current_metrics() -> dict[str, Any]:
    return load_model_metrics()



# ---------- Final visual/decision helpers ----------

MODEL_DESCRIPTIONS.update({
    "HistGradientBoosting Poisson": MODEL_DESCRIPTIONS["hgb_poisson"],
    "Especialista demanda recurrente": MODEL_DESCRIPTIONS["specialist_recurrent"],
    "Hybrid Router: inactive + naive + specialist + HGB": MODEL_DESCRIPTIONS["hybrid_router_v1"],
    "Rolling mean últimos 3 lags": MODEL_DESCRIPTIONS["rolling_mean_3"],
    "LightGBM original dos etapas": MODEL_DESCRIPTIONS["incumbent_two_stage"],
    "Hurdle HGB": MODEL_DESCRIPTIONS["hurdle_hgb"],
    "Naive último periodo": MODEL_DESCRIPTIONS["naive_lag1"],
})

MODEL_SCOPE_DESCRIPTIONS = {
    "inactive:no_recent_sales": "Regla de inactividad: producto-tienda sin ventas recientes o sin historial útil; predicción conservadora cercana a cero.",
    "baseline:naive_recent_demand": "Baseline naive: usa señal reciente, normalmente cnt_lag_1. Útil cuando el último periodo conserva mejor la demanda.",
    "specialist:recurrent_demand": "Especialista de demanda recurrente: modelo entrenado para productos con señales históricas de venta recurrente.",
    "challenger:hurdle_hgb": "Modelo Hurdle HGB: modelo de dos etapas para demanda baja e intermitente con muchos ceros.",
    "hybrid_router_v1": "Router híbrido: política que elige entre regla inactiva, naive, especialista recurrente y HGB.",
    "incumbent:lightgbm_two_stage": "Modelo original LightGBM de dos etapas con las transformaciones originales del proyecto.",
    "incumbent_two_stage": "Modelo original LightGBM de dos etapas con las transformaciones originales del proyecto.",
    "original_two_stage": "Diseño original de dos etapas: clasifica venta/no venta y estima unidades si hay venta.",
    "global": "Modelo global aplicado sin segmentación específica.",
    "fallback": "Regla de respaldo cuando no hay historial suficiente.",
}

MODEL_ID_DESCRIPTIONS = {
    "hybrid_router_v1": "Router híbrido: decide por fila si usar regla cero, naive, especialista recurrente o HGB con base en variables históricas.",
    "hurdle_hgb": "Hurdle HGB: dos etapas para demanda intermitente con muchos ceros.",
    "incumbent_two_stage": "LightGBM original de dos etapas: modelo original del proyecto con transformaciones previas.",
    "LightGBM original dos etapas": "LightGBM original de dos etapas: incumbent histórico del proyecto.",
    "original_lightgbm_two_stage": "LightGBM original de dos etapas: clasificador venta/no venta + regresor de unidades.",
    "naive_lag1": "Naive último periodo: usa la venta del último mes observado.",
    "Naive último periodo": "Naive último periodo: baseline simple para comparar calidad.",
    "rolling_mean_3": "Promedio móvil últimos 3 rezagos: baseline robusto para demanda reciente.",
    "Rolling mean últimos 3 lags": "Promedio móvil últimos 3 rezagos: baseline para demanda reciente.",
    "specialist_recurrent": "Especialista de demanda recurrente: entrenado para series con señales de ventas frecuentes.",
    "Especialista demanda recurrente": "Especialista de demanda recurrente: se usa cuando hay evidencia histórica de ventas recurrentes.",
    "hgb_poisson": "HistGradientBoosting Poisson: modelo de conteo no negativo para unidades vendidas.",
    "HistGradientBoosting Poisson": "HistGradientBoosting con pérdida Poisson para conteos de ventas.",
    "item_mean_fallback": "Promedio histórico por producto: fallback cuando el historial es limitado.",
    "Promedio histórico por producto": "Promedio histórico por producto: baseline/fallback interpretativo.",
    "Modelo ganador": "Modelo seleccionado como champion o componente final usado por el router.",
    "Real": "Valor real promedio observado en validación para el bucket de demanda.",
}


def first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return first existing column from candidates."""
    if df is None or df.empty:
        return None
    for col in candidates:
        if col in df.columns:
            return col
    return None


def short_text(value: object, max_len: int = 55) -> str:
    """Short label for plot axes."""
    if value is None or pd.isna(value):
        return "sin dato"
    text = str(value)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def metric_col_for_error(df: pd.DataFrame) -> str | None:
    """Best available error metric for evaluation tables/charts."""
    return first_existing_col(df, ["rmse", "mae", "abs_error"])


def category_display_series(df: pd.DataFrame) -> pd.Series:
    """Stable category label: ID first and short name when available."""
    if "item_category_id" in df.columns:
        base = "cat_" + df["item_category_id"].astype(str)
    elif "segment_key" in df.columns:
        base = df["segment_key"].astype(str)
    else:
        base = pd.Series(["sin_categoria"] * len(df), index=df.index)

    for name_col in ["segment_name", "item_category_name", "category_group"]:
        if name_col in df.columns:
            return base + " — " + df[name_col].map(lambda x: short_text(x, 38))
    return base


def product_display_series(df: pd.DataFrame) -> pd.Series:
    """Stable product label: item_id + short name when available."""
    if "item_id" in df.columns:
        base = df["item_id"].astype(str)
    else:
        base = pd.Series(["sin_item"] * len(df), index=df.index)
    if "item_name" in df.columns:
        return base + " — " + df["item_name"].map(lambda x: short_text(x, 50))
    return base




def decision_label_from_row(row: pd.Series) -> str:
    """Human-readable final decision/model recommendation for a prediction row."""
    scope = str(row.get("model_scope", "") or "").strip()
    reason = str(row.get("routing_reason", "") or "").strip()
    model_id = str(row.get("model_id", "") or "").strip()
    text = f"{scope} {reason} {model_id}".lower()

    # Order matters: specialist scopes often contain routing reasons with
    # rolling_mean, but the final decision is the specialist model, not a
    # rolling baseline.
    if scope.startswith("inactive:") or "inactive" in text or "recency>=99" in text:
        return "Regla cero / producto inactivo"
    if scope.startswith("specialist:") or "specialist" in text or "recurrent" in scope.lower():
        return "Modelo especialista de demanda recurrente"
    if scope.startswith("baseline:naive") or "naive" in text:
        return "Baseline naive por demanda reciente"
    if scope.startswith("baseline:rolling") or "rolling" in text:
        return "Baseline promedio móvil"
    if "incumbent" in text or "lightgbm" in text or "original_two_stage" in text:
        return "Modelo original LightGBM dos etapas"
    if scope.startswith("challenger:hurdle") or "hurdle" in text:
        return "Modelo ganador Hurdle HGB"
    if scope:
        return f"Política/modelo: {scope}"
    if model_id:
        return f"Modelo registrado: {model_id}"

    # Aggregated evaluation/performance tables often do not have model_scope.
    # Avoid showing an empty/noisy decision in those cases.
    if any(col in row.index for col in ["rmse", "mae", "wape", "smape", "bias"]):
        return "Performance agregada del segmento/producto"
    if any(col in row.index for col in ["prediction", "pred_mean", "y", "y_true", "true_mean", "y_mean"]):
        return "Predicción evaluada del modelo actual"
    return "Sin decisión registrada"

def review_reason_from_row(row: pd.Series) -> str:
    """Human-readable reason for feedback/review tables."""
    if "review_reason" in row.index and pd.notna(row.get("review_reason")):
        value = str(row.get("review_reason")).strip()
        if value and value.lower() not in UNKNOWN_TEXT:
            return value
    error = row.get("error_signed", np.nan)
    abs_error = row.get("abs_error", np.nan)
    try:
        error_f = float(error)
    except Exception:
        error_f = np.nan
    try:
        abs_f = float(abs_error)
    except Exception:
        abs_f = np.nan

    if pd.notna(error_f) and error_f > 0:
        return "Sobreestimación: predicción mayor que el real"
    if pd.notna(error_f) and error_f < 0:
        return "Subestimación: predicción menor que el real"
    if pd.notna(abs_f) and abs_f > 0:
        return "Error absoluto alto en validación"

    scope = str(row.get("model_scope", "") or "")
    reason = str(row.get("routing_reason", "") or "")
    if scope or reason:
        return f"Revisión por {scope or 'política'} / {reason or 'sin razón detallada'}"
    return "Registro sugerido por ModelOps para revisión"


def add_intuitive_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add user-friendly decision/review/y columns without removing existing data."""
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()
    out = df.copy()
    if "decision_recommendation" not in out.columns:
        out["decision_recommendation"] = out.apply(decision_label_from_row, axis=1)
    if "review_reason" not in out.columns:
        out["review_reason"] = out.apply(review_reason_from_row, axis=1)
    if "y_mean" not in out.columns:
        for candidate in ["true_mean", "real_mean", "y"]:
            if candidate in out.columns:
                out["y_mean"] = out[candidate]
                break
    if "pred_mean" not in out.columns and "prediction" in out.columns:
        out["pred_mean"] = out["prediction"]
    return out

def decision_first_table(df: pd.DataFrame) -> pd.DataFrame:
    """Move business/decision columns to the front of a table."""
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()
    out = add_intuitive_columns(df)
    preferred = [
        "shop_id",
        "shop_name",
        "item_id",
        "item_name",
        "item_category_id",
        "item_category_name",
        "segment_key",
        "segment_name",
        "prediction",
        "decision_recommendation",
        "model_scope",
        "routing_reason",
        "review_reason",
        "model_id",
        "model_name",
        "y",
        "y_true",
        "y_mean",
        "true_mean",
        "real_mean",
        "pred_mean",
        "error_signed",
        "abs_error",
        "rmse",
        "mae",
        "wape",
        "smape",
        "bias",
        "nonzero_recall",
        "n",
        "pred_champion",
        "pred_naive_lag1",
        "pred_rolling_mean_3",
        "pred_specialist_recurrent",
    ]
    cols = [c for c in preferred if c in out.columns]
    cols.extend([c for c in out.columns if c not in cols and not c.endswith("_label")])
    return out[cols]



def performance_table(df: pd.DataFrame) -> pd.DataFrame:
    """Business-friendly performance table without routing decision columns.

    Performance/KPI rows are aggregates, so decision_recommendation and
    review_reason are intentionally hidden to avoid implying row-level routing.
    """
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()
    out = df.copy()

    if "y_mean" not in out.columns:
        for candidate in ["true_mean", "real_mean", "y"]:
            if candidate in out.columns:
                out["y_mean"] = out[candidate]
                break
    if "pred_mean" not in out.columns and "prediction" in out.columns:
        out["pred_mean"] = out["prediction"]

    out = out.drop(columns=["decision_recommendation", "review_reason"], errors="ignore")

    preferred = [
        "shop_id",
        "shop_name",
        "item_category_id",
        "item_category_name",
        "category_group",
        "segment_key",
        "segment_name",
        "item_id",
        "item_name",
        "n",
        "y_mean",
        "true_mean",
        "real_mean",
        "pred_mean",
        "prediction",
        "rmse",
        "mae",
        "wape",
        "smape",
        "bias",
        "nonzero_recall",
        "nonzero_precision",
    ]
    cols = [c for c in preferred if c in out.columns]
    cols.extend([c for c in out.columns if c not in cols and c not in {"decision_recommendation", "review_reason"} and not c.endswith("_label")])
    return out[cols]


def evaluation_detail_table(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluation sample table without decision/review columns."""
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()
    out = df.copy()

    if "y_mean" not in out.columns:
        for candidate in ["y", "y_true", "true_mean", "real_mean"]:
            if candidate in out.columns:
                out["y_mean"] = out[candidate]
                break
    if "pred_mean" not in out.columns and "prediction" in out.columns:
        out["pred_mean"] = out["prediction"]

    out = out.drop(columns=["decision_recommendation", "review_reason"], errors="ignore")

    preferred = [
        "shop_id",
        "shop_name",
        "item_id",
        "item_name",
        "item_category_id",
        "item_category_name",
        "y",
        "y_true",
        "y_mean",
        "prediction",
        "pred_mean",
        "naive_prediction",
        "abs_error",
        "error_signed",
        "model_scope",
        "routing_reason",
        "recency",
        "recency_label",
    ]
    cols = [c for c in preferred if c in out.columns]
    cols.extend([c for c in out.columns if c not in cols and c not in {"decision_recommendation", "review_reason"} and not c.endswith("_label")])
    return out[cols]



def render_decision_help(context: str = "") -> None:
    """Explain model_id, model_scope, routing_reason and decision_recommendation."""
    with st.expander(f"Cómo leer model_id, model_scope, routing_reason y decisión final{context}"):
        st.markdown(
            """
            - **decision_recommendation**: traducción operativa de la decisión final. Es la columna más directa para negocio.
            - **model_scope**: componente o política que decidió la predicción final de esa fila.
            - **routing_reason**: regla histórica usada para enrutar la fila. No usa el valor real futuro.
            - **model_id**: artefacto/familia registrada; puede seguir diciendo `hurdle_hgb` aunque el router haya decidido otro scope, porque el bundle base se registró con ese artefacto.
            - **prediction**: predicción final consumida por CFO/app.
            - **pred_champion / pred_naive_lag1 / pred_rolling_mean_3 / pred_specialist_recurrent**: candidatos internos cuando están disponibles.
            """
        )




DECISION_RECOMMENDATION_DESCRIPTIONS = {
    "Regla cero / producto inactivo": "Se predice cero porque el historial reciente sugiere producto-tienda inactivo o sin ventas recientes.",
    "Baseline naive por demanda reciente": "Se usa el último valor observado o señal reciente porque en demanda recurrente el naive puede superar al modelo global.",
    "Baseline promedio móvil": "Se usa un promedio móvil histórico como baseline cuando la señal reciente es más estable que el modelo global.",
    "Modelo especialista de demanda recurrente": "Se usa el modelo especialista recurrente (`pred_specialist_recurrent`): no es una media móvil simple. Se activa por señales históricas como `rolling_mean` o `nonzero_rate`, pero la predicción final viene del especialista entrenado para demanda recurrente.",
    "Modelo original LightGBM dos etapas": "Se usa/consulta el modelo original de dos etapas del proyecto, conservado como incumbent.",
    "Modelo ganador Hurdle HGB": "Se usa el modelo Hurdle HGB de dos etapas, fuerte para demanda intermitente con muchos ceros.",
    "Performance agregada del segmento/producto": "Fila agregada de evaluación. No representa una decisión individual de ruteo, sino desempeño por segmento/producto.",
    "Predicción evaluada del modelo actual": "Fila de evaluación local o feedback donde no hay ruteo explícito; la predicción corresponde al modelo evaluado.",
    "Sin decisión registrada": "El artefacto no incluye model_scope/routing_reason suficientes para explicar el ruteo.",
}


def decision_recommendation_meaning(value: object) -> str:
    text = str(value or "").strip()
    if text in DECISION_RECOMMENDATION_DESCRIPTIONS:
        return DECISION_RECOMMENDATION_DESCRIPTIONS[text]
    if text.startswith("Política/modelo:"):
        return "Scope registrado por ModelOps; revisar model_scope/routing_reason para detalle operativo."
    if text.startswith("Modelo registrado:"):
        return "Modelo registrado en ModelOps; no se identificó una política específica para esa fila."
    return "Decisión generada a partir de model_scope, routing_reason y/o model_id."


def review_reason_meaning(value: object) -> str:
    text = str(value or "").strip()
    lower = text.lower()
    if "sobreestim" in lower:
        return "La predicción quedó por encima del real; revisar riesgo de sobreinventario."
    if "subestim" in lower:
        return "La predicción quedó por debajo del real; revisar riesgo de subabasto."
    if "inactive" in lower or "recency" in lower:
        return "Revisión por regla de inactividad o ausencia de ventas recientes."
    if "naive" in lower or "cnt_lag" in lower:
        return "Revisión por uso de señal reciente / baseline naive."
    if "specialist" in lower or "recurrent" in lower or "rolling_mean" in lower:
        return "Revisión por política de demanda recurrente o modelo especialista."
    if "hurdle" in lower or "sparse" in lower:
        return "Revisión por ruteo al modelo Hurdle para demanda baja/intermitente."
    if "error absoluto" in lower:
        return "Registro sugerido por error alto en validación."
    if text:
        return "Razón registrada por ModelOps para priorizar revisión del equipo de negocio/ML."
    return "Sin razón disponible en el artefacto."


def render_decision_recommendation_legend(df: pd.DataFrame, title: str = "Qué significa cada decision_recommendation") -> None:
    if df is None or df.empty or "decision_recommendation" not in df.columns:
        return
    values = [v for v in df["decision_recommendation"].dropna().astype(str).unique() if v.strip()]
    if not values:
        return
    rows = [
        {"decision_recommendation": value, "significado": decision_recommendation_meaning(value)}
        for value in sorted(values)
    ]
    st.markdown(f"**{title}**")
    st.dataframe(pd.DataFrame(rows), width="stretch")


def render_review_reason_legend(df: pd.DataFrame, title: str = "Qué significa cada review_reason") -> None:
    if df is None or df.empty or "review_reason" not in df.columns:
        return
    values = [v for v in df["review_reason"].dropna().astype(str).unique() if v.strip()]
    if not values:
        return
    rows = [
        {"review_reason": value, "significado": review_reason_meaning(value)}
        for value in values[:20]
    ]
    st.markdown(f"**{title}**")
    st.dataframe(pd.DataFrame(rows), width="stretch")

def render_model_scope_legend(scopes: pd.Series | list[str] | None = None) -> None:
    """Show model_scope labels below coverage charts/tables."""
    if scopes is None:
        keys = list(MODEL_SCOPE_DESCRIPTIONS)
    else:
        keys = [str(x) for x in pd.Series(scopes).dropna().astype(str).unique()]
    rows = []
    for key in keys:
        desc = MODEL_SCOPE_DESCRIPTIONS.get(key)
        if desc is None:
            if key.startswith("challenger:"):
                desc = "Modelo challenger usado para generar la predicción final."
            elif key.startswith("baseline:"):
                desc = "Baseline simple usado por la política de enrutamiento."
            elif key.startswith("specialist:"):
                desc = "Modelo especialista para un régimen o segmento de demanda."
            elif key.startswith("inactive:"):
                desc = "Regla de inactividad o producto sin ventas recientes."
            else:
                desc = "Scope registrado por ModelOps."
        rows.append({"model_scope": key, "significado": desc})
    if rows:
        st.markdown("**Qué significa cada model_scope**")
        st.dataframe(pd.DataFrame(rows), width="stretch")


def render_model_id_legend(model_ids: pd.Series | list[str] | None = None) -> None:
    """Show model descriptions below model comparison chart."""
    if model_ids is None:
        keys = list(MODEL_ID_DESCRIPTIONS)
    else:
        keys = [str(x) for x in pd.Series(model_ids).dropna().astype(str).unique()]
    rows = []
    for key in keys:
        if key.lower() == "real":
            continue
        desc = MODEL_ID_DESCRIPTIONS.get(key)
        if desc is None:
            if "naive" in key.lower():
                desc = "Baseline naive que usa señal reciente o último periodo."
            elif "hurdle" in key.lower():
                desc = "Modelo de dos etapas para demanda intermitente."
            elif "rolling" in key.lower():
                desc = "Promedio móvil usado como baseline."
            elif "specialist" in key.lower():
                desc = "Modelo especialista para un régimen de demanda."
            elif "lightgbm" in key.lower() or "incumbent" in key.lower():
                desc = "Modelo original LightGBM de dos etapas del proyecto."
            else:
                desc = "Modelo candidato registrado en ModelOps."
        rows.append({"modelo/serie": key, "significado": desc})
    if rows:
        st.markdown("**Qué representa cada modelo en la gráfica**")
        st.dataframe(pd.DataFrame(rows), width="stretch")


def render_summary_intro_cards() -> None:
    """Blue cards explaining how to use the dashboard."""
    st.write(
        "Este tablero resume el flujo operativo del producto de datos: consulta individual, "
        "batch CFO, evaluación, KPIs, feedback y registry."
    )
    c1, c2, c3 = st.columns(3)
    card_style = """
        <div style="background-color:#14314a; padding:18px; border-radius:12px; min-height:118px;">
        <span style="color:#4aa3ff; font-size:20px; font-weight:700;">{title}</span>
        <br><span style="font-size:18px; line-height:1.35;">{body}</span>
        </div>
    """
    c1.markdown(card_style.format(title="Resumen", body="volumen general, distribución de pronósticos y principales tiendas/categorías."), unsafe_allow_html=True)
    c2.markdown(card_style.format(title="Inferencia / Batch CFO", body="consulta un par tienda-producto o genera archivos descargables para finanzas."), unsafe_allow_html=True)
    c3.markdown(card_style.format(title="Evaluación / KPIs / Feedback", body="revisa errores, detecta productos problemáticos y captura observaciones."), unsafe_allow_html=True)
    st.caption("El objetivo es que negocio pueda entender dónde venderemos más, dónde falla el modelo y qué productos requieren revisión.")


def safe_category_summary(raw_summary: pd.DataFrame, batch_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Normalize category forecast summary across local and AWS schemas."""
    source = raw_summary.copy() if raw_summary is not None and not raw_summary.empty else pd.DataFrame()
    if source.empty and batch_df is not None and not batch_df.empty:
        source = batch_df.copy()

    if source.empty:
        return pd.DataFrame()

    df = source.copy()
    category_col = first_existing_col(
        df,
        ["item_category_name", "category_group", "segment_name", "segment_key", "item_category_id", "category_label"],
    )
    if category_col is None:
        return pd.DataFrame()

    if "total_prediction" in df.columns:
        df["__value"] = pd.to_numeric(df["total_prediction"], errors="coerce")
    elif "forecast_units" in df.columns:
        df["__value"] = pd.to_numeric(df["forecast_units"], errors="coerce")
    elif "total_forecast" in df.columns:
        df["__value"] = pd.to_numeric(df["total_forecast"], errors="coerce")
    elif {"avg_prediction", "n_items"}.issubset(df.columns):
        df["__value"] = pd.to_numeric(df["avg_prediction"], errors="coerce") * pd.to_numeric(df["n_items"], errors="coerce")
    elif {"mean_prediction", "n"}.issubset(df.columns):
        df["__value"] = pd.to_numeric(df["mean_prediction"], errors="coerce") * pd.to_numeric(df["n"], errors="coerce")
    else:
        value_col = first_existing_col(df, ["prediction", "forecast", "mean_prediction", "mean_forecast", "avg_prediction", "item_cnt_month"])
        if value_col is None:
            return pd.DataFrame()
        df["__value"] = pd.to_numeric(df[value_col], errors="coerce")

    df["category_display"] = df[category_col].astype(str).map(lambda x: short_text(x, 55))
    return (
        df.groupby("category_display", dropna=False, as_index=False)
        .agg(total_prediction=("__value", "sum"))
        .sort_values("total_prediction", ascending=False)
    )


def render_category_forecast_chart(batch_df: pd.DataFrame) -> None:
    """Summary chart for top forecast categories, robust to AWS/local schema."""
    st.markdown("**Categorías con mayor pronóstico**")
    try:
        raw_summary = load_forecast_summary_by_category()
    except Exception:
        raw_summary = pd.DataFrame()
    summary = safe_category_summary(raw_summary, batch_df=batch_df)
    if summary.empty:
        st.info("No encontré columnas suficientes para graficar categorías.")
        return
    plot = summary.head(12).sort_values("total_prediction", ascending=True)
    fig = px.bar(
        plot,
        x="total_prediction",
        y="category_display",
        orientation="h",
        title="Categorías con mayor pronóstico total",
        labels={"total_prediction": "Pronóstico total", "category_display": "Categoría"},
    )
    st.plotly_chart(fig, width="stretch", key="summary_top_categories_safe")
    st.caption("Lectura: identifica las categorías que concentran el mayor volumen pronosticado.")


def _valid_label_value(value: object) -> str | None:
    """Return a clean model label or None when value is missing/invalid."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "nat", "noce"}:
        return None
    return text


def _row_model_label(row: pd.Series, fallback: str = "Modelo ganador") -> str:
    """Pick the best model label from a curve row.

    Priority matters because AWS/hybrid outputs sometimes have model_name = NaN
    but model_id is still valid. This prevents the selector from showing `nan`.
    """
    for col in ["model_name", "model_id", "section", "model_scope"]:
        if col in row.index:
            label = _valid_label_value(row.get(col))
            if label is not None:
                return label
    return fallback


def normalize_curves_for_plot(curves_df: pd.DataFrame, eval_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize demand curves to long format: demand_bucket, series, mean_value.

    Important fixes:
    - If model_name is NaN but model_id exists, uses model_id.
    - Removes/renames invalid labels such as `nan`.
    - Aggregates duplicated bucket/series pairs to avoid vertical lines.
    - Keeps the historical model curves when they exist in the curves artifact.
    """
    if curves_df is not None and not curves_df.empty:
        df = curves_df.copy()

        if {"demand_bucket", "series", "mean_value"}.issubset(df.columns):
            out = df.copy()
            out["demand_bucket"] = out["demand_bucket"].astype(str)
            out["series_raw"] = out["series"].astype(str)
            out["mean_value"] = pd.to_numeric(out["mean_value"], errors="coerce")

            def _curve_label(row: pd.Series) -> str:
                raw = str(row.get("series_raw", "")).strip()
                raw_lower = raw.lower()
                if raw_lower in {"real", "real_mean", "y", "y_mean", "true", "true_mean"}:
                    return "Real"
                if raw_lower in {"pred", "prediction", "pred_mean", "forecast", "mean_prediction", "nan", "none", "null", ""}:
                    return _row_model_label(row, fallback="Modelo ganador")
                cleaned_raw = _valid_label_value(raw)
                return cleaned_raw or _row_model_label(row, fallback="Modelo ganador")

            out["series"] = out.apply(_curve_label, axis=1)
            out = out[["demand_bucket", "series", "mean_value"]].dropna(subset=["mean_value"])
            out = out[~out["series"].astype(str).str.lower().isin({"nan", "none", "null", ""})]
            return out.groupby(["demand_bucket", "series"], as_index=False).agg(mean_value=("mean_value", "mean"))

        if {"demand_bucket", "real_mean", "pred_mean"}.issubset(df.columns):
            # Fill a usable model label row-wise. Do not choose only the first
            # existing column globally, because model_name may exist but be NaN.
            df["__model_label"] = df.apply(lambda row: _row_model_label(row, fallback="Modelo ganador"), axis=1)
            grouped = df.groupby(["demand_bucket", "__model_label"], as_index=False).agg(
                real_mean=("real_mean", "mean"),
                pred_mean=("pred_mean", "mean"),
            )
            real = grouped.groupby("demand_bucket", as_index=False).agg(mean_value=("real_mean", "mean"))
            real["series"] = "Real"
            pred = grouped.rename(columns={"__model_label": "series", "pred_mean": "mean_value"})[
                ["demand_bucket", "series", "mean_value"]
            ]
            out = pd.concat([real[["demand_bucket", "series", "mean_value"]], pred], ignore_index=True)
            out["demand_bucket"] = out["demand_bucket"].astype(str)
            out["series"] = out["series"].astype(str)
            out["mean_value"] = pd.to_numeric(out["mean_value"], errors="coerce")
            out = out.dropna(subset=["mean_value"])
            out = out[~out["series"].astype(str).str.lower().isin({"nan", "none", "null", ""})]
            return out.groupby(["demand_bucket", "series"], as_index=False).agg(mean_value=("mean_value", "mean"))

    if eval_df.empty or not {"y", "prediction"}.issubset(eval_df.columns):
        return pd.DataFrame(columns=["demand_bucket", "series", "mean_value"])

    local = eval_df.copy()
    local["demand_bucket"] = pd.cut(local["y"], bins=[-0.1, 0, 1, 3, 7, 20], include_lowest=True).astype(str)
    agg = local.groupby("demand_bucket", as_index=False, observed=False).agg(
        real_mean=("y", "mean"),
        pred_mean=("prediction", "mean"),
        naive_mean=("naive_prediction", "mean") if "naive_prediction" in local.columns else ("prediction", "mean"),
    )
    out = agg.melt(
        id_vars="demand_bucket",
        value_vars=[c for c in ["real_mean", "pred_mean", "naive_mean"] if c in agg.columns],
        var_name="series",
        value_name="mean_value",
    )
    out["series"] = out["series"].replace({"real_mean": "Real", "pred_mean": "Modelo ganador", "naive_mean": "Naive"})
    return out

def ordered_curve_df(df: pd.DataFrame) -> pd.DataFrame:
    """Sort curve dataframe by demand bucket order."""
    out = df.copy()
    order = out["demand_bucket"].dropna().astype(str).drop_duplicates().tolist()
    out["demand_bucket"] = pd.Categorical(out["demand_bucket"].astype(str), categories=order, ordered=True)
    return out.sort_values(["demand_bucket", "series"])


def champion_series_name(champion: dict[str, Any], curves_long: pd.DataFrame) -> str | None:
    """Find champion/current model series in curves."""
    if curves_long.empty:
        return None
    available = set(curves_long["series"].astype(str).unique())
    candidates = [champion.get("model_id"), champion.get("model_name"), champion.get("display_name"), "Modelo ganador", "Predicho ganador", "champion", "hurdle_hgb", "hybrid_router_v1"]
    for candidate in candidates:
        if candidate is not None and str(candidate) in available:
            return str(candidate)
    non_real_non_naive = [s for s in available if s.lower() not in {"real", "real_mean", "naive", "naive_lag1"}]
    return sorted(non_real_non_naive)[0] if non_real_non_naive else None


def render_winner_curve(curves_df: pd.DataFrame, eval_df: pd.DataFrame, champion: dict[str, Any]) -> None:
    """First evaluation curve: Real vs predicted champion/current model only."""
    st.subheader("Real vs predicho del modelo ganador")
    curves_long = normalize_curves_for_plot(curves_df, eval_df)
    if curves_long.empty:
        st.info("No hay datos suficientes para graficar la curva del modelo ganador.")
        return
    champ = champion_series_name(champion, curves_long)
    if champ is None:
        st.info("No pude identificar la serie del modelo ganador.")
        return
    plot = curves_long[curves_long["series"].isin(["Real", "real_mean", champ])].copy()
    plot["series"] = plot["series"].replace({"real_mean": "Real", champ: "Modelo ganador"})
    plot = ordered_curve_df(plot)
    fig = px.line(
        plot,
        x="demand_bucket",
        y="mean_value",
        color="series",
        markers=True,
        title="Modelo ganador: promedio real vs promedio predicho",
        labels={"demand_bucket": "Rango de demanda real", "mean_value": "Unidades promedio", "series": "Serie"},
    )
    st.plotly_chart(fig, width="stretch", key="winner_curve_final")
    st.caption("Lectura: si la línea predicha queda por debajo de Real en alta demanda, hay riesgo de subabasto.")


def render_model_curve_comparison(curves_df: pd.DataFrame, eval_df: pd.DataFrame) -> None:
    """Second evaluation curve: Real + selectable model curves."""
    st.subheader("Comparación de curvas por modelo")
    curves_long = normalize_curves_for_plot(curves_df, eval_df)
    if curves_long.empty:
        st.info("No hay datos suficientes para comparar curvas por modelo.")
        return

    available_models = [
        s for s in curves_long["series"].dropna().astype(str).drop_duplicates().tolist()
        if s.lower() not in {"real", "real_mean"}
    ]
    # Keep a deterministic readable order, with important models first.
    priority_tokens = ["hurdle", "hybrid", "incumbent", "lightgbm", "poisson", "specialist", "rolling", "naive"]
    def _priority(name: str) -> tuple[int, str]:
        lower = name.lower()
        for idx, token in enumerate(priority_tokens):
            if token in lower:
                return (idx, name)
        return (len(priority_tokens), name)
    available_models = sorted(available_models, key=_priority)
    desired_extra = [m for m in available_models if any(token in m.lower() for token in ["specialist", "recurrent", "poisson", "hgb_poisson"])]
    if not desired_extra:
        st.caption("Nota: si no aparecen specialist_recurrent o hgb_poisson en el selector, sus curvas no están guardadas en evaluation_curves_by_model.parquet. El modelo puede aparecer en Registry, pero sin curva no se puede graficar exactamente.")

    selected_models = st.multiselect(
        "Modelos a mostrar en la comparación",
        options=available_models,
        default=available_models,
        key="eval_curve_model_selector_v3",
        help="Real siempre se muestra. Selecciona o quita modelos para comparar contra la curva real.",
    )

    selected_series = ["Real"] + selected_models
    plot = curves_long[curves_long["series"].isin(selected_series + ["real_mean"])].copy()
    plot["series"] = plot["series"].replace({"real_mean": "Real"})
    plot = ordered_curve_df(plot)

    fig = px.line(
        plot,
        x="demand_bucket",
        y="mean_value",
        color="series",
        markers=True,
        title="Comparación: Real vs modelos seleccionados",
        labels={"demand_bucket": "Rango de demanda real", "mean_value": "Unidades promedio", "series": "Serie"},
    )
    st.plotly_chart(fig, width="stretch", key="model_curve_comparison_final")
    st.info(
        "Cómo leerla: la línea Real es la referencia. Cada modelo seleccionado muestra su promedio predicho por bucket de demanda. "
        "Entre más cerca esté de Real, mejor calibrado está en ese rango. Usa el selector para aislar naive, LightGBM, Poisson, el especialista recurrente, el router híbrido o el modelo ganador."
    )
    render_model_id_legend(selected_models)


def render_current_and_naive_curve(curves_df: pd.DataFrame, eval_df: pd.DataFrame, champion: dict[str, Any]) -> None:
    """Third evaluation curve: Real + current/champion model + naive."""
    st.subheader("Real vs modelo utilizado y naive")
    curves_long = normalize_curves_for_plot(curves_df, eval_df)
    if curves_long.empty:
        st.info("No hay datos suficientes para graficar modelo utilizado y naive.")
        return
    champ = champion_series_name(champion, curves_long)
    available = curves_long["series"].astype(str).unique().tolist()
    naive = next((s for s in available if "naive" in s.lower()), None)
    selected = ["Real"]
    if champ is not None:
        selected.append(champ)
    if naive is not None and naive not in selected:
        selected.append(naive)
    plot = curves_long[curves_long["series"].isin(selected + ["real_mean"])].copy()
    rename = {"real_mean": "Real"}
    if champ is not None:
        rename[champ] = "Modelo utilizado"
    if naive is not None:
        rename[naive] = "Naive"
    plot["series"] = plot["series"].replace(rename)
    plot = ordered_curve_df(plot)
    fig = px.line(
        plot,
        x="demand_bucket",
        y="mean_value",
        color="series",
        markers=True,
        title="Real vs modelo utilizado y naive",
        labels={"demand_bucket": "Rango de demanda real", "mean_value": "Unidades promedio", "series": "Serie"},
    )
    st.plotly_chart(fig, width="stretch", key="current_vs_naive_curve_final")
    st.caption("Lectura: compara directamente el modelo utilizado contra el baseline naive usando la línea Real como referencia.")


def render_demand_bucket_distribution(eval_df: pd.DataFrame) -> None:
    """Show number of records by real-demand bucket."""
    st.subheader("Distribución de registros por rango de demanda real")
    if eval_df.empty or "y" not in eval_df.columns:
        st.info("No hay variable real para graficar distribución por rango.")
        return
    temp = eval_df.copy()
    temp["demand_bucket"] = pd.cut(temp["y"], bins=[-0.1, 0, 1, 3, 7, 20], include_lowest=True).astype(str)
    counts = temp.groupby("demand_bucket", as_index=False, observed=False).agg(n_registros=("y", "size"))
    fig = px.bar(
        counts,
        x="demand_bucket",
        y="n_registros",
        title="Número de registros por rango de demanda real",
        labels={"demand_bucket": "Rango de demanda real", "n_registros": "Número de registros"},
    )
    st.plotly_chart(fig, width="stretch", key="demand_bucket_distribution")
    st.caption("Lectura: permite ver que la mayoría de registros está en demanda cercana a cero, lo que explica por qué un modelo global puede favorecer muchos ceros.")


def prepare_segment_table(segment_source: pd.DataFrame, fallback_eval: pd.DataFrame) -> pd.DataFrame:
    """Build full segment performance table."""
    source = enrich_business_names(segment_source) if segment_source is not None and not segment_source.empty else pd.DataFrame()
    if source.empty:
        source = enrich_business_names(fallback_eval)
        group_cols = [col for col in ["item_category_id", "item_category_name", "category_group", "segment_key", "segment_name"] if col in source.columns]
        if group_cols and "abs_error" in source.columns:
            source = source.groupby(group_cols, dropna=False, as_index=False).agg(
                n=("abs_error", "size"),
                rmse=("abs_error", lambda x: float(np.sqrt(np.mean(np.square(x))))),
                mae=("abs_error", "mean"),
                pred_mean=("prediction", "mean"),
                true_mean=("y", "mean") if "y" in source.columns else ("abs_error", "mean"),
            )
    sort_col = "rmse" if "rmse" in source.columns else metric_col_for_error(source)
    if sort_col:
        source = source.sort_values(sort_col, ascending=False)
    return performance_table(source)


def prepare_item_table(item_source: pd.DataFrame, fallback_eval: pd.DataFrame) -> pd.DataFrame:
    """Build full item performance table."""
    source = enrich_business_names(item_source) if item_source is not None and not item_source.empty else pd.DataFrame()
    if source.empty:
        source = enrich_business_names(fallback_eval)
        group_cols = [col for col in ["item_id", "item_name", "item_category_id", "item_category_name"] if col in source.columns]
        if group_cols and "abs_error" in source.columns:
            source = source.groupby(group_cols, dropna=False, as_index=False).agg(
                n=("abs_error", "size"),
                rmse=("abs_error", lambda x: float(np.sqrt(np.mean(np.square(x))))),
                mae=("abs_error", "mean"),
                pred_mean=("prediction", "mean"),
                true_mean=("y", "mean") if "y" in source.columns else ("abs_error", "mean"),
            )
    sort_col = "rmse" if "rmse" in source.columns else metric_col_for_error(source)
    if sort_col:
        source = source.sort_values(sort_col, ascending=False)
    return performance_table(source)


def render_full_performance_tables(segment_df: pd.DataFrame, item_df: pd.DataFrame, eval_df: pd.DataFrame) -> None:
    """Navigable performance tables with filters."""
    st.subheader("Performance por segmento/categoría y producto")
    st.caption("Estas tablas son agregadas de performance; por eso no muestran decision_recommendation ni review_reason fila por fila.")
    seg = prepare_segment_table(segment_df, eval_df)
    item = prepare_item_table(item_df, eval_df)

    left, right = st.columns(2)
    with left:
        st.markdown("**Todos los segmentos/categorías**")
        if seg.empty:
            st.info("No hay tabla de segmentos/categorías disponible.")
        else:
            seg_filter = st.text_input("Buscar segmento/categoría por ID o nombre", value="", key="performance_segment_filter")
            show = seg.copy()
            if seg_filter.strip():
                mask = pd.Series(False, index=show.index)
                for col in ["segment_key", "segment_name", "item_category_id", "item_category_name", "category_group"]:
                    if col in show.columns:
                        mask = mask | show[col].astype(str).str.contains(seg_filter, case=False, na=False)
                show = show[mask]
            render_dataframe(show, height=430)
    with right:
        st.markdown("**Todos los productos**")
        if item.empty:
            st.info("No hay tabla de productos disponible.")
        else:
            item_filter = st.text_input("Buscar producto por item_id o nombre", value="", key="performance_item_filter")
            show = item.copy()
            if item_filter.strip():
                mask = pd.Series(False, index=show.index)
                for col in ["item_id", "item_name", "item_category_id", "item_category_name"]:
                    if col in show.columns:
                        mask = mask | show[col].astype(str).str.contains(item_filter, case=False, na=False)
                show = show[mask]
            render_dataframe(show, height=430)


def render_kpi_graphs(segment_df: pd.DataFrame, item_df: pd.DataFrame, eval_df: pd.DataFrame) -> None:
    """KPI charts with optional shop_id filter and readable labels."""
    st.header("KPIs")
    shop_filter = st.text_input("Filtrar KPIs por shop_id (opcional)", value="", key="kpi_shop_filter")
    base_eval = eval_df.copy()
    if shop_filter.strip() and "shop_id" in base_eval.columns:
        try:
            shop_id_value = int(shop_filter.strip())
            base_eval = base_eval[base_eval["shop_id"] == shop_id_value].copy()
            st.caption(f"KPIs filtrados por shop_id={shop_id_value}.")
        except ValueError:
            st.warning("shop_id debe ser numérico. Se muestran KPIs generales.")

    seg = prepare_segment_table(segment_df, base_eval)
    item = prepare_item_table(item_df, base_eval)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Categorías con mayor RMSE")
        if seg.empty or "rmse" not in seg.columns:
            st.info("No hay datos por categoría.")
        else:
            plot = seg.sort_values("rmse", ascending=False).head(15).copy()
            plot["category_display"] = category_display_series(plot)
            fig = px.bar(
                plot.sort_values("rmse", ascending=True),
                x="rmse",
                y="category_display",
                orientation="h",
                title="Top categorías por RMSE",
                labels={"rmse": "RMSE", "category_display": "Categoría"},
            )
            st.plotly_chart(fig, width="stretch", key="kpi_category_rmse_clean")
            render_dataframe(seg, max_rows=100, height=320)
    with col2:
        st.subheader("Productos con mayor RMSE")
        if item.empty or "rmse" not in item.columns:
            st.info("No hay datos por producto.")
        else:
            plot = item.sort_values("rmse", ascending=False).head(15).copy()
            plot["product_display"] = product_display_series(plot)
            fig = px.bar(
                plot.sort_values("rmse", ascending=True),
                x="rmse",
                y="product_display",
                orientation="h",
                title="Top productos por RMSE",
                labels={"rmse": "RMSE", "product_display": "Producto"},
            )
            st.plotly_chart(fig, width="stretch", key="kpi_product_rmse_clean")
            render_dataframe(item, max_rows=100, height=320)

    render_shop_kpi_block(base_eval)


def render_shop_kpi_block(eval_df: pd.DataFrame) -> None:
    """Shop-level RMSE chart and searchable table."""
    st.subheader("Tiendas con mayor RMSE")
    if eval_df is None or eval_df.empty or not {"shop_id", "prediction", "y"}.issubset(eval_df.columns):
        st.info("No hay datos suficientes para KPIs por tienda.")
        return
    df = enrich_business_names(eval_df).copy()
    df["error"] = pd.to_numeric(df["prediction"], errors="coerce") - pd.to_numeric(df["y"], errors="coerce")
    shop = (
        df.groupby([c for c in ["shop_id", "shop_name"] if c in df.columns], dropna=False, as_index=False)
        .agg(
            n=("error", "size"),
            rmse=("error", lambda x: float(np.sqrt(np.mean(np.square(x))))),
            mae=("error", lambda x: float(np.mean(np.abs(x)))),
            pred_mean=("prediction", "mean"),
            y_mean=("y", "mean"),
        )
        .sort_values("rmse", ascending=False)
    )
    plot = shop.head(15).sort_values("rmse", ascending=True)
    y_col = "shop_name" if "shop_name" in plot.columns else "shop_id"
    fig = px.bar(
        plot,
        x="rmse",
        y=y_col,
        orientation="h",
        title="Top tiendas por RMSE",
        labels={"rmse": "RMSE", y_col: "Tienda"},
    )
    st.plotly_chart(fig, width="stretch", key="kpi_shop_rmse_clean")
    shop_filter = st.text_input("Buscar tienda por shop_id o nombre", value="", key="kpi_shop_table_filter")
    show = shop.copy()
    if shop_filter.strip():
        mask = pd.Series(False, index=show.index)
        for col in ["shop_id", "shop_name"]:
            if col in show.columns:
                mask = mask | show[col].astype(str).str.contains(shop_filter, case=False, na=False)
        show = show[mask]
    render_dataframe(show, max_rows=200, height=320)


def build_extreme_errors(eval_detail_df: pd.DataFrame, fallback_eval: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build overestimated and underestimated tables from detailed evaluation."""
    source = eval_detail_df if eval_detail_df is not None and not eval_detail_df.empty else fallback_eval
    if source is None or source.empty:
        return pd.DataFrame(), pd.DataFrame()
    df = enrich_business_names(source).copy()
    y_col = "y" if "y" in df.columns else "y_true" if "y_true" in df.columns else "true" if "true" in df.columns else None
    pred_col = "prediction" if "prediction" in df.columns else "pred" if "pred" in df.columns else None
    if y_col is None or pred_col is None:
        return pd.DataFrame(), pd.DataFrame()
    df["error_signed"] = pd.to_numeric(df[pred_col], errors="coerce") - pd.to_numeric(df[y_col], errors="coerce")
    df["abs_error"] = df["error_signed"].abs()
    over = df.sort_values("error_signed", ascending=False).head(100)
    under = df.sort_values("error_signed", ascending=True).head(100)
    return decision_first_table(over), decision_first_table(under)

# ---------- App layout ----------

st.set_page_config(page_title="Producto de Datos — Pronóstico de Ventas", page_icon="📦", layout="wide")
st.title("📦 Producto de Datos — Pronóstico de Ventas")
st.caption("MVP Streamlit para consultar pronósticos mensuales, evaluar modelos, generar archivos CFO y capturar feedback operativo.")

with st.sidebar:
    st.subheader("Estado de conexiones")
    st.write("ModelOps S3:", "activo" if USE_MODELOPS_S3 else "local")
    st.write("RDS writes:", "desactivado" if DISABLE_RDS_WRITES else "activo")
    st.write("Root:", os.getenv("MODELOPS_LOCAL_ROOT", "modelops_outputs"))

TAB_NAMES = ["Resumen", "Inferencia individual", "Batch CFO", "Evaluación", "KPIs", "Feedback", "Model Registry"]

# `st.tabs` computes every tab on every rerun. A segmented/radio navigation keeps
# the same sections but renders only the selected one, which is much lighter in ECS.
try:
    active_tab = st.segmented_control(
        "Navegación principal",
        TAB_NAMES,
        default="Resumen",
        label_visibility="collapsed",
        key="main_navigation_section",
    )
except Exception:
    active_tab = st.radio(
        "Navegación principal",
        TAB_NAMES,
        index=0,
        horizontal=True,
        label_visibility="collapsed",
        key="main_navigation_section",
    )

valid_df = load_valid_data()
test_features = load_test_features()
test_pairs = load_test_pairs()
model_payload = load_model()
batch_df = load_batch_forecast()
if st.session_state.get("uploaded_batch_dashboard_active") and isinstance(st.session_state.get("uploaded_batch_dashboard_df"), pd.DataFrame):
    batch_df = enrich_business_names(st.session_state["uploaded_batch_dashboard_df"].copy())
    st.sidebar.success("Forecast del tablero: archivo cargado")
    st.sidebar.caption(str(st.session_state.get("uploaded_batch_dashboard_source", "uploaded batch")))
    if st.sidebar.button("Restaurar forecast ModelOps", key="restore_uploaded_forecast_sidebar"):
        for _key in ["uploaded_batch_dashboard_active", "uploaded_batch_dashboard_df", "uploaded_batch_dashboard_source"]:
            st.session_state.pop(_key, None)
        st.rerun()

# Only the evaluation/KPI/feedback sections need the local eval sample.
# Other sections skip this expensive local inference work.
eval_df = get_eval_df_cached() if active_tab in {"Evaluación", "KPIs", "Feedback"} else pd.DataFrame()
if active_tab == "Resumen":
    st.header("Resumen ejecutivo")
    render_summary_intro_cards()

    champion = current_champion()
    metrics = current_metrics()
    global_metrics = metrics.get("global") or metrics.get("champion") or {}
    model_name = champion.get("model_name") or champion.get("model_id") or global_metrics.get("model_name") or "Modelo local"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas de validación", f"{len(valid_df):,}")
    c2.metric("Pares tienda-producto", f"{len(test_pairs):,}")
    c3.metric("Predicciones batch", f"{len(batch_df):,}")
    c4.metric("Modelo en uso", model_name)

    st.subheader("Resumen ModelOps")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Champion", champion.get("model_run_id") or champion.get("model_id") or "sin registry")
    c2.metric("Modelo final", model_name)
    c3.metric("RMSE", metric_text(champion.get("rmse") or global_metrics.get("rmse")))
    c4.metric("MAE", metric_text(champion.get("mae") or global_metrics.get("mae")))
    st.caption(metrics.get("selection_policy") or champion.get("selection_policy") or "Selección: RMSE primero; MAE como desempate.")

    metric_cols = st.columns(4)
    metric_cols[0].metric("RMSE modelo", metric_text(global_metrics.get("rmse")))
    metric_cols[1].metric("MAE modelo", metric_text(global_metrics.get("mae")))
    metric_cols[2].metric("SMAPE", metric_text(global_metrics.get("smape")))
    metric_cols[3].metric("Bias", metric_text(global_metrics.get("bias")))

    render_dataframe(visible_forecast_columns(batch_df), "Muestra de pronósticos disponibles", max_rows=100, height=380)

    st.subheader("Gráficas resumen")
    col_a, col_b = st.columns(2)
    with col_a:
        top_shops = (
            batch_df.groupby(["shop_id", "shop_name"], dropna=False, as_index=False)
            .agg(total_forecast=("prediction", "sum"))
            .sort_values("total_forecast", ascending=False)
            .head(15)
        )
        fig = px.bar(top_shops.sort_values("total_forecast"), x="total_forecast", y="shop_name", orientation="h", title="Tiendas con mayor pronóstico total")
        st.plotly_chart(fig, width="stretch", key="summary_top_shops")
        st.caption("Lectura: tiendas con mayor volumen esperado para priorizar planeación comercial y CFO.")
    with col_b:
        dist = batch_df[["prediction"]].copy()
        dist["prediction_log1p"] = np.log1p(pd.to_numeric(dist["prediction"], errors="coerce").fillna(0).clip(lower=0))
        scale = st.radio("Escala de distribución", ["Original", "Log1p"], horizontal=True, key="summary_dist_scale")
        x_col = "prediction" if scale == "Original" else "prediction_log1p"
        fig = px.histogram(dist.sample(min(25_000, len(dist)), random_state=42), x=x_col, nbins=60, title="Distribución de pronósticos")
        st.plotly_chart(fig, width="stretch", key="summary_distribution")
        st.caption("Lectura: la escala Log1p ayuda a observar mejor una distribución muy concentrada cerca de cero.")

    col_c, col_d = st.columns(2)
    with col_c:
        render_category_forecast_chart(batch_df)
    with col_d:
        by_scope = batch_df.groupby("model_scope", dropna=False, as_index=False).agg(n=("prediction", "size"), total_prediction=("prediction", "sum")) if "model_scope" in batch_df.columns else pd.DataFrame()
        if not by_scope.empty:
            fig = px.pie(by_scope, values="n", names="model_scope", title="Cobertura por model_scope")
            st.plotly_chart(fig, width="stretch", key="summary_scope_pie")
            st.caption("Lectura: muestra qué proporción de predicciones viene del modelo global, segmentado o fallback.")
            render_model_scope_legend(batch_df["model_scope"])

elif active_tab == "Inferencia individual":
    st.header("Inferencia individual")
    st.write("Selecciona con la lista o escribe el ID manualmente.")
    col1, col2 = st.columns(2)
    with col1:
        selected_shop = choose_id_with_dropdown_and_text(batch_df, "shop_id", "shop_name", "Tienda", "single_shop")
    item_df = batch_df[batch_df["shop_id"] == selected_shop] if selected_shop is not None and "shop_id" in batch_df.columns else batch_df
    with col2:
        selected_item = choose_id_with_dropdown_and_text(item_df, "item_id", "item_name", "Producto", "single_item")

    rows = test_pairs[(test_pairs["shop_id"] == selected_shop) & (test_pairs["item_id"] == selected_item)] if selected_shop is not None and selected_item is not None else pd.DataFrame()
    if rows.empty:
        st.warning("No encontré ese par tienda-producto en el conjunto futuro.")
    else:
        features = test_features.iloc[[rows.index[0]]]
        pred = predict_with_model(model_payload, features)[0]
        st.metric("Pronóstico próximo mes", f"{pred:.2f} unidades")
        meta = enrich_business_names(pd.DataFrame([{"shop_id": selected_shop, "item_id": selected_item}]))
        render_dataframe(meta, "Par seleccionado")
        with st.expander("Ver features usadas"):
            st.dataframe(features, width="stretch")
        if insert_usage_event and not DISABLE_RDS_WRITES:
            try:
                insert_usage_event(event_type="single_inference", shop_id=int(selected_shop), item_id=int(selected_item), records_count=1, status="success", message="Inferencia individual")
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("usage event skipped: %s", exc)

elif active_tab == "Batch CFO":
    st.header("Batch CFO")
    with st.expander("¿Qué significa model_scope y routing_reason?"):
        st.markdown(
            """
            `model_scope` indica qué componente generó la predicción final. En el router híbrido puede ser:

            - `inactive:no_recent_sales`: regla para producto-tienda sin ventas recientes.
            - `baseline:naive_recent_demand`: usa señal reciente como `cnt_lag_1`.
            - `specialist:recurrent_demand`: modelo especialista para demanda recurrente.
            - `challenger:hurdle_hgb`: modelo Hurdle HGB para demanda baja/intermitente.
            - `incumbent:lightgbm_two_stage` / `incumbent_two_stage`: modelo LightGBM original de dos etapas.

            `routing_reason` explica la regla histórica usada para enrutar la fila, por ejemplo:
            `recency>=99_or_rolling_sum_6==0`, `cnt_lag_1>=1` o `rolling_mean_or_nonzero_rate_signal`.
            """
        )

    scope = st.radio("Alcance", ["Todos los productos de una tienda", "Segmento / categoría", "Catálogo completo"], horizontal=True)
    selected_category = None
    if scope == "Todos los productos de una tienda":
        selected_shop = choose_id_with_dropdown_and_text(batch_df, "shop_id", "shop_name", "Tienda", "cfo_shop")
        filtered_batch = batch_df[batch_df["shop_id"] == selected_shop].copy() if selected_shop is not None else batch_df.copy()
    elif scope == "Segmento / categoría":
        category_options = sorted(batch_df["item_category_id"].dropna().astype(int).astype(str).unique()) if "item_category_id" in batch_df.columns else []
        selected_category = st.selectbox("Selecciona item_category_id / segment_id", category_options)
        if category_options:
            category_series = pd.to_numeric(batch_df["item_category_id"], errors="coerce").astype("Int64").astype(str)
            filtered_batch = batch_df[category_series.eq(str(selected_category))].copy()
            st.caption(f"Segmento/categoría seleccionado: item_category_id={selected_category}")
        else:
            filtered_batch = batch_df.copy()
            st.info("No hay item_category_id disponible para filtrar por segmento/categoría.")
        selected_shop = None
    else:
        selected_shop = None
        filtered_batch = batch_df.copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Registros", f"{len(filtered_batch):,}")
    c2.metric("Pronóstico total", f"{filtered_batch['prediction'].sum():,.1f}")
    c3.metric("Promedio", f"{filtered_batch['prediction'].mean():.3f}")
    render_decision_help(" en Batch CFO")
    cfo_table = decision_first_table(visible_forecast_columns(filtered_batch))
    render_dataframe(cfo_table, "Vista previa del archivo CFO", max_rows=1000, height=420)
    render_decision_recommendation_legend(cfo_table, "Qué significa la recomendación de decisión en CFO")
    render_review_reason_legend(cfo_table, "Qué significa cada routing/review reason en CFO")

    csv = visible_forecast_columns(filtered_batch).to_csv(index=False).encode("utf-8")
    if st.button("Generar archivo CFO y guardar en S3"):
        if upload_batch_dataframe_to_s3 is None:
            st.warning("backend.storage.upload_batch_dataframe_to_s3 no está disponible.")
        else:
            try:
                shop_id_for_export = selected_shop if scope == "Todos los productos de una tienda" else None
                category_id_for_export = selected_category if scope == "Segmento / categoría" else None
                if upload_cfo_dataframe_partitioned_to_s3 is not None:
                    s3_uri = upload_cfo_dataframe_partitioned_to_s3(
                        filtered_batch,
                        scope=scope,
                        shop_id=shop_id_for_export,
                        category_id=category_id_for_export,
                    )
                else:
                    s3_uri = upload_batch_dataframe_to_s3(filtered_batch, scope=scope, shop_id=shop_id_for_export)
                if insert_batch_export and not DISABLE_RDS_WRITES:
                    insert_batch_export(scope=scope, shop_id=selected_shop if scope == "Todos los productos de una tienda" else None, records_count=len(filtered_batch), total_prediction=float(filtered_batch["prediction"].sum()), s3_uri=s3_uri)
                st.success("Archivo CFO guardado.")
                st.code(s3_uri)
            except Exception as exc:  # noqa: BLE001
                st.error(f"No se pudo guardar: {exc}")
    st.download_button("Descargar archivo CFO", data=csv, file_name="forecast_cfo_next_month.csv", mime="text/csv")

    if read_batch_exports and not DISABLE_RDS_WRITES:
        try:
            render_dataframe(read_batch_exports(limit=20), "Historial de archivos generados")
        except Exception:
            st.info("Sin historial disponible.")

    render_uploaded_batch_inference(
        model_payload=model_payload,
        enrich_fn=enrich_business_names,
        visible_fn=visible_forecast_columns,
        render_dataframe_fn=render_dataframe,
        upload_batch_dataframe_to_s3=upload_batch_dataframe_to_s3,
        insert_batch_export=insert_batch_export,
        disable_rds_writes=DISABLE_RDS_WRITES,
    )

elif active_tab == "Evaluación":
    st.header("Evaluación")
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE modelo local", f"{rmse(eval_df['y'], eval_df['prediction']):.4f}")
    c2.metric("RMSE naive", f"{rmse(eval_df['y'], eval_df['naive_prediction']):.4f}")
    c3.metric("MAE modelo local", f"{mae(eval_df['y'], eval_df['prediction']):.4f}")

    curves = load_evaluation_curves_by_model()
    champion = current_champion()
    render_winner_curve(curves, eval_df, champion)
    render_model_curve_comparison(curves, eval_df)
    render_demand_bucket_distribution(eval_df)

    segment_perf = load_evaluation_by_segment()
    item_perf = load_evaluation_by_item()
    render_full_performance_tables(segment_perf, item_perf, eval_df)

    with st.expander("Ver muestra local de errores enriquecida"):
        eval_sample_table = evaluation_detail_table(enrich_business_names(eval_df))
        render_dataframe(eval_sample_table, max_rows=500, height=420)

elif active_tab == "KPIs":
    segment_perf = load_evaluation_by_segment()
    item_perf = load_evaluation_by_item()
    render_kpi_graphs(segment_perf, item_perf, eval_df)

    render_low_activity_products_table(
        batch_df,
        enrich_fn=enrich_business_names,
        render_dataframe_fn=render_dataframe,
    )

elif active_tab == "Feedback":
    st.header("Feedback")
    c1, c2 = st.columns(2)
    with c1:
        feedback_shop = choose_id_with_dropdown_and_text(batch_df, "shop_id", "shop_name", "Tienda", "feedback_shop")
    with c2:
        items_for_shop = batch_df[batch_df["shop_id"] == feedback_shop] if feedback_shop is not None else batch_df
        feedback_item = choose_id_with_dropdown_and_text(items_for_shop, "item_id", "item_name", "Producto", "feedback_item")
    issue_type = st.selectbox("Tipo de observación", ["Predicción muy alta", "Predicción muy baja", "Producto descontinuado", "Otro"])
    analyst_name = st.text_input("Nombre del analista", value="")
    comment = st.text_area("Comentario")
    if st.button("Guardar feedback en RDS"):
        if DISABLE_RDS_WRITES or insert_business_feedback is None:
            st.warning("RDS está desactivado en local. Prueba feedback real en ECS/Fargate.")
        else:
            try:
                insert_business_feedback(shop_id=int(feedback_shop), item_id=int(feedback_item), issue_type=issue_type, comment=comment, analyst_name=analyst_name or None)
                st.success("Feedback guardado.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"No se pudo guardar feedback: {exc}")

    if read_business_feedback and not DISABLE_RDS_WRITES:
        try:
            render_dataframe(read_business_feedback(limit=100), "Feedback capturado", height=300)
        except Exception:
            pass

    eval_detail = load_evaluation_detail()
    over_fallback, under_fallback = build_extreme_errors(eval_detail, eval_df)

    over = enrich_business_names(load_overestimated_products())
    if over.empty:
        over = over_fallback

    under = enrich_business_names(load_underestimated_products())
    if under.empty:
        under = under_fallback

    render_decision_help(" en Feedback")

    st.subheader("100 productos más sobreestimados")
    st.caption("Predicción mayor que el real. Riesgo principal: sobreinventario.")
    over_table = decision_first_table(enrich_business_names(over))
    render_dataframe(over_table, max_rows=100, height=420)

    st.subheader("100 productos más subestimados")
    st.caption("Predicción menor que el real. Riesgo principal: subabasto.")
    under_table = decision_first_table(enrich_business_names(under))
    render_dataframe(under_table, max_rows=100, height=420)

    st.subheader("Registros sugeridos para revisión")
    review_table = decision_first_table(enrich_business_names(load_review_suggestions()))
    render_dataframe(review_table, max_rows=100, height=360)

    combined_feedback_explain = pd.concat([over_table, under_table, review_table], ignore_index=True)
    render_decision_recommendation_legend(combined_feedback_explain, "Qué significa cada decision_recommendation en Feedback")
    render_review_reason_legend(combined_feedback_explain, "Qué significa cada review_reason en Feedback")

elif active_tab == "Model Registry":
    st.header("Model Registry")
    champion = current_champion()
    metrics = current_metrics()
    global_metrics = metrics.get("global") or metrics.get("champion") or {}
    model_name = champion.get("model_name") or champion.get("model_id") or global_metrics.get("model_name") or "N/D"
    c1, c2, c3 = st.columns(3)
    c1.metric("Modelo", model_name)
    c2.metric("RMSE", metric_text(champion.get("rmse") or global_metrics.get("rmse")))
    c3.metric("MAE", metric_text(champion.get("mae") or global_metrics.get("mae")))
    st.caption("El ranking usa RMSE como primer criterio y MAE como desempate.")

    runs = load_model_runs()
    if not runs.empty:
        runs = runs.copy()
        if "is_champion" not in runs.columns and "model_id" in runs.columns:
            runs["is_champion"] = runs["model_id"].astype(str) == str(champion.get("model_id"))
        if "rmse" in runs.columns:
            runs["rmse"] = pd.to_numeric(runs["rmse"], errors="coerce")
        if "mae" in runs.columns:
            runs["mae"] = pd.to_numeric(runs["mae"], errors="coerce")
        champ = runs[runs.get("is_champion", False).astype(bool)].sort_values(["rmse", "mae"], ascending=[True, True])
        chall = runs[~runs.get("is_champion", False).astype(bool)].sort_values(["rmse", "mae"], ascending=[True, True])
        ordered = pd.concat([champ, chall], ignore_index=True)
        name_col = "model_name" if "model_name" in ordered.columns else "model_id"
        ordered["plot_name"] = ordered.apply(lambda row: f"🏆 {row[name_col]}" if bool(row.get("is_champion")) else str(row[name_col]), axis=1)
        fig = px.bar(ordered, x="rmse", y="plot_name", orientation="h", title="Modelos entrenados ordenados por RMSE")
        fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(ordered["plot_name"].tolist())))
        st.plotly_chart(fig, width="stretch", key="registry_models")
        def _registry_description(row: pd.Series) -> str:
            for col in ["model_id", "model_name", "model_family"]:
                if col in row.index:
                    key = str(row.get(col))
                    if key in MODEL_DESCRIPTIONS:
                        return MODEL_DESCRIPTIONS[key]
                    if key in MODEL_ID_DESCRIPTIONS:
                        return MODEL_ID_DESCRIPTIONS[key]
            name = str(row.get("model_name", row.get("model_id", "modelo"))).lower()
            if "hybrid" in name or "router" in name:
                return MODEL_DESCRIPTIONS["hybrid_router_v1"]
            if "lightgbm" in name or "original" in name or "incumbent" in name:
                return MODEL_DESCRIPTIONS["incumbent_two_stage"]
            if "poisson" in name:
                return MODEL_DESCRIPTIONS["hgb_poisson"]
            if "specialist" in name or "recurrent" in name:
                return MODEL_DESCRIPTIONS["specialist_recurrent"]
            if "rolling" in name:
                return MODEL_DESCRIPTIONS["rolling_mean_3"]
            if "naive" in name:
                return MODEL_DESCRIPTIONS["naive_lag1"]
            if "hurdle" in name or "hgb" in name:
                return MODEL_DESCRIPTIONS["hurdle_hgb"]
            return "Modelo candidato evaluado contra el mismo validation set; revisar métricas y curvas para decidir uso operativo."
        ordered["descripción"] = ordered.apply(_registry_description, axis=1)
        show_cols = [c for c in ["model_id", "model_name", "is_champion", "rmse", "mae", "smape", "wape", "bias", "nonzero_recall", "descripción"] if c in ordered.columns]
        render_dataframe(ordered[show_cols], "Historial de modelos/corridas", height=420)
    else:
        st.info("No hay model_runs.csv ni all_models disponible.")

    with st.expander("Ver champion.json técnico"):
        st.json(champion)
    with st.expander("Debug ModelOps"):
        st.json(modelops_healthcheck())
