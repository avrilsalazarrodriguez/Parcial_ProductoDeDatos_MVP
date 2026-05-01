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
    "hurdle_hgb": "Modelo de dos etapas: estima probabilidad de venta y luego unidades esperadas si hay venta.",
    "incumbent_two_stage": "Modelo original del proyecto con diseño de dos etapas. Sirve como incumbent productivo.",
    "original_lightgbm_two_stage": "LightGBM original de dos etapas: clasificador de venta + regresor de unidades.",
    "hgb_poisson": "Gradient boosting con pérdida Poisson, útil para conteos no negativos.",
    "poisson_hgb": "Gradient boosting con pérdida Poisson, útil para conteos no negativos.",
    "item_mean_fallback": "Baseline por promedio histórico de producto; sirve como fallback simple.",
    "naive_lag1": "Baseline naive que usa la venta del último periodo observado.",
    "naive_last_observed": "Baseline naive que usa la venta del último periodo observado.",
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
    preferred = [
        "shop_id", "shop_name", "item_id", "item_name", "prediction", "model_id", "model_name", "model_scope",
        "item_category_id", "item_category_name", "category_group", "recency_label", "recency",
    ]
    cols = [c for c in preferred if c in df.columns]
    cols.extend([c for c in df.columns if c not in cols and not c.endswith("_label")])
    return df[cols]


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


def current_champion() -> dict[str, Any]:
    return load_champion()


def current_metrics() -> dict[str, Any]:
    return load_model_metrics()



# ---------- Visual analytics helpers ----------

def short_label(value: object, max_len: int = 55) -> str:
    """Shorten long labels so Plotly charts remain readable."""
    if value is None or pd.isna(value):
        return "sin dato"
    text = str(value)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def metric_col(df: pd.DataFrame) -> str | None:
    """Return the preferred error metric column."""
    for col in ["rmse", "mae", "abs_error"]:
        if col in df.columns:
            return col
    return None


def category_display(df: pd.DataFrame) -> pd.Series:
    """Stable category label: ID first, short name after."""
    if "item_category_id" in df.columns:
        base = "cat_" + df["item_category_id"].astype(str)
    elif "segment_key" in df.columns:
        base = df["segment_key"].astype(str)
    else:
        base = pd.Series("sin_categoria", index=df.index)

    if "segment_name" in df.columns:
        return base + " — " + df["segment_name"].map(lambda x: short_label(x, 36))
    if "item_category_name" in df.columns:
        return base + " — " + df["item_category_name"].map(lambda x: short_label(x, 36))
    return base


def product_display(df: pd.DataFrame) -> pd.Series:
    """Stable product label: item_id first, short item name after."""
    if "item_id" in df.columns:
        base = df["item_id"].astype(str)
    else:
        base = pd.Series("sin_item", index=df.index)
    if "item_name" in df.columns:
        return base + " — " + df["item_name"].map(lambda x: short_label(x, 48))
    return base


def local_demand_curve(eval_source: pd.DataFrame) -> pd.DataFrame:
    """Build local demand curve with real, model prediction and naive."""
    if eval_source.empty or not {"y", "prediction"}.issubset(eval_source.columns):
        return pd.DataFrame()
    out = eval_source.copy()
    out["demand_bucket"] = pd.cut(
        out["y"],
        bins=[-0.1, 0, 1, 3, 7, 20],
        include_lowest=True,
    ).astype(str)
    return (
        out.groupby("demand_bucket", as_index=False, observed=False)
        .agg(
            real_mean=("y", "mean"),
            pred_mean=("prediction", "mean"),
            naive_mean=("naive_prediction", "mean") if "naive_prediction" in out.columns else ("prediction", "mean"),
        )
    )


def _standardize_remote_curves(curves: pd.DataFrame) -> pd.DataFrame:
    """Normalize curve files from different pipeline versions to long format."""
    if curves is None or curves.empty or "demand_bucket" not in curves.columns:
        return pd.DataFrame()
    df = curves.copy()

    if {"demand_bucket", "series", "mean_value"}.issubset(df.columns):
        out = df.copy()
        if "model_id" in out.columns:
            out["model_key"] = out["model_id"].astype(str)
        elif "section" in out.columns:
            out["model_key"] = out["section"].astype(str)
        else:
            out["model_key"] = out["series"].astype(str)
        if "model_name" not in out.columns:
            out["model_name"] = out["model_key"]
        return out[["demand_bucket", "model_key", "model_name", "series", "mean_value"]]

    if {"demand_bucket", "real_mean", "pred_mean"}.issubset(df.columns):
        if "model_id" in df.columns:
            df["model_key"] = df["model_id"].astype(str)
        elif "section" in df.columns:
            df["model_key"] = df["section"].astype(str)
        else:
            df["model_key"] = "model"
        if "model_name" not in df.columns:
            df["model_name"] = df["model_key"]

        real = df.groupby("demand_bucket", as_index=False).agg(mean_value=("real_mean", "mean"))
        real["model_key"] = "real"
        real["model_name"] = "Real"
        real["series"] = "Real"

        pred = df[["demand_bucket", "model_key", "model_name", "pred_mean"]].copy()
        pred = pred.rename(columns={"pred_mean": "mean_value"})
        pred["series"] = pred["model_name"]
        return pd.concat(
            [
                real[["demand_bucket", "model_key", "model_name", "series", "mean_value"]],
                pred[["demand_bucket", "model_key", "model_name", "series", "mean_value"]],
            ],
            ignore_index=True,
        )

    return pd.DataFrame()


def render_winner_curve(curves: pd.DataFrame, eval_source: pd.DataFrame, champion: dict[str, Any]) -> None:
    """Plot only Real vs predicted of the winning model."""
    st.subheader("Real vs predicho del modelo ganador")
    normalized = _standardize_remote_curves(curves)
    champion_id = str(champion.get("model_id") or champion.get("model_run_id") or "")
    champion_name = str(champion.get("model_name") or champion.get("display_name") or "")

    plot_df = pd.DataFrame()
    if not normalized.empty:
        real_line = normalized[normalized["series"].astype(str).str.lower().eq("real")].copy()
        pred_candidates = normalized[normalized["series"].astype(str).str.lower().ne("real")].copy()
        pred_line = pd.DataFrame()
        if champion_id:
            pred_line = pred_candidates[pred_candidates["model_key"].astype(str).eq(champion_id)].copy()
        if pred_line.empty and champion_name:
            pred_line = pred_candidates[pred_candidates["model_name"].astype(str).eq(champion_name)].copy()
        if pred_line.empty:
            pred_line = pred_candidates[~pred_candidates["model_key"].astype(str).str.contains("naive", case=False, na=False)].head(5).copy()
        if not real_line.empty and not pred_line.empty:
            pred_line = pred_line.copy()
            pred_line["series"] = "Predicho ganador"
            real_line = real_line.copy()
            real_line["series"] = "Real"
            plot_df = pd.concat([real_line, pred_line], ignore_index=True)

    if plot_df.empty:
        summary = local_demand_curve(eval_source)
        if summary.empty:
            st.info("No hay datos suficientes para graficar la curva del ganador.")
            return
        plot_df = summary.melt(
            id_vars="demand_bucket",
            value_vars=["real_mean", "pred_mean"],
            var_name="series",
            value_name="mean_value",
        )
        plot_df["series"] = plot_df["series"].map({"real_mean": "Real", "pred_mean": "Predicho ganador"})

    fig = px.line(
        plot_df,
        x="demand_bucket",
        y="mean_value",
        color="series",
        markers=True,
        title="Modelo ganador: promedio real vs promedio predicho",
        labels={"demand_bucket": "Rango de demanda real", "mean_value": "Unidades promedio", "series": "Serie"},
        color_discrete_map={"Real": "#7cc7ff", "Predicho ganador": "#ff6b6b"},
    )
    st.plotly_chart(fig, width="stretch", key="eval_winner_curve")
    st.caption("Lectura: si la línea del modelo queda por debajo de Real en alta demanda, hay riesgo de subabasto.")


def render_model_curve_comparison(curves: pd.DataFrame, eval_source: pd.DataFrame, champion: dict[str, Any]) -> None:
    """Compare the same demand curve across models with clear labels."""
    st.subheader("Comparación de curvas por modelo")
    normalized = _standardize_remote_curves(curves)

    if not normalized.empty:
        plot_df = normalized.copy()
        plot_df["series"] = plot_df.apply(
            lambda row: "Real" if str(row["series"]).lower() == "real" else str(row["model_name"]),
            axis=1,
        )
        real = plot_df[plot_df["series"].eq("Real")].drop_duplicates("demand_bucket")
        models = plot_df[~plot_df["series"].eq("Real")].copy()
        if "model_key" in models.columns:
            priority = models["model_key"].astype(str).str.contains("champion|second|naive|hurdle|incumbent", case=False, na=False)
            selected = models[priority].copy()
            if selected.empty:
                selected_names = models["series"].drop_duplicates().head(4).tolist()
                selected = models[models["series"].isin(selected_names)]
            models = selected
        plot_df = pd.concat([real, models], ignore_index=True)
    else:
        summary = local_demand_curve(eval_source)
        if summary.empty:
            st.info("No hay datos suficientes para comparar curvas.")
            return
        plot_df = summary.melt(
            id_vars="demand_bucket",
            value_vars=["real_mean", "pred_mean", "naive_mean"],
            var_name="series",
            value_name="mean_value",
        )
        plot_df["series"] = plot_df["series"].map({"real_mean": "Real", "pred_mean": "Predicho ganador", "naive_mean": "Naive"})

    fig = px.line(
        plot_df,
        x="demand_bucket",
        y="mean_value",
        color="series",
        markers=True,
        title="Comparación: Real vs modelos",
        labels={"demand_bucket": "Rango de demanda real", "mean_value": "Unidades promedio", "series": "Serie"},
    )
    fig.update_layout(legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02))
    st.plotly_chart(fig, width="stretch", key="eval_model_curve_comparison")
    st.caption("Lectura: compara el sesgo por rango de demanda. Revisa si el champion mejora al naive y dónde falla.")


def prepare_segment_table(segment_source: pd.DataFrame, fallback_eval: pd.DataFrame) -> pd.DataFrame:
    source = enrich_business_names(segment_source) if segment_source is not None and not segment_source.empty else pd.DataFrame()
    if source.empty:
        source = enrich_business_names(fallback_eval)
        group_cols = [col for col in ["item_category_id", "item_category_name", "category_group"] if col in source.columns]
        if group_cols and "abs_error" in source.columns:
            source = (
                source.groupby(group_cols, dropna=False, as_index=False)
                .agg(
                    n=("abs_error", "size"),
                    rmse=("abs_error", lambda x: float(np.sqrt(np.mean(np.square(x))))),
                    mae=("abs_error", "mean"),
                    pred_mean=("prediction", "mean"),
                    true_mean=("y", "mean") if "y" in source.columns else ("abs_error", "mean"),
                )
            )
    sort_col = "rmse" if "rmse" in source.columns else metric_col(source)
    if sort_col:
        source = source.sort_values(sort_col, ascending=False)
    return source


def prepare_item_table(item_source: pd.DataFrame, fallback_eval: pd.DataFrame) -> pd.DataFrame:
    source = enrich_business_names(item_source) if item_source is not None and not item_source.empty else pd.DataFrame()
    if source.empty:
        source = enrich_business_names(fallback_eval)
        group_cols = [col for col in ["item_id", "item_name", "item_category_id", "item_category_name"] if col in source.columns]
        if group_cols and "abs_error" in source.columns:
            source = (
                source.groupby(group_cols, dropna=False, as_index=False)
                .agg(
                    n=("abs_error", "size"),
                    rmse=("abs_error", lambda x: float(np.sqrt(np.mean(np.square(x))))),
                    mae=("abs_error", "mean"),
                    pred_mean=("prediction", "mean"),
                    true_mean=("y", "mean") if "y" in source.columns else ("abs_error", "mean"),
                )
            )
    sort_col = "rmse" if "rmse" in source.columns else metric_col(source)
    if sort_col:
        source = source.sort_values(sort_col, ascending=False)
    return source


def render_full_performance_tables(segment_df: pd.DataFrame, item_df: pd.DataFrame) -> None:
    st.subheader("Tablas de performance para navegación")
    st.caption("Usa el buscador nativo de Streamlit o los filtros para revisar cualquier segmento/categoría o producto.")
    seg = prepare_segment_table(segment_df, eval_df)
    item = prepare_item_table(item_df, eval_df)
    left, right = st.columns(2)
    with left:
        st.markdown("**Todos los segmentos/categorías**")
        if not seg.empty:
            seg_filter = st.text_input("Filtrar segmento/categoría", value="", key="eval_segment_filter")
            show = seg.copy()
            if seg_filter.strip():
                mask = pd.Series(False, index=show.index)
                for col in ["segment_key", "segment_name", "item_category_id", "item_category_name", "category_group"]:
                    if col in show.columns:
                        mask = mask | show[col].astype(str).str.contains(seg_filter, case=False, na=False)
                show = show[mask]
            render_dataframe(show, height=430)
        else:
            st.info("No hay tabla de segmentos/categorías disponible.")
    with right:
        st.markdown("**Todos los productos**")
        if not item.empty:
            item_filter = st.text_input("Filtrar producto", value="", key="eval_item_filter")
            show = item.copy()
            if item_filter.strip():
                mask = pd.Series(False, index=show.index)
                for col in ["item_id", "item_name", "item_category_id", "item_category_name"]:
                    if col in show.columns:
                        mask = mask | show[col].astype(str).str.contains(item_filter, case=False, na=False)
                show = show[mask]
            render_dataframe(show, height=430)
        else:
            st.info("No hay tabla de productos disponible.")


def render_kpi_graphs(segment_df: pd.DataFrame, item_df: pd.DataFrame) -> None:
    st.header("KPIs")
    seg = prepare_segment_table(segment_df, eval_df)
    item = prepare_item_table(item_df, eval_df)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Categorías con mayor RMSE")
        if seg.empty:
            st.info("No hay datos por categoría.")
        else:
            mcol = "rmse" if "rmse" in seg.columns else metric_col(seg)
            plot = seg.sort_values(mcol, ascending=False).head(15).copy()
            plot["category_display"] = category_display(plot)
            fig = px.bar(
                plot.sort_values(mcol, ascending=True),
                x=mcol,
                y="category_display",
                orientation="h",
                title="Top categorías por RMSE",
                labels={mcol: "RMSE", "category_display": "Categoría"},
            )
            st.plotly_chart(fig, width="stretch", key="kpi_category_rmse_clean")
            render_dataframe(seg, max_rows=100, height=320)
    with col2:
        st.subheader("Productos con mayor RMSE")
        if item.empty:
            st.info("No hay datos por producto.")
        else:
            mcol = "rmse" if "rmse" in item.columns else metric_col(item)
            plot = item.sort_values(mcol, ascending=False).head(15).copy()
            plot["product_display"] = product_display(plot)
            fig = px.bar(
                plot.sort_values(mcol, ascending=True),
                x=mcol,
                y="product_display",
                orientation="h",
                title="Top productos por RMSE",
                labels={mcol: "RMSE", "product_display": "Producto"},
            )
            st.plotly_chart(fig, width="stretch", key="kpi_product_rmse_clean")
            render_dataframe(item, max_rows=100, height=320)


def build_extreme_errors(eval_detail_df: pd.DataFrame, fallback_eval: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return over, under

# ---------- App layout ----------

st.set_page_config(page_title="Producto de Datos — Pronóstico de Ventas", page_icon="📦", layout="wide")
st.title("📦 Producto de Datos — Pronóstico de Ventas")
st.caption("MVP Streamlit para consultar pronósticos mensuales, evaluar modelos, generar archivos CFO y capturar feedback operativo.")

with st.sidebar:
    st.subheader("Estado de conexiones")
    st.write("ModelOps S3:", "activo" if USE_MODELOPS_S3 else "local")
    st.write("RDS writes:", "desactivado" if DISABLE_RDS_WRITES else "activo")
    st.write("Root:", os.getenv("MODELOPS_LOCAL_ROOT", "modelops_outputs"))

valid_df = load_valid_data()
test_features = load_test_features()
test_pairs = load_test_pairs()
model_payload = load_model()
batch_df = load_batch_forecast()
eval_df = build_local_eval_sample(valid_df, model_payload, batch_df)

TAB_NAMES = ["Resumen", "Inferencia individual", "Batch CFO", "Evaluación", "KPIs", "Feedback", "Model Registry"]
tab_summary, tab_single, tab_cfo, tab_eval, tab_kpis, tab_feedback, tab_registry = st.tabs(TAB_NAMES)

with tab_summary:
    st.header("Resumen ejecutivo")
    st.write("Este tablero resume el flujo operativo del producto de datos: consulta individual, batch CFO, evaluación, KPIs, feedback y registry.")
    help_cols = st.columns(3)
    help_cols[0].info("**Resumen**: volumen general, distribución de pronósticos y principales tiendas/categorías.")
    help_cols[1].info("**Inferencia / Batch CFO**: consulta un par tienda-producto o genera archivos descargables para finanzas.")
    help_cols[2].info("**Evaluación / KPIs / Feedback**: revisa errores, detecta productos problemáticos y captura observaciones.")
    st.caption("El objetivo es que negocio pueda entender dónde venderemos más, dónde falla el modelo y qué productos requieren revisión.")

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
    with col_b:
        dist = batch_df[["prediction"]].copy()
        dist["prediction_log1p"] = np.log1p(pd.to_numeric(dist["prediction"], errors="coerce").fillna(0).clip(lower=0))
        scale = st.radio("Escala de distribución", ["Original", "Log1p"], horizontal=True, key="summary_dist_scale")
        x_col = "prediction" if scale == "Original" else "prediction_log1p"
        fig = px.histogram(dist.sample(min(25_000, len(dist)), random_state=42), x=x_col, nbins=60, title="Distribución de pronósticos")
        st.plotly_chart(fig, width="stretch", key="summary_distribution")

    col_c, col_d = st.columns(2)
    with col_c:
        cat_summary = load_forecast_summary_by_category()
        if not cat_summary.empty:
            name_col = "item_category_name" if "item_category_name" in cat_summary.columns else "item_category_id"
            val_col = "total_prediction" if "total_prediction" in cat_summary.columns else "mean_prediction"
            fig = px.bar(cat_summary.sort_values(val_col, ascending=False).head(12).sort_values(val_col), x=val_col, y=name_col, orientation="h", title="Categorías con mayor pronóstico")
            st.plotly_chart(fig, width="stretch", key="summary_top_categories")
    with col_d:
        by_scope = batch_df.groupby("model_scope", dropna=False, as_index=False).agg(n=("prediction", "size"), total_prediction=("prediction", "sum")) if "model_scope" in batch_df.columns else pd.DataFrame()
        if not by_scope.empty:
            fig = px.pie(by_scope, values="n", names="model_scope", title="Cobertura por model_scope")
            st.plotly_chart(fig, width="stretch", key="summary_scope_pie")

with tab_single:
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

with tab_cfo:
    st.header("Batch CFO")
    with st.expander("¿Qué significa model_scope?"):
        st.markdown("""
        `model_scope` indica qué lógica generó el pronóstico: `global` usa el modelo general,
        `segment:*` usa un modelo especializado, `original_two_stage` usa el diseño original de dos etapas,
        y `fallback` usa una regla conservadora cuando no hay suficiente historial.
        """)

    scope = st.radio("Alcance", ["Todos los productos de una tienda", "Segmento / categoría", "Catálogo completo"], horizontal=True)
    if scope == "Todos los productos de una tienda":
        selected_shop = choose_id_with_dropdown_and_text(batch_df, "shop_id", "shop_name", "Tienda", "cfo_shop")
        filtered_batch = batch_df[batch_df["shop_id"] == selected_shop].copy() if selected_shop is not None else batch_df.copy()
    elif scope == "Segmento / categoría":
        category_options = sorted(batch_df["item_category_id"].dropna().astype(int).astype(str).unique()) if "item_category_id" in batch_df.columns else []
        selected_category = st.selectbox("Selecciona item_category_id", category_options)
        filtered_batch = batch_df[batch_df["item_category_id"].astype("Int64").astype(str) == str(selected_category)].copy() if category_options else batch_df.copy()
    else:
        selected_shop = None
        filtered_batch = batch_df.copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Registros", f"{len(filtered_batch):,}")
    c2.metric("Pronóstico total", f"{filtered_batch['prediction'].sum():,.1f}")
    c3.metric("Promedio", f"{filtered_batch['prediction'].mean():.3f}")
    render_dataframe(visible_forecast_columns(filtered_batch), "Vista previa del archivo CFO", max_rows=1000, height=420)

    csv = visible_forecast_columns(filtered_batch).to_csv(index=False).encode("utf-8")
    if st.button("Generar archivo CFO y guardar en S3"):
        if upload_batch_dataframe_to_s3 is None:
            st.warning("backend.storage.upload_batch_dataframe_to_s3 no está disponible.")
        else:
            try:
                s3_uri = upload_batch_dataframe_to_s3(filtered_batch, scope=scope, shop_id=selected_shop if scope == "Todos los productos de una tienda" else None)
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

with tab_eval:
    st.header("Evaluación")
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE modelo local", f"{rmse(eval_df['y'], eval_df['prediction']):.4f}")
    c2.metric("RMSE naive", f"{rmse(eval_df['y'], eval_df['naive_prediction']):.4f}")
    c3.metric("MAE modelo local", f"{mae(eval_df['y'], eval_df['prediction']):.4f}")

    curves = load_evaluation_curves_by_model()
    champion = current_champion()
    render_winner_curve(curves, eval_df, champion)
    render_model_curve_comparison(curves, eval_df, champion)

    st.divider()
    segment_perf = load_evaluation_by_segment()
    item_perf = load_evaluation_by_item()
    render_full_performance_tables(segment_perf, item_perf)

    with st.expander("Ver muestra local de errores enriquecida"):
        render_dataframe(visible_forecast_columns(enrich_business_names(eval_df)), max_rows=500, height=420)

with tab_kpis:
    segment_perf = load_evaluation_by_segment()
    item_perf = load_evaluation_by_item()
    render_kpi_graphs(segment_perf, item_perf)

with tab_feedback:
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

    st.subheader("100 productos más sobreestimados")
    st.caption("Predicción mayor que el real. Riesgo principal: sobreinventario.")
    render_dataframe(visible_forecast_columns(enrich_business_names(over)), max_rows=100, height=420)

    st.subheader("100 productos más subestimados")
    st.caption("Predicción menor que el real. Riesgo principal: subabasto.")
    render_dataframe(visible_forecast_columns(enrich_business_names(under)), max_rows=100, height=420)

    st.subheader("Registros sugeridos para revisión")
    render_dataframe(visible_forecast_columns(enrich_business_names(load_review_suggestions())), max_rows=100, height=360)

with tab_registry:
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
        if "model_id" in ordered.columns:
            ordered["descripción"] = ordered["model_id"].map(lambda x: MODEL_DESCRIPTIONS.get(str(x), "Modelo candidato evaluado contra el mismo validation set."))
        show_cols = [c for c in ["model_id", "model_name", "is_champion", "rmse", "mae", "smape", "wape", "bias", "nonzero_recall", "descripción"] if c in ordered.columns]
        render_dataframe(ordered[show_cols], "Historial de modelos/corridas", height=420)
    else:
        st.info("No hay model_runs.csv ni all_models disponible.")

    with st.expander("Ver champion.json técnico"):
        st.json(champion)
    with st.expander("Debug ModelOps"):
        st.json(modelops_healthcheck())
