from __future__ import annotations

"""Uploaded CSV batch inference component for the Streamlit app.

This module is isolated so the main app design remains unchanged. It supports
three input styles:

1. Full feature CSV: the uploaded file already has all model feature columns.
2. Test-like CSV: the uploaded file only has identifiers such as shop_id/item_id,
   and features are completed from data/prep/test_features.parquet.
3. Catalog mode: infer over the full prepared test catalog, regardless of the
   number of rows in the uploaded file.

The hybrid-routing policy is applied after feature completion, so uploaded batch
predictions use the same business criteria as the ModelOps outputs.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
import os

import numpy as np
import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class UploadedBatchResult:
    dataframe: pd.DataFrame
    missing_features: list[str]
    used_zero_fill: bool
    used_hybrid_router: bool
    specialist_source: str
    feature_source: str
    unmatched_uploaded_rows: int = 0


def _get_bundle(model_payload: dict[str, Any] | Any) -> Any:
    if isinstance(model_payload, dict):
        return model_payload.get("bundle", model_payload)
    return model_payload


def required_feature_columns(model_payload: dict[str, Any] | Any) -> list[str]:
    bundle = _get_bundle(model_payload)
    if isinstance(bundle, dict) and "feature_cols" in bundle:
        return list(bundle["feature_cols"])
    if hasattr(bundle, "feature_names_in_"):
        return [str(col) for col in bundle.feature_names_in_]
    return []


def _clip_prediction(values: Any) -> np.ndarray:
    return np.clip(np.nan_to_num(np.asarray(values, dtype="float64"), nan=0.0, posinf=20.0, neginf=0.0), 0, 20)


def predict_uploaded_batch(model_payload: dict[str, Any] | Any, features: pd.DataFrame) -> np.ndarray:
    """Predict with the original two-stage bundle or a normal estimator."""
    bundle = _get_bundle(model_payload)

    if isinstance(bundle, dict) and {"clf", "reg", "feature_cols"}.issubset(bundle.keys()):
        feature_cols = list(bundle["feature_cols"])
        x_test = features[feature_cols]
        prob = bundle["clf"].predict_proba(x_test)[:, 1].astype("float64")
        mu = bundle["reg"].predict(x_test).astype("float64")
        return _clip_prediction(prob * mu)

    if hasattr(bundle, "predict"):
        feature_cols = required_feature_columns(model_payload)
        x_test = features[feature_cols] if feature_cols else features
        pred = bundle.predict(x_test)
        return _clip_prediction(pred)

    raise TypeError("No pude identificar cómo predecir con model_payload. Se esperaba bundle clf/reg o estimador con predict().")


def _coerce_feature_frame(df: pd.DataFrame, feature_cols: list[str], fill_missing_with_zero: bool) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    missing = [col for col in feature_cols if col not in out.columns]
    if missing and fill_missing_with_zero:
        for col in missing:
            out[col] = 0
    for col in feature_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    return out, missing


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _num(df: pd.DataFrame, col: str | None, default: float = 0.0) -> pd.Series:
    if col and col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


def _lag_col(df: pd.DataFrame, lag: int) -> str | None:
    return _first_existing(df, [f"cnt_lag_{lag}", f"lag_{lag}", f"target_lag_{lag}", f"item_cnt_month_lag_{lag}", f"y_lag_{lag}"])


def _safe_mean_from_lags(df: pd.DataFrame, lags: list[int]) -> pd.Series:
    values = [_num(df, _lag_col(df, lag), 0.0) for lag in lags]
    if not values:
        return pd.Series(0.0, index=df.index, dtype="float64")
    return pd.concat(values, axis=1).mean(axis=1)


def _safe_sum_from_lags(df: pd.DataFrame, lags: list[int]) -> pd.Series:
    values = [_num(df, _lag_col(df, lag), 0.0) for lag in lags]
    if not values:
        return pd.Series(0.0, index=df.index, dtype="float64")
    return pd.concat(values, axis=1).sum(axis=1)


def _safe_nonzero_rate_from_lags(df: pd.DataFrame, lags: list[int]) -> pd.Series:
    values = [(_num(df, _lag_col(df, lag), 0.0) > 0).astype(float) for lag in lags]
    if not values:
        return pd.Series(0.0, index=df.index, dtype="float64")
    return pd.concat(values, axis=1).mean(axis=1)


def _candidate_reference_paths() -> tuple[list[Path], list[Path]]:
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    features_env = os.getenv("TEST_FEATURES_PATH") or os.getenv("BATCH_TEST_FEATURES_PATH")
    pairs_env = os.getenv("TEST_PAIRS_PATH") or os.getenv("BATCH_TEST_PAIRS_PATH")

    feature_paths: list[Path] = []
    pair_paths: list[Path] = []
    if features_env:
        feature_paths.append(Path(features_env))
    if pairs_env:
        pair_paths.append(Path(pairs_env))

    feature_paths.extend([
        data_dir / "prep" / "test_features.parquet",
        data_dir / "test_features.parquet",
        Path("data/prep/test_features.parquet"),
        Path("data/test_features.parquet"),
    ])
    pair_paths.extend([
        data_dir / "prep" / "test_pairs.parquet",
        data_dir / "test_pairs.parquet",
        Path("data/prep/test_pairs.parquet"),
        Path("data/test_pairs.parquet"),
    ])
    return feature_paths, pair_paths


@st.cache_data(ttl=900, show_spinner=False)
def _load_reference_dataset_cached(feature_path: str, pair_path: str | None) -> pd.DataFrame:
    features = pd.read_parquet(feature_path).reset_index(drop=True)
    if pair_path:
        pairs = pd.read_parquet(pair_path).reset_index(drop=True)
        # Keep identifiers/metadata from pairs without duplicating feature columns.
        extra_cols = [col for col in pairs.columns if col not in features.columns]
        reference = pd.concat([pairs[extra_cols], features], axis=1)
        for id_col in ["shop_id", "item_id", "item_category_id"]:
            if id_col not in reference.columns and id_col in pairs.columns:
                reference[id_col] = pairs[id_col]
    else:
        reference = features
    return reference


def load_reference_dataset() -> tuple[pd.DataFrame, str]:
    """Load prepared dataset features used for production/test inference."""
    feature_paths, pair_paths = _candidate_reference_paths()
    feature_path = next((path for path in feature_paths if path.exists()), None)
    if feature_path is None:
        return pd.DataFrame(), "no_reference_features_found"
    pair_path = next((path for path in pair_paths if path.exists()), None)
    reference = _load_reference_dataset_cached(str(feature_path), str(pair_path) if pair_path else None)
    source = f"features:{feature_path}"
    if pair_path:
        source += f" | pairs:{pair_path}"
    return reference, source


def _as_unique_values(series: pd.Series) -> list[Any]:
    return series.dropna().drop_duplicates().tolist()


def _filter_reference_by_uploaded(reference: pd.DataFrame, uploaded_df: pd.DataFrame) -> tuple[pd.DataFrame, int, str]:
    """Interpret uploaded CSV as filters or exact pairs over the reference feature matrix."""
    if reference.empty:
        return pd.DataFrame(), len(uploaded_df), "reference_empty"

    ref = reference.copy()
    uploaded = uploaded_df.copy()

    has_shop = "shop_id" in uploaded.columns and "shop_id" in ref.columns
    has_item = "item_id" in uploaded.columns and "item_id" in ref.columns

    # Exact product-store pairs. Preserve uploaded ID/order and duplicates so a
    # validation/test-style file (ID, shop_id, item_id) returns one prediction
    # per uploaded row.
    #
    # Important: for production-like unseen pairs, we do NOT drop unmatched rows.
    # We keep them and mark them as cold-start. Feature columns for those rows are
    # filled with neutral historical defaults downstream, so the hybrid router can
    # route them as inactive/no-history instead of requiring leakage from a future
    # feature store.
    if has_shop and has_item:
        keys = ["shop_id", "item_id"]
        uploaded_keys = uploaded.copy()
        uploaded_keys["__upload_order"] = np.arange(len(uploaded_keys))
        for key in keys:
            uploaded_keys[key] = pd.to_numeric(uploaded_keys[key], errors="coerce").astype("Int64")
            ref[key] = pd.to_numeric(ref[key], errors="coerce").astype("Int64")
        merged = uploaded_keys.merge(ref, on=keys, how="left", indicator=True, suffixes=("", "_ref"))
        merged["__cold_start_row"] = merged["_merge"].eq("left_only")
        unmatched = int(merged["__cold_start_row"].sum())
        merged = merged.drop(columns=["_merge"], errors="ignore").sort_values("__upload_order")
        merged = merged.drop(columns=["__upload_order"], errors="ignore")
        source = "exact_shop_item_pairs_from_reference_preserving_uploaded_order"
        if unmatched:
            source += f" | cold_start_unmatched_pairs={unmatched}"
        return merged.reset_index(drop=True), unmatched, source

    mask = pd.Series(True, index=ref.index)
    filters = []

    if has_shop:
        values = [int(v) for v in _as_unique_values(pd.to_numeric(uploaded["shop_id"], errors="coerce")) if pd.notna(v)]
        if values:
            mask &= pd.to_numeric(ref["shop_id"], errors="coerce").isin(values)
            filters.append(f"shop_id in {values[:10]}")

    if has_item:
        values = [int(v) for v in _as_unique_values(pd.to_numeric(uploaded["item_id"], errors="coerce")) if pd.notna(v)]
        if values:
            mask &= pd.to_numeric(ref["item_id"], errors="coerce").isin(values)
            filters.append(f"item_id in {values[:10]}")

    for col in ["item_category_id", "category_id", "segment_id"]:
        if col in uploaded.columns and "item_category_id" in ref.columns:
            values = [int(v) for v in _as_unique_values(pd.to_numeric(uploaded[col], errors="coerce")) if pd.notna(v)]
            if values:
                mask &= pd.to_numeric(ref["item_category_id"], errors="coerce").isin(values)
                filters.append(f"item_category_id in {values[:10]}")
                break

    for col in ["item_category_name", "category_group", "shop_name", "item_name"]:
        if col in uploaded.columns and col in ref.columns:
            values = [str(v) for v in _as_unique_values(uploaded[col].astype(str)) if v and v.lower() != "nan"]
            if values:
                mask &= ref[col].astype(str).isin(values)
                filters.append(f"{col} in {values[:5]}")

    if not filters:
        return ref.reset_index(drop=True), 0, "no_filters_detected_using_full_reference_catalog"

    filtered = ref[mask].copy().reset_index(drop=True)
    return filtered, 0, "filters_expanded_from_reference: " + "; ".join(filters)


def complete_uploaded_features(
    uploaded_df: pd.DataFrame,
    feature_cols: list[str],
    mode: str,
    fill_missing_with_zero: bool,
) -> tuple[pd.DataFrame, list[str], bool, str, int]:
    """Return a feature-complete dataframe for inference.

    mode:
    - uploaded_as_is: use uploaded file as feature matrix.
    - complete_from_reference: complete/expand using prepared test_features.
    - full_reference_catalog: ignore uploaded rows and use the full prepared catalog.
    """
    if mode == "uploaded_as_is":
        prepared, missing = _coerce_feature_frame(uploaded_df, feature_cols, fill_missing_with_zero)
        return prepared, missing, fill_missing_with_zero and bool(missing), "uploaded_features_direct", 0

    reference, reference_source = load_reference_dataset()
    if reference.empty:
        prepared, missing = _coerce_feature_frame(uploaded_df, feature_cols, fill_missing_with_zero)
        return prepared, missing, fill_missing_with_zero and bool(missing), reference_source, 0

    if mode == "full_reference_catalog":
        prepared = reference.copy().reset_index(drop=True)
        unmatched = 0
        feature_source = f"full_reference_catalog | {reference_source}"
    else:
        prepared, unmatched, filter_source = _filter_reference_by_uploaded(reference, uploaded_df)
        feature_source = f"{filter_source} | {reference_source}"
        if prepared.empty:
            # Explicitly fail back to uploaded to show missing columns rather than silently predicting nothing.
            prepared = uploaded_df.copy()
            feature_source = f"reference_match_empty_fallback_uploaded | {reference_source}"

    # For true unseen product-store pairs, reference features are not available.
    # Fill the model feature columns with neutral historical defaults only for
    # those cold-start rows. This is not leakage: it encodes "no known history".
    if "__cold_start_row" in prepared.columns and prepared["__cold_start_row"].fillna(False).any():
        cold_mask = prepared["__cold_start_row"].fillna(False).astype(bool)
        for col in feature_cols:
            if col not in prepared.columns:
                prepared[col] = 0.0
            prepared.loc[cold_mask, col] = pd.to_numeric(prepared.loc[cold_mask, col], errors="coerce").fillna(0.0)
        for col in ["recency", "price_missing_last", "never_sold_before"]:
            if col in prepared.columns:
                if col == "recency":
                    prepared.loc[cold_mask, col] = pd.to_numeric(prepared.loc[cold_mask, col], errors="coerce").fillna(99.0)
                else:
                    prepared.loc[cold_mask, col] = pd.to_numeric(prepared.loc[cold_mask, col], errors="coerce").fillna(1.0)
        feature_source += " | cold_start_rows_filled_with_no_history_defaults"

    prepared, missing = _coerce_feature_frame(prepared, feature_cols, fill_missing_with_zero or ("__cold_start_row" in prepared.columns))
    return prepared, missing, (fill_missing_with_zero and bool(missing)), feature_source, unmatched


def _load_recurrent_specialist(feature_cols: list[str]) -> tuple[Any | None, str]:
    """Load optional recurrent specialist created by ModelOps, if present."""
    candidates = []
    root_env = os.getenv("MODELOPS_LOCAL_ROOT") or os.getenv("MODEL_OPS_LOCAL_ROOT")
    if root_env:
        root = Path(root_env)
        candidates.extend([root / "latest/model/recurrent_specialist.joblib", root / "model/recurrent_specialist.joblib"])
    candidates.extend([
        Path("modelops_outputs_hybrid/latest/model/recurrent_specialist.joblib"),
        Path("modelops_outputs/latest/model/recurrent_specialist.joblib"),
    ])
    for path in candidates:
        if path.exists():
            try:
                import joblib

                return joblib.load(path), f"specialist_joblib:{path}"
            except Exception:
                continue
    return None, "rolling_mean_proxy:no_specialist_joblib"


def _predict_specialist_if_available(specialist_model: Any | None, features: pd.DataFrame, feature_cols: list[str]) -> np.ndarray | None:
    if specialist_model is None:
        return None
    try:
        pred = specialist_model.predict(features[feature_cols])
        pred = np.expm1(np.asarray(pred, dtype="float64"))
        return _clip_prediction(pred)
    except Exception:
        return None


def apply_uploaded_hybrid_policy(
    prepared_features: pd.DataFrame,
    pred_champion: np.ndarray,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, str]:
    """Apply the same high-level criteria used by the hybrid router."""
    out = prepared_features.copy()

    lag1 = _num(out, _lag_col(out, 1), 0.0)
    rolling_mean_3 = _num(out, _first_existing(out, ["router_rolling_mean_3", "rolling_mean_3", "mean_3"]), np.nan)
    rolling_mean_3 = rolling_mean_3.where(rolling_mean_3.notna(), _safe_mean_from_lags(out, [1, 2, 3]))

    rolling_sum_6 = _num(out, _first_existing(out, ["router_rolling_sum_6", "rolling_sum_6", "sum_6"]), np.nan)
    rolling_sum_6 = rolling_sum_6.where(rolling_sum_6.notna(), _safe_sum_from_lags(out, [1, 2, 3, 4, 5, 6]))

    nonzero_rate_6 = _num(out, _first_existing(out, ["router_nonzero_rate_6", "nonzero_rate_6", "nz_rate_6"]), np.nan)
    nonzero_rate_6 = nonzero_rate_6.where(nonzero_rate_6.notna(), _safe_nonzero_rate_from_lags(out, [1, 2, 3, 4, 5, 6]))

    recency = _num(out, "recency", np.nan)
    inferred_recency = pd.Series(np.where(rolling_sum_6 > 0, 1.0, 99.0), index=out.index, dtype="float64")
    recency = recency.where(recency.notna(), inferred_recency)

    specialist_model, specialist_source = _load_recurrent_specialist(feature_cols)
    pred_specialist = _predict_specialist_if_available(specialist_model, out, feature_cols)
    if pred_specialist is None:
        pred_specialist = _clip_prediction(rolling_mean_3)

    pred_naive = _clip_prediction(lag1)
    pred_rolling = _clip_prediction(rolling_mean_3)
    pred_zero = np.zeros(len(out), dtype="float64")
    pred_champion = _clip_prediction(pred_champion)

    inactive_mask = (recency >= 99) | (rolling_sum_6 <= 0)
    recent_mask = (~inactive_mask) & (lag1 >= 1)
    recurrent_mask = (~inactive_mask) & (~recent_mask) & ((rolling_mean_3 >= 1) | (nonzero_rate_6 >= 0.40))

    prediction = pd.Series(pred_champion, index=out.index, dtype="float64")
    model_scope = pd.Series("challenger:hurdle_hgb", index=out.index, dtype="object")
    routing_reason = pd.Series("default_sparse_low_demand", index=out.index, dtype="object")
    decision = pd.Series("Modelo ganador Hurdle HGB", index=out.index, dtype="object")

    prediction.loc[inactive_mask] = pred_zero[inactive_mask]
    model_scope.loc[inactive_mask] = "inactive:no_recent_sales"
    routing_reason.loc[inactive_mask] = "recency>=99_or_rolling_sum_6==0"
    decision.loc[inactive_mask] = "Regla cero / producto inactivo"

    prediction.loc[recent_mask] = pred_naive[recent_mask]
    model_scope.loc[recent_mask] = "baseline:naive_recent_demand"
    routing_reason.loc[recent_mask] = "cnt_lag_1>=1"
    decision.loc[recent_mask] = "Baseline naive por demanda reciente"

    prediction.loc[recurrent_mask] = pred_specialist[recurrent_mask]
    model_scope.loc[recurrent_mask] = "specialist:recurrent_demand"
    routing_reason.loc[recurrent_mask] = "rolling_mean_or_nonzero_rate_signal"
    if specialist_source.startswith("specialist_joblib"):
        decision.loc[recurrent_mask] = "Modelo especialista de demanda recurrente"
    else:
        decision.loc[recurrent_mask] = "Modelo especialista de demanda recurrente (proxy rolling_mean_3)"

    routed = pd.DataFrame(index=out.index)
    routed["prediction"] = prediction.to_numpy()
    routed["model_id"] = "hybrid_router_uploaded_batch"
    routed["model_name"] = "Inferencia batch cargada con política híbrida"
    routed["model_scope"] = model_scope.to_numpy()
    routed["routing_reason"] = routing_reason.to_numpy()
    routed["decision_recommendation"] = decision.to_numpy()
    routed["pred_champion"] = pred_champion
    routed["pred_naive_lag1"] = pred_naive
    routed["pred_rolling_mean_3"] = pred_rolling
    routed["pred_specialist_recurrent"] = pred_specialist
    routed["recency_router"] = recency.to_numpy()
    routed["rolling_sum_6_router"] = rolling_sum_6.to_numpy()
    routed["rolling_mean_3_router"] = rolling_mean_3.to_numpy()
    routed["nonzero_rate_6_router"] = nonzero_rate_6.to_numpy()
    return routed, specialist_source


def run_uploaded_batch(
    uploaded_df: pd.DataFrame,
    model_payload: dict[str, Any] | Any,
    fill_missing_with_zero: bool = False,
    use_hybrid_policy: bool = True,
    feature_completion_mode: str = "complete_from_reference",
) -> UploadedBatchResult:
    feature_cols = required_feature_columns(model_payload)
    if not feature_cols:
        raise ValueError("No encontré feature_cols en el modelo. No puedo validar el CSV cargado.")

    prepared_df, missing, used_zero_fill, feature_source, unmatched = complete_uploaded_features(
        uploaded_df,
        feature_cols,
        mode=feature_completion_mode,
        fill_missing_with_zero=fill_missing_with_zero,
    )
    if missing and not fill_missing_with_zero:
        return UploadedBatchResult(
            dataframe=pd.DataFrame(),
            missing_features=missing,
            used_zero_fill=False,
            used_hybrid_router=False,
            specialist_source="not_run",
            feature_source=feature_source,
            unmatched_uploaded_rows=unmatched,
        )

    pred_champion = predict_uploaded_batch(model_payload, prepared_df)
    result = prepared_df.copy()

    if use_hybrid_policy:
        routed, specialist_source = apply_uploaded_hybrid_policy(prepared_df, pred_champion, feature_cols)
        for col in routed.columns:
            result[col] = routed[col].to_numpy()
        return UploadedBatchResult(
            dataframe=result,
            missing_features=missing,
            used_zero_fill=used_zero_fill,
            used_hybrid_router=True,
            specialist_source=specialist_source,
            feature_source=feature_source,
            unmatched_uploaded_rows=unmatched,
        )

    result["prediction"] = pred_champion
    result["model_id"] = "uploaded_batch_direct_model"
    result["model_name"] = "Inferencia batch directa por archivo cargado"
    result["model_scope"] = "uploaded_file:direct_inference"
    result["routing_reason"] = "uploaded_csv_direct_inference"
    result["decision_recommendation"] = "Inferencia directa del modelo"
    return UploadedBatchResult(
        dataframe=result,
        missing_features=missing,
        used_zero_fill=used_zero_fill,
        used_hybrid_router=False,
        specialist_source="direct",
        feature_source=feature_source,
        unmatched_uploaded_rows=unmatched,
    )


def _safe_enrich(df: pd.DataFrame, enrich_fn: Callable[[pd.DataFrame], pd.DataFrame] | None) -> pd.DataFrame:
    if enrich_fn is None or df.empty:
        return df
    try:
        return enrich_fn(df)
    except Exception:
        return df


def _safe_visible(df: pd.DataFrame, visible_fn: Callable[[pd.DataFrame], pd.DataFrame] | None) -> pd.DataFrame:
    if visible_fn is None or df.empty:
        return df
    try:
        return visible_fn(df)
    except Exception:
        return df


def _resolve_s3_bucket() -> tuple[str | None, str]:
    """Resolve the S3 bucket for uploaded-batch predictions."""
    for env_name in ["MODEL_BUCKET", "MODELOPS_BUCKET", "MODEL_OPS_BUCKET", "S3_BUCKET", "BUCKET_NAME"]:
        value = os.getenv(env_name)
        if value:
            return value, f"env:{env_name}"

    try:
        import backend.modelops_s3 as modelops_s3  # type: ignore

        for attr in ["get_config", "get_modelops_config", "_config", "config", "CONFIG"]:
            candidate = getattr(modelops_s3, attr, None)
            cfg = candidate() if callable(candidate) else candidate
            if cfg is not None and getattr(cfg, "bucket", None):
                return getattr(cfg, "bucket"), f"backend.modelops_s3.{attr}.bucket"
    except Exception:
        pass

    return None, "not_found"


def _s3_client():
    import boto3

    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
    return boto3.client("s3", region_name=region)


def _uploaded_predictions_prefix() -> tuple[str | None, str, str]:
    bucket, source = _resolve_s3_bucket()
    prefix = os.getenv("UPLOADED_BATCH_S3_PREFIX", "app/batch_uploads/predictions").strip("/")
    return bucket, source, prefix


def _uploaded_history_key() -> str:
    prefix = os.getenv("UPLOADED_BATCH_HISTORY_PREFIX", "app/batch_uploads/history").strip("/")
    return f"{prefix}/uploaded_predictions_history.csv"


def _uploaded_s3_template() -> tuple[str | None, str]:
    bucket, _source, prefix = _uploaded_predictions_prefix()
    if not bucket:
        return None, prefix
    return f"s3://{bucket}/{prefix}/uploaded_batch_predictions_<timestamp>_<archivo>.csv", prefix


def _read_uploaded_prediction_history() -> pd.DataFrame:
    bucket, _source, _prefix = _uploaded_predictions_prefix()
    if not bucket:
        return pd.DataFrame()
    key = _uploaded_history_key()
    try:
        obj = _s3_client().get_object(Bucket=bucket, Key=key)
        return pd.read_csv(obj["Body"])
    except Exception:
        return pd.DataFrame()


def _write_uploaded_prediction_history(history: pd.DataFrame) -> str:
    bucket, _source, _prefix = _uploaded_predictions_prefix()
    if not bucket:
        raise RuntimeError("No encontré bucket S3 para guardar el historial de predicciones cargadas.")
    key = _uploaded_history_key()
    body = history.to_csv(index=False).encode("utf-8")
    _s3_client().put_object(Bucket=bucket, Key=key, Body=body, ContentType="text/csv")
    return f"s3://{bucket}/{key}"


def _append_uploaded_prediction_history(record: dict[str, Any]) -> str:
    return _append_uploaded_prediction_history_many([record])


def _append_uploaded_prediction_history_many(records: list[dict[str, Any]]) -> str:
    history = _read_uploaded_prediction_history()
    rows = pd.DataFrame(records)
    if history.empty:
        updated = rows
    else:
        updated = pd.concat([history, rows], ignore_index=True)
    if "created_at" in updated.columns:
        updated = updated.sort_values("created_at", ascending=False).head(2000)
    return _write_uploaded_prediction_history(updated)


def _safe_file_stem(name: str | None) -> str:
    if not name:
        return "uploaded_file"
    stem = Path(str(name)).stem.replace(" ", "_").replace("/", "_").replace("\\", "_")
    keep = "".join(ch for ch in stem if ch.isalnum() or ch in {"_", "-", "."})
    return keep[:80] or "uploaded_file"


def _safe_partition_value(value: Any, prefix: str = "value") -> str:
    if value is None:
        raw = "unknown"
    else:
        try:
            if pd.isna(value):
                raw = "unknown"
            else:
                raw = str(value)
        except Exception:
            raw = str(value)
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in raw.strip())
    cleaned = cleaned.strip("_") or "unknown"
    return f"{prefix}_{cleaned}" if prefix else cleaned


def _category_columns(df: pd.DataFrame) -> tuple[str | None, str | None, str | None]:
    category_id_col = _first_existing(df, ["item_category_id", "segment_id", "category_id"])
    category_name_col = _first_existing(df, ["item_category_name", "segment_name", "category_name"])
    category_group_col = _first_existing(df, ["category_group", "segment_group", "group_name"])
    return category_id_col, category_name_col, category_group_col


def _partition_uploaded_batch_by_category(df: pd.DataFrame) -> list[tuple[dict[str, Any], pd.DataFrame]]:
    """Return category/segment partitions for uploaded-batch predictions.

    The full combined file is always saved separately. These partitions add the
    category/segment-level files the user needs for catalog exports.
    """
    category_id_col, category_name_col, category_group_col = _category_columns(df)
    if category_id_col is None or category_id_col not in df.columns:
        return []

    work = df.copy()
    work[category_id_col] = work[category_id_col].where(work[category_id_col].notna(), "unknown")
    partitions: list[tuple[dict[str, Any], pd.DataFrame]] = []
    for category_id, group in work.groupby(category_id_col, dropna=False):
        meta: dict[str, Any] = {
            "partition_type": "category",
            "item_category_id": category_id,
            "segment_id": category_id,
        }
        if category_name_col and category_name_col in group.columns:
            values = group[category_name_col].dropna().astype(str).drop_duplicates().tolist()
            meta["item_category_name"] = values[0] if values else ""
            meta["segment_name"] = meta["item_category_name"]
        if category_group_col and category_group_col in group.columns:
            values = group[category_group_col].dropna().astype(str).drop_duplicates().tolist()
            meta["category_group"] = values[0] if values else ""
        partitions.append((meta, group.copy()))
    return partitions


def _history_record_for_saved_file(
    *,
    s3_uri: str,
    df: pd.DataFrame,
    source_filename: str,
    feature_source: str,
    feature_completion_mode: str,
    bucket_source: str,
    partition_type: str,
    partition_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pred = pd.to_numeric(df.get("prediction", pd.Series(dtype=float)), errors="coerce").fillna(0)
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_file": source_filename,
        "scope": "Archivo cargado / predicción batch",
        "partition_type": partition_type,
        "records_count": int(len(df)),
        "total_prediction": float(pred.sum()),
        "mean_prediction": float(pred.mean()) if len(df) else 0.0,
        "s3_uri": s3_uri,
        "feature_completion_mode": feature_completion_mode,
        "feature_source": feature_source,
        "bucket_source": bucket_source,
    }
    if partition_meta:
        record.update(partition_meta)
    return record


def _direct_s3_upload_csv(
    df: pd.DataFrame,
    *,
    source_filename: str = "uploaded.csv",
    feature_source: str = "unknown",
    feature_completion_mode: str = "unknown",
) -> tuple[str, str]:
    """Save uploaded-batch predictions under a single exclusive S3 prefix.

    Important: uploaded predictions are NOT partitioned by category. They are
    stored in one shared predictions folder, while normal CFO exports can be
    partitioned by shop/category in app/batch_exports/.
    """
    bucket, bucket_source, prefix = _uploaded_predictions_prefix()
    if not bucket:
        raise RuntimeError("No encontré bucket S3. Define MODEL_BUCKET o MODELOPS_BUCKET en el entorno de ECS/local.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    file_stem = _safe_file_stem(source_filename)
    client = _s3_client()

    key = f"{prefix}/uploaded_batch_predictions_{timestamp}_{file_stem}.csv"
    client.put_object(Bucket=bucket, Key=key, Body=df.to_csv(index=False).encode("utf-8"), ContentType="text/csv")
    uri = f"s3://{bucket}/{key}"

    record = _history_record_for_saved_file(
        s3_uri=uri,
        df=df,
        source_filename=source_filename,
        feature_source=feature_source,
        feature_completion_mode=feature_completion_mode,
        bucket_source=bucket_source,
        partition_type="uploaded_prediction_file",
        partition_meta={"partition_label": "sin_particiones"},
    )
    history_uri = _append_uploaded_prediction_history(record)
    return uri, history_uri

def _save_uploaded_to_s3(
    result_df: pd.DataFrame,
    *,
    source_filename: str,
    feature_source: str,
    feature_completion_mode: str,
) -> tuple[str, str]:
    return _direct_s3_upload_csv(
        result_df,
        source_filename=source_filename,
        feature_source=feature_source,
        feature_completion_mode=feature_completion_mode,
    )


def _list_uploaded_prediction_objects(limit: int = 50) -> pd.DataFrame:
    bucket, _source, prefix = _uploaded_predictions_prefix()
    if not bucket:
        return pd.DataFrame()
    try:
        response = _s3_client().list_objects_v2(Bucket=bucket, Prefix=prefix.rstrip("/") + "/")
    except Exception:
        return pd.DataFrame()
    rows = []
    for obj in response.get("Contents", []):
        key = obj.get("Key", "")
        if not key.endswith(".csv"):
            continue
        rows.append({
            "last_modified": obj.get("LastModified"),
            "size_bytes": obj.get("Size"),
            "s3_uri": f"s3://{bucket}/{key}",
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("last_modified", ascending=False).head(limit)


def render_uploaded_prediction_history() -> None:
    st.subheader("Historial de predicciones por archivo cargado")
    st.caption("Historial exclusivo de resultados generados desde la carga CSV de inferencia batch. No mezcla archivos CFO normales.")
    history = _read_uploaded_prediction_history()
    if not history.empty:
        cols = [
            "created_at",
            "source_file",
            "partition_type",
            "partition_label",
            "segment_id",
            "segment_name",
            "item_category_id",
            "item_category_name",
            "category_group",
            "records_count",
            "total_prediction",
            "mean_prediction",
            "feature_completion_mode",
            "s3_uri",
        ]
        shown_cols = [col for col in cols if col in history.columns]
        st.dataframe(history[shown_cols].head(100), width="stretch", height=320)
        return

    objects = _list_uploaded_prediction_objects()
    if not objects.empty:
        st.info("No encontré el CSV de historial, pero sí objetos guardados en el prefijo exclusivo de predicciones cargadas.")
        st.dataframe(objects, width="stretch", height=260)
        return

    expected, _prefix = _uploaded_s3_template()
    if expected:
        st.info("Aún no hay predicciones cargadas guardadas en S3. Cuando guardes, aparecerán aquí.")
        st.code(expected)
    else:
        st.warning("No se puede consultar historial porque no encontré MODEL_BUCKET / MODELOPS_BUCKET en el entorno.")

def render_uploaded_batch_inference(
    *,
    model_payload: dict[str, Any] | Any,
    enrich_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    visible_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    render_dataframe_fn: Callable[..., None] | None = None,
    upload_batch_dataframe_to_s3: Callable[..., str] | None = None,
    insert_batch_export: Callable[..., None] | None = None,
    disable_rds_writes: bool = True,
) -> None:
    st.divider()
    st.subheader("Batch por archivo cargado")
    st.write(
        "Puedes cargar un CSV con features completas o un CSV tipo test/validación con `ID`, `shop_id`, `item_id`. "
        "La app completará las features desde `data/prep/test_features.parquet` y pronosticará una fila por par producto-tienda."
    )

    feature_cols = required_feature_columns(model_payload)
    with st.expander("Columnas requeridas por el modelo"):
        if feature_cols:
            st.write(f"Total de columnas requeridas: **{len(feature_cols)}**")
            st.dataframe(pd.DataFrame({"feature_col": feature_cols}), width="stretch", height=280)
        else:
            st.warning("No pude leer feature_cols del modelo cargado.")

    uploaded = st.file_uploader("Subir CSV", type=["csv"], key="uploaded_batch_csv_for_inference")
    if uploaded is None:
        st.caption("Formatos aceptados: features completas, pares shop_id/item_id, filtros por shop_id/item_id/category o catálogo completo.")
        return

    try:
        uploaded_df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"No pude leer el CSV: {exc}")
        return

    st.write(f"Archivo cargado: **{uploaded.name}** — filas: **{len(uploaded_df):,}**, columnas: **{len(uploaded_df.columns):,}**")

    missing_initial = [col for col in feature_cols if col not in uploaded_df.columns]
    has_all_features = not missing_initial

    mode_labels = {
        "uploaded_as_is": "Usar filas del CSV tal cual (requiere features completas)",
        "complete_from_reference": "Completar/expandir usando features preparadas del dataset",
        "full_reference_catalog": "Pronosticar catálogo completo del dataset",
    }
    default_mode = "uploaded_as_is" if has_all_features else "complete_from_reference"
    mode_label = st.radio(
        "Cómo interpretar el CSV cargado",
        options=list(mode_labels.values()),
        index=list(mode_labels.keys()).index(default_mode),
        horizontal=False,
        key="uploaded_batch_feature_completion_mode_label",
        help="Si el CSV solo trae shop_id/item_id, usa la opción de completar/expandir. Para todos los productos, usa catálogo completo.",
    )
    feature_completion_mode = {v: k for k, v in mode_labels.items()}[mode_label]

    fill_missing = False
    if missing_initial and feature_completion_mode == "uploaded_as_is":
        st.warning(f"Faltan {len(missing_initial)} columnas requeridas por el modelo.")
        st.dataframe(pd.DataFrame({"missing_feature": missing_initial[:200]}), width="stretch", height=220)
        fill_missing = st.checkbox(
            "Modo prueba: rellenar columnas faltantes con 0",
            value=False,
            key="uploaded_batch_fill_missing_zero",
            help="Útil para probar el flujo con CSVs de ejemplo. Para inferencia real, usa features preparadas del dataset.",
        )
    elif missing_initial:
        st.info(
            f"El CSV no trae {len(missing_initial)} features del modelo. "
            "Intentaré completarlas desde `test_features.parquet` usando los ids/filtros del archivo cargado."
        )

    use_hybrid_policy = st.checkbox(
        "Aplicar política híbrida de ruteo a este archivo cargado",
        value=True,
        key="uploaded_batch_use_hybrid_policy",
        help="Usa recency, lags, rolling_mean y nonzero_rate para decidir entre regla cero, naive, especialista recurrente o modelo HGB.",
    )

    run_clicked = st.button("Ejecutar inferencia sobre archivo cargado", key="run_uploaded_batch_inference")

    cache_key = "uploaded_batch_last_inference_result_v9"

    if run_clicked:
        try:
            batch_result = run_uploaded_batch(
                uploaded_df,
                model_payload,
                fill_missing_with_zero=fill_missing,
                use_hybrid_policy=use_hybrid_policy,
                feature_completion_mode=feature_completion_mode,
            )
        except Exception as exc:
            st.error(f"No pude ejecutar inferencia batch: {exc}")
            return

        if batch_result.dataframe.empty:
            st.error("No ejecuté inferencia porque faltan columnas y no activaste el modo de prueba con ceros.")
            st.info("Cambia a 'Completar/expandir usando features preparadas del dataset' o genera CSVs reales con test_features.")
            return

        result_df = _safe_enrich(batch_result.dataframe, enrich_fn)
        visible_df = _safe_visible(result_df, visible_fn)

        # Persist the inference result in Streamlit session_state. This is critical:
        # clicking the separate S3 save button triggers a rerun, so the previous
        # result must survive even though the execute button is no longer True.
        st.session_state[cache_key] = {
            "result_df": result_df.copy(),
            "visible_df": visible_df.copy(),
            "source_filename": getattr(uploaded, "name", "uploaded.csv"),
            "feature_source": batch_result.feature_source,
            "feature_completion_mode": feature_completion_mode,
            "used_zero_fill": bool(batch_result.used_zero_fill),
            "used_hybrid_router": bool(batch_result.used_hybrid_router),
            "specialist_source": batch_result.specialist_source,
            "unmatched_uploaded_rows": int(batch_result.unmatched_uploaded_rows),
        }
    elif cache_key not in st.session_state:
        render_uploaded_prediction_history()
        return

    cached = st.session_state.get(cache_key)
    if not cached:
        render_uploaded_prediction_history()
        return

    result_df = cached["result_df"].copy()
    visible_df = cached["visible_df"].copy()
    source_filename = cached.get("source_filename", getattr(uploaded, "name", "uploaded.csv"))
    cached_feature_source = cached.get("feature_source", "unknown")
    cached_mode = cached.get("feature_completion_mode", feature_completion_mode)

    c1, c2, c3 = st.columns(3)
    c1.metric("Registros inferidos", f"{len(result_df):,}")
    c2.metric("Pronóstico total", f"{result_df['prediction'].sum():,.2f}")
    c3.metric("Promedio", f"{result_df['prediction'].mean():.4f}")

    st.caption(f"Fuente de features usada: {cached_feature_source}")
    if cached.get("unmatched_uploaded_rows", 0):
        st.info(
            f"{cached['unmatched_uploaded_rows']} filas no existían en test_features/test_pairs. "
            "Se trataron como cold-start sin historial conocido y se rutearon con reglas conservadoras."
        )
    if cached.get("used_zero_fill"):
        st.info("Inferencia ejecutada en modo prueba: algunas columnas faltantes fueron rellenadas con 0.")
    if cached.get("used_hybrid_router"):
        st.success("Estas predicciones usan la política híbrida: inactive, naive reciente, especialista recurrente o HGB según señales históricas.")
        st.caption(f"Fuente del especialista recurrente: {cached.get('specialist_source', 'unknown')}")
    else:
        st.info("Estas predicciones usan inferencia directa del modelo, sin política híbrida.")

    if render_dataframe_fn is not None:
        render_dataframe_fn(visible_df, "Resultado de inferencia por archivo cargado", max_rows=1000, height=420)
    else:
        st.dataframe(visible_df.head(1000), width="stretch", height=420)

    csv_bytes = visible_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar resultado de inferencia cargada",
        data=csv_bytes,
        file_name="uploaded_batch_predictions.csv",
        mime="text/csv",
        key="download_uploaded_batch_predictions",
    )

    st.subheader("Actualizar tablero con este resultado")
    st.caption(
        "Esto actualiza las pestañas de Resumen/Batch CFO que dependen del forecast batch. "
        "No actualiza métricas de performance porque el archivo cargado todavía no tiene y real."
    )
    if st.button("Usar este resultado como forecast del tablero", key="use_uploaded_batch_as_dashboard_forecast"):
        st.session_state["uploaded_batch_dashboard_active"] = True
        st.session_state["uploaded_batch_dashboard_df"] = result_df.copy()
        st.session_state["uploaded_batch_dashboard_source"] = cached_feature_source
        st.success("El tablero usará este resultado como forecast durante esta sesión.")
        try:
            st.rerun()
        except Exception:
            pass

    if st.session_state.get("uploaded_batch_dashboard_active"):
        if st.button("Restaurar forecast original ModelOps", key="restore_modelops_batch_forecast_from_upload"):
            for key in ["uploaded_batch_dashboard_active", "uploaded_batch_dashboard_df", "uploaded_batch_dashboard_source"]:
                st.session_state.pop(key, None)
            st.success("Forecast original restaurado para esta sesión.")
            try:
                st.rerun()
            except Exception:
                pass

    st.subheader("Guardar resultado en S3")
    expected_uri, _prefix = _uploaded_s3_template()
    if expected_uri:
        st.caption("Las predicciones por archivo cargado se guardarán en una sola carpeta S3 exclusiva, sin particionar por categoría:")
        st.code(expected_uri)
        st.info(
            "Nota: este flujo guarda un único archivo de predicciones cargadas. "
            "Las particiones por shop_id o category_id aplican al flujo normal de archivo CFO, no a este batch upload."
        )
    else:
        st.warning("No encontré bucket S3 en variables de entorno. Define MODEL_BUCKET o MODELOPS_BUCKET para guardar resultados.")

    if "last_uploaded_prediction_s3_uri" in st.session_state:
        st.success("Último resultado de archivo cargado guardado en S3 durante esta sesión:")
        st.code(st.session_state["last_uploaded_prediction_s3_uri"])
        history_uri = st.session_state.get("last_uploaded_prediction_history_s3_uri")
        if history_uri:
            st.caption("Historial exclusivo actualizado en:")
            st.code(history_uri)

    if st.button("Guardar resultado de archivo cargado en S3", key="save_uploaded_batch_result_to_s3"):
        try:
            s3_uri, history_uri = _save_uploaded_to_s3(
                result_df,
                source_filename=source_filename,
                feature_source=cached_feature_source,
                feature_completion_mode=cached_mode,
            )
            st.session_state["last_uploaded_prediction_s3_uri"] = s3_uri
            st.session_state["last_uploaded_prediction_history_s3_uri"] = history_uri
            st.success("Resultado de inferencia cargada guardado en S3.")
            st.caption("Ruta S3 exacta donde quedó guardado el resultado:")
            st.code(s3_uri)
            st.caption("Historial exclusivo de predicciones cargadas actualizado en:")
            st.code(history_uri)

            if insert_batch_export is not None and not disable_rds_writes:
                insert_batch_export(
                    scope="Archivo cargado / predicción batch",
                    shop_id=None,
                    records_count=len(result_df),
                    total_prediction=float(result_df["prediction"].sum()),
                    s3_uri=s3_uri,
                )
        except Exception as exc:
            st.error(f"No pude guardar en S3/RDS: {exc}")
            st.info("Revisa que MODEL_BUCKET o MODELOPS_BUCKET esté configurado en ECS/local y que el rol/usuario tenga s3:PutObject sobre app/batch_uploads/.")

    render_uploaded_prediction_history()
