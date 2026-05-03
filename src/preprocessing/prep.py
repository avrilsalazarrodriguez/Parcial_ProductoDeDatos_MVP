"""
prep.py

Construye datasets mensuales para entrenamiento/validación/test con features de serie de
tiempo para problema de inventario.

Entradas
--------
- data/raw/sales_train.csv
- data/raw/test.csv

Salidas
-------
- data/prep/train.parquet
- data/prep/valid.parquet
- data/prep/test_features.parquet
- data/prep/test_pairs.parquet
- data/prep/meta.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gc
import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CANTIDAD_DIA_MAX,
    CANTIDAD_DIA_MIN,
    CUANTIL_PRECIO,
    LAGS,
    LOG_DIR,
    PRECIO_MAX,
    PREP_DIR,
    RAW_DIR,
    RECENCY_FILL,
    TARGET_MAX,
    TARGET_MIN,
)
from src.utils.data_validation import require_columns, require_file
from src.utils.logging_utils import get_logger

RAW_DIR = Path("/opt/ml/processing/input")

@dataclass(frozen=True)
class RawData:
    """Datos raw leídos desde CSV."""

    train: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class Aggregates:
    """Agregados laggeados sin leakage."""

    global_agg: pd.DataFrame
    item_agg: pd.DataFrame
    shop_agg: pd.DataFrame


@dataclass(frozen=True)
class PanelBuild:
    """Panel denso y blocks calculados."""

    panel: pd.DataFrame
    max_block: int
    test_block: int


@dataclass(frozen=True)
class PrepOutputs:
    """Salidas finales de prep."""

    train_out: pd.DataFrame
    valid_out: pd.DataFrame
    test_out: pd.DataFrame
    test_pairs: pd.DataFrame
    meta: dict


@contextmanager
def log_stage(logger: logging.Logger, label: str):
    """
    Context manager para loggear duración de una etapa.
    """
    start = time.perf_counter()
    logger.info("Iniciando etapa: %s", label)
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        logger.info("Etapa terminada: %s duracion_seg=%.2f", label, duration)


def log_df_info(logger: logging.Logger, name: str, df: pd.DataFrame) -> None:
    """
    Loggea información básica de un DataFrame (shape + NA total) sin imprimir datos.
    """
    logger.info("%s: filas=%d cols=%d", name, len(df), df.shape[1])
    na_total = int(df.isna().sum().sum())
    if na_total > 0:
        logger.warning("%s: total_NA=%d", name, na_total)


def cargar_datos_raw(raw_dir: Path) -> RawData:
    """
    Carga los CSV originales y estandariza nombres de columnas.
    """
    ruta_train = raw_dir / "sales_train.csv"
    ruta_test = raw_dir / "test.csv"

    require_file(ruta_train)
    require_file(ruta_test)

    train = pd.read_csv(ruta_train, encoding="utf-8")
    test = pd.read_csv(ruta_test, encoding="utf-8")

    train.columns = train.columns.str.strip()
    test.columns = test.columns.str.strip()

    require_columns(
        train,
        ["date_block_num", "shop_id", "item_id", "item_price", "item_cnt_day"],
        "sales_train.csv",
    )
    require_columns(test, ["ID", "shop_id", "item_id"], "test.csv")

    return RawData(train=train, test=test)


def tipificar_y_filtrar(raw: RawData) -> RawData:
    """
    Convierte columnas a numérico, elimina filas inválidas y filtra outliers básicos.
    """
    train = raw.train.copy()
    test = raw.test.copy()

    train_cols = ["date_block_num", "shop_id", "item_id", "item_price", "item_cnt_day"]
    test_cols = ["ID", "shop_id", "item_id"]

    for col in train_cols:
        train[col] = pd.to_numeric(train[col], errors="coerce")
    for col in test_cols:
        test[col] = pd.to_numeric(test[col], errors="coerce")

    train = train.dropna(subset=train_cols)
    test = test.dropna(subset=test_cols)

    train["date_block_num"] = train["date_block_num"].astype(np.int16)
    train["shop_id"] = train["shop_id"].astype(np.int16)
    train["item_id"] = train["item_id"].astype(np.int16)

    test["ID"] = test["ID"].astype(np.int32)
    test["shop_id"] = test["shop_id"].astype(np.int16)
    test["item_id"] = test["item_id"].astype(np.int16)

    train = train[train["item_price"] >= 0]
    train = train[train["item_price"] < PRECIO_MAX]
    train = train[
        (train["item_cnt_day"] > CANTIDAD_DIA_MIN)
        & (train["item_cnt_day"] < CANTIDAD_DIA_MAX)
    ]

    return RawData(train=train, test=test)


def construir_target_mensual(train: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega ventas diarias a target mensual por (mes, tienda, producto).
    """
    monthly = train.groupby(
        ["date_block_num", "shop_id", "item_id"], as_index=False
    ).agg(item_cnt_month=("item_cnt_day", "sum"), price_mean=("item_price", "mean"))

    monthly["item_cnt_month"] = (
        monthly["item_cnt_month"].clip(TARGET_MIN, TARGET_MAX).astype(np.float32)
    )
    monthly["price_mean"] = monthly["price_mean"].astype(np.float32)
    return monthly


def agregados_laggeados(monthly: pd.DataFrame) -> Aggregates:
    """
    Agregados global/item/shop con shift +1 mes para evitar leakage.
    """
    global_agg = monthly.groupby("date_block_num", as_index=False).agg(
        global_mean=("item_cnt_month", "mean"),
        global_sum=("item_cnt_month", "sum"),
        global_pairs=("item_cnt_month", "size"),
    )
    global_agg["date_block_num"] = (global_agg["date_block_num"] + 1).astype(np.int16)

    item_agg = monthly.groupby(["date_block_num", "item_id"], as_index=False).agg(
        item_mean=("item_cnt_month", "mean"),
        item_shops=("shop_id", "nunique"),
        item_price_mean=("price_mean", "mean"),
    )
    item_agg["date_block_num"] = (item_agg["date_block_num"] + 1).astype(np.int16)

    shop_agg = monthly.groupby(["date_block_num", "shop_id"], as_index=False).agg(
        shop_mean=("item_cnt_month", "mean"),
        shop_items=("item_id", "nunique"),
    )
    shop_agg["date_block_num"] = (shop_agg["date_block_num"] + 1).astype(np.int16)

    return Aggregates(global_agg=global_agg, item_agg=item_agg, shop_agg=shop_agg)


def construir_panel_denso(
    test: pd.DataFrame, monthly: pd.DataFrame, aggs: Aggregates
) -> PanelBuild:
    """
    Construye panel denso por meses para todos los pares del test.
    """
    max_block = int(monthly["date_block_num"].max())
    test_block = max_block + 1

    test_pairs = test[["shop_id", "item_id"]].drop_duplicates()
    months = np.arange(0, test_block + 1, dtype=np.int16)

    panel = pd.DataFrame(
        {
            "date_block_num": np.tile(months, len(test_pairs)).astype(np.int16),
            "shop_id": np.repeat(test_pairs["shop_id"].values, len(months)).astype(
                np.int16
            ),
            "item_id": np.repeat(test_pairs["item_id"].values, len(months)).astype(
                np.int16
            ),
        }
    )

    panel = panel.merge(
        monthly[
            ["date_block_num", "shop_id", "item_id", "item_cnt_month", "price_mean"]
        ],
        on=["date_block_num", "shop_id", "item_id"],
        how="left",
    )

    panel["item_cnt_month"] = panel["item_cnt_month"].fillna(0).astype(np.float32)
    panel["price_mean"] = panel["price_mean"].astype(np.float32)

    panel["month"] = (panel["date_block_num"] % 12).astype(np.int8)
    panel["year"] = (panel["date_block_num"] // 12).astype(np.int8)
    panel["month_sin"] = np.sin(2 * np.pi * panel["month"] / 12).astype(np.float32)
    panel["month_cos"] = np.cos(2 * np.pi * panel["month"] / 12).astype(np.float32)

    panel = panel.merge(aggs.global_agg, on="date_block_num", how="left")
    panel = panel.merge(aggs.item_agg, on=["date_block_num", "item_id"], how="left")
    panel = panel.merge(aggs.shop_agg, on=["date_block_num", "shop_id"], how="left")

    fill_cols = [
        "global_mean",
        "global_sum",
        "global_pairs",
        "item_mean",
        "item_shops",
        "item_price_mean",
        "shop_mean",
        "shop_items",
    ]
    for col in fill_cols:
        panel[col] = panel[col].fillna(0).astype(np.float32)

    return PanelBuild(panel=panel, max_block=max_block, test_block=test_block)


def agregar_features_precio(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega features de precio: log precio observado, promedio por item y último precio conocido.
    """
    panel = panel.copy()
    obs = panel["price_mean"].dropna()
    cap = float(obs.quantile(CUANTIL_PRECIO)) if len(obs) else 0.0

    clipped = panel["price_mean"].clip(0, cap)
    panel["log_price_obs"] = np.where(
        panel["price_mean"].notna(), np.log1p(clipped), np.nan
    ).astype(np.float32)

    panel["item_log_price_mean"] = np.log1p(
        panel["item_price_mean"].clip(0, cap)
    ).astype(np.float32)

    panel.sort_values(
        ["shop_id", "item_id", "date_block_num"], inplace=True, ignore_index=True
    )
    grouped = panel.groupby(["shop_id", "item_id"], sort=False)

    panel["log_price_last"] = grouped["log_price_obs"].ffill()
    panel["log_price_last"] = grouped["log_price_last"].shift(1).astype(np.float32)

    panel["price_missing_last"] = panel["log_price_last"].isna().astype(np.int8)
    panel["log_price_last"] = (
        panel["log_price_last"].fillna(panel["item_log_price_mean"]).astype(np.float32)
    )
    panel["price_gap_item"] = (
        panel["log_price_last"] - panel["item_log_price_mean"]
    ).astype(np.float32)

    return panel


def agregar_lags_y_ventanas(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega lags del target y features de ventana (3 y 6 meses).
    """
    panel = panel.copy()
    panel.sort_values(
        ["shop_id", "item_id", "date_block_num"], inplace=True, ignore_index=True
    )
    grouped = panel.groupby(["shop_id", "item_id"], sort=False)

    for lag in LAGS:
        panel[f"cnt_lag_{lag}"] = (
            grouped["item_cnt_month"].shift(lag).fillna(0).astype(np.float32)
        )

    eps = 1e-6
    l1, l2, l3 = panel["cnt_lag_1"], panel["cnt_lag_2"], panel["cnt_lag_3"]
    l4, l5, l6 = panel["cnt_lag_4"], panel["cnt_lag_5"], panel["cnt_lag_6"]
    l12 = panel["cnt_lag_12"]

    panel["sum_3"] = (l1 + l2 + l3).astype(np.float32)
    panel["mean_3"] = (panel["sum_3"] / 3.0).astype(np.float32)
    panel["max_3"] = np.maximum.reduce([l1.values, l2.values, l3.values]).astype(
        np.float32
    )
    panel["nz_3"] = (
        (l1 > 0).astype(np.int8) + (l2 > 0).astype(np.int8) + (l3 > 0).astype(np.int8)
    )

    mean_sq_3 = ((l1 * l1 + l2 * l2 + l3 * l3) / 3.0).astype(np.float32)
    panel["std_3"] = np.sqrt(
        np.maximum(mean_sq_3 - panel["mean_3"] * panel["mean_3"], 0)
    ).astype(np.float32)

    panel["sum_6"] = (l1 + l2 + l3 + l4 + l5 + l6).astype(np.float32)
    panel["mean_6"] = (panel["sum_6"] / 6.0).astype(np.float32)
    panel["max_6"] = np.maximum.reduce(
        [l1.values, l2.values, l3.values, l4.values, l5.values, l6.values]
    ).astype(np.float32)
    panel["nz_6"] = (
        (l1 > 0) + (l2 > 0) + (l3 > 0) + (l4 > 0) + (l5 > 0) + (l6 > 0)
    ).astype(np.int8)

    mean_sq_6 = (
        (l1 * l1 + l2 * l2 + l3 * l3 + l4 * l4 + l5 * l5 + l6 * l6) / 6.0
    ).astype(np.float32)
    panel["std_6"] = np.sqrt(
        np.maximum(mean_sq_6 - panel["mean_6"] * panel["mean_6"], 0)
    ).astype(np.float32)

    panel["rate_6"] = (panel["nz_6"] / 6.0).astype(np.float32)
    panel["mean_nonzero_6"] = (
        panel["sum_6"] / (panel["nz_6"].astype(np.float32) + eps)
    ).astype(np.float32)
    panel["interval_6"] = (6.0 / (panel["nz_6"].astype(np.float32) + eps)).astype(
        np.float32
    )

    panel["active_1"] = (l1 > 0).astype(np.int8)
    panel["dead_6"] = (panel["nz_6"] == 0).astype(np.int8)

    panel["trend_1_3"] = (l1 - l3).astype(np.float32)
    panel["trend_1_12"] = (l1 - l12).astype(np.float32)
    panel["ratio_1_12"] = (l1 / (l12 + eps)).astype(np.float32)

    return panel


def agregar_recency_e_historial(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega recency e historial acumulado sin leakage.
    """
    panel = panel.copy()

    sale_month = panel["date_block_num"].where(panel["item_cnt_month"] > 0, np.nan)
    last_sale_inclusive = sale_month.groupby(
        [panel["shop_id"], panel["item_id"]]
    ).ffill()
    last_sale_past = last_sale_inclusive.groupby(
        [panel["shop_id"], panel["item_id"]]
    ).shift(1)

    panel["recency"] = (
        (panel["date_block_num"] - last_sale_past)
        .fillna(RECENCY_FILL)
        .clip(0, RECENCY_FILL)
        .astype(np.int16)
    )

    panel.sort_values(
        ["shop_id", "item_id", "date_block_num"], inplace=True, ignore_index=True
    )
    grouped = panel.groupby(["shop_id", "item_id"], sort=False)

    panel["sold"] = (panel["item_cnt_month"] > 0).astype(np.int8)
    panel["sold_cum"] = grouped["sold"].cumsum().astype(np.int16)
    panel["sold_cum_lag1"] = grouped["sold_cum"].shift(1).fillna(0).astype(np.int16)

    panel["sales_cum"] = grouped["item_cnt_month"].cumsum().astype(np.float32)
    panel["sales_cum_lag1"] = grouped["sales_cum"].shift(1).fillna(0).astype(np.float32)
    panel["log_sales_cum_lag1"] = np.log1p(panel["sales_cum_lag1"]).astype(np.float32)

    panel["never_sold_before"] = (panel["sold_cum_lag1"] == 0).astype(np.int8)
    return panel


def columnas_features_finales() -> list[str]:
    """
    Lista de features utilizadas por el modelo final.
    """
    return [
        "date_block_num",
        "month",
        "year",
        "month_sin",
        "month_cos",
        "shop_id",
        "item_id",
        "global_mean",
        "global_sum",
        "global_pairs",
        "item_mean",
        "item_shops",
        "shop_mean",
        "shop_items",
        "log_price_last",
        "item_log_price_mean",
        "price_gap_item",
        "price_missing_last",
        "cnt_lag_1",
        "cnt_lag_2",
        "cnt_lag_3",
        "cnt_lag_4",
        "cnt_lag_5",
        "cnt_lag_6",
        "cnt_lag_12",
        "sum_3",
        "mean_3",
        "std_3",
        "max_3",
        "nz_3",
        "sum_6",
        "mean_6",
        "std_6",
        "max_6",
        "nz_6",
        "rate_6",
        "mean_nonzero_6",
        "interval_6",
        "active_1",
        "dead_6",
        "recency",
        "trend_1_3",
        "trend_1_12",
        "ratio_1_12",
        "sold_cum_lag1",
        "log_sales_cum_lag1",
        "never_sold_before",
    ]


def construir_outputs(panel_build: PanelBuild, feature_cols: list[str]) -> PrepOutputs:
    """
    Construye train/valid/test + meta a partir del panel ya featureado.
    """
    panel = panel_build.panel
    max_block = panel_build.max_block
    test_block = panel_build.test_block

    train_df = panel[panel["date_block_num"] <= max_block - 1].copy()
    valid_df = panel[panel["date_block_num"] == max_block].copy()
    test_df = panel[panel["date_block_num"] == test_block].copy()

    train_out = train_df[feature_cols].copy()
    train_out["y"] = train_df["item_cnt_month"].astype(np.float32).values

    valid_out = valid_df[feature_cols].copy()
    valid_out["y"] = valid_df["item_cnt_month"].astype(np.float32).values

    test_out = test_df[feature_cols].copy()
    test_pairs = test_df[["shop_id", "item_id"]].copy()

    meta = {
        "feature_cols": feature_cols,
        "max_block": int(max_block),
        "test_block": int(test_block),
    }

    return PrepOutputs(
        train_out=train_out,
        valid_out=valid_out,
        test_out=test_out,
        test_pairs=test_pairs,
        meta=meta,
    )


def guardar_salidas(prep_dir: Path, outputs: PrepOutputs) -> None:
    """
    Guarda datasets preparados y metadata.
    Compatible con SageMaker Processing.
    """

    import os

    # Detectamos si estamos en SageMaker Processing
    if os.environ.get("SM_MODEL_DIR") or Path("/opt/ml/processing").exists():
        base_dir = Path("/opt/ml/processing/output")

        train_dir = base_dir / "train"
        valid_dir = base_dir / "validation"
        test_dir = base_dir / "test"

        train_dir.mkdir(parents=True, exist_ok=True)
        valid_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        # Guardamos parquet en rutas que espera SageMaker
        outputs.train_out.to_parquet(train_dir / "train.parquet", index=False)
        outputs.valid_out.to_parquet(valid_dir / "valid.parquet", index=False)
        outputs.test_out.to_csv(test_dir / "test.csv", index=False)

        # meta también (lo va a usar training)
        with open(train_dir / "meta.json", "w") as f:
            json.dump(outputs.meta, f)

    else:
        # Comportamiento local
        prep_dir.mkdir(parents=True, exist_ok=True)

        outputs.train_out.to_parquet(prep_dir / "train.parquet", index=False)
        outputs.valid_out.to_parquet(prep_dir / "valid.parquet", index=False)
        outputs.test_out.to_parquet(prep_dir / "test_features.parquet", index=False)
        outputs.test_pairs.to_parquet(prep_dir / "test_pairs.parquet", index=False)

        with (prep_dir / "meta.json").open("w", encoding="utf-8") as file:
            json.dump(outputs.meta, file, indent=2)


def construir_panel_con_features(clean: RawData) -> PanelBuild:
    """
    Arma panel denso y aplica features (precio, lags/ventanas, recency/historial).
    """
    monthly = construir_target_mensual(clean.train)
    aggs = agregados_laggeados(monthly)
    panel_build = construir_panel_denso(clean.test, monthly, aggs)

    del monthly, aggs
    gc.collect()

    panel = panel_build.panel
    panel = agregar_features_precio(panel)
    panel = agregar_lags_y_ventanas(panel)
    panel = agregar_recency_e_historial(panel)

    return PanelBuild(
        panel=panel, max_block=panel_build.max_block, test_block=panel_build.test_block
    )


def log_outputs_summary(
    logger: logging.Logger, outputs: PrepOutputs, panel_build: PanelBuild
) -> None:
    """
    Loggea un resumen de salidas del pipeline.
    """
    log_df_info(logger, "train_out", outputs.train_out)
    log_df_info(logger, "valid_out", outputs.valid_out)
    log_df_info(logger, "test_out", outputs.test_out)
    log_df_info(logger, "test_pairs", outputs.test_pairs)
    logger.info(
        "Blocks: max_block=%d test_block=%d",
        panel_build.max_block,
        panel_build.test_block,
    )

    y_train = outputs.train_out["y"]
    y_valid = outputs.valid_out["y"]

    logger.info(
        "Target y train: min=%.2f p50=%.2f max=%.2f mean=%.3f",
        float(y_train.min()),
        float(y_train.median()),
        float(y_train.max()),
        float(y_train.mean()),
    )
    logger.info(
        "Target y valid: min=%.2f p50=%.2f max=%.2f mean=%.3f",
        float(y_valid.min()),
        float(y_valid.median()),
        float(y_valid.max()),
        float(y_valid.mean()),
    )

    peaks = [5, 10, 15]
    counts = [int((y_valid >= p).sum()) for p in peaks]
    logger.info(
        "Picos valid: y>=5=%d y>=10=%d y>=15=%d", counts[0], counts[1], counts[2]
    )


def run_prep_pipeline(
    logger: logging.Logger,
    raw_dir: Path,
    prep_dir: Path
) -> None:
    """
    Ejecuta el pipeline completo de prep, con logging por etapa.

    Parameters
    ----------
    logger : logging.Logger
        Logger para seguimiento de ejecución.
    raw_dir : Path
        Directorio donde están los datos raw (SageMaker o local).
    prep_dir : Path
        Directorio donde se guardan los outputs.
    """
    with log_stage(logger, "cargar_datos_raw"):
        raw = cargar_datos_raw(raw_dir)
        log_df_info(logger, "train_raw", raw.train)
        log_df_info(logger, "test_raw", raw.test)

    with log_stage(logger, "tipificar_y_filtrar"):
        clean = tipificar_y_filtrar(raw)
        log_df_info(logger, "train_clean", clean.train)
        log_df_info(logger, "test_clean", clean.test)

        dropped_train = len(raw.train) - len(clean.train)
        dropped_test = len(raw.test) - len(clean.test)
        if dropped_train > 0 or dropped_test > 0:
            logger.warning(
                "Filas eliminadas por limpieza: train=%d test=%d",
                dropped_train,
                dropped_test,
            )

    with log_stage(logger, "construir_panel_con_features"):
        panel_build = construir_panel_con_features(clean)
        log_df_info(logger, "panel", panel_build.panel)

    with log_stage(logger, "construir_outputs"):
        feature_cols = columnas_features_finales()
        logger.info("Feature cols: n=%d | lags=%s", len(feature_cols), str(LAGS))
        outputs = construir_outputs(panel_build, feature_cols)
        log_outputs_summary(logger, outputs, panel_build)

    with log_stage(logger, "guardar_salidas"):
        guardar_salidas(prep_dir, outputs)
        logger.info("Archivos guardados en: %s", prep_dir.as_posix())


def main() -> None:
    """
    Punto de entrada principal del pipeline de preparación y features.

    Detecta automáticamente si se está ejecutando en entorno SageMaker
    o local, y ajusta las rutas de entrada/salida en consecuencia.
    """
    logger = get_logger(__name__, log_dir=LOG_DIR, prefijo_archivo="prep")
    start_total = time.perf_counter()

    logger.info("Iniciando pipeline de preparación (prep).")

    try:
        from pathlib import Path

        if Path("/opt/ml/processing").exists():
            raw_dir = Path("/opt/ml/processing/input")
            prep_dir = Path("/opt/ml/processing/output")
        else:
            raw_dir = RAW_DIR
            prep_dir = PREP_DIR

        run_prep_pipeline(logger, raw_dir=raw_dir, prep_dir=prep_dir)

    except Exception as exc:  # noqa: BLE001
        logger.exception("Prep falló: %s", str(exc))
        raise
    finally:
        duration_total = time.perf_counter() - start_total
        logger.info("Prep terminado. duracion_total_seg=%.2f", duration_total)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocessing y feature engineering")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--prep-dir", type=Path, default=PREP_DIR)
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
    return parser


def cli_main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    global RAW_DIR, PREP_DIR, LOG_DIR
    RAW_DIR = args.raw_dir
    PREP_DIR = args.prep_dir
    LOG_DIR = args.log_dir

    main()

if __name__ == "__main__":
    main()
