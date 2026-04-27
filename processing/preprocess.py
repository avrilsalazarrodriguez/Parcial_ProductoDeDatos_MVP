from __future__ import annotations

import json
from pathlib import Path

from src.preprocessing.prep import (
    cargar_datos_raw,
    tipificar_y_filtrar,
    construir_panel_con_features,
    columnas_features_finales,
    construir_outputs,
    log_df_info,
    log_outputs_summary,
    log_stage,
)
from src.utils.logging_utils import get_logger

RAW_DIR = Path("/opt/ml/processing/input/raw")
OUTPUT_DIR = Path("/opt/ml/processing/output")
LOG_DIR = Path("/opt/ml/processing/logs")


def save_outputs_csv(outputs) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs.train_out.to_csv(OUTPUT_DIR / "train.csv", index=False)
    outputs.valid_out.to_csv(OUTPUT_DIR / "valid.csv", index=False)
    outputs.test_out.to_csv(OUTPUT_DIR / "test_features.csv", index=False)
    outputs.test_pairs.to_csv(OUTPUT_DIR / "test_pairs.csv", index=False)

    with (OUTPUT_DIR / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(outputs.meta, f, indent=2)

    print("Contenido final de /opt/ml/processing/output:")
    for p in OUTPUT_DIR.rglob("*"):
        print(p)


def main() -> None:
    logger = get_logger(
        "processing.preprocess",
        log_dir=LOG_DIR,
        prefijo_archivo="processing",
    )

    with log_stage(logger, "cargar_datos_raw"):
        raw = cargar_datos_raw(RAW_DIR)
        log_df_info(logger, "train_raw", raw.train)
        log_df_info(logger, "test_raw", raw.test)

    with log_stage(logger, "tipificar_y_filtrar"):
        clean = tipificar_y_filtrar(raw)
        log_df_info(logger, "train_clean", clean.train)
        log_df_info(logger, "test_clean", clean.test)

    with log_stage(logger, "construir_panel_con_features"):
        panel_build = construir_panel_con_features(clean)
        log_df_info(logger, "panel", panel_build.panel)

    with log_stage(logger, "construir_outputs"):
        feature_cols = columnas_features_finales()
        outputs = construir_outputs(panel_build, feature_cols)
        log_outputs_summary(logger, outputs, panel_build)

    with log_stage(logger, "guardar_salidas_csv"):
        save_outputs_csv(outputs)

    logger.info("Processing complete. Files written to %s", OUTPUT_DIR.as_posix())


if __name__ == "__main__":
    main()