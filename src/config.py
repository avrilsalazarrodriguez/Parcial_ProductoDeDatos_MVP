"""
Configuración del proyecto.

Centraliza rutas, constantes y parámetros del modelo para evitar duplicación
entre scripts.
"""

from __future__ import annotations

from pathlib import Path

# Rutas y directorios
REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PREP_DIR = DATA_DIR / "prep"
INFERENCE_DIR = DATA_DIR / "inference"
PREDICTIONS_DIR = DATA_DIR / "predictions"

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
LOG_DIR = ARTIFACTS_DIR / "logs"

# Reproducibilidad
SEMILLA = 42

# Límites / clipping para el target y otras variables
TARGET_MIN = 0.0
TARGET_MAX = 20.0

PRECIO_MAX = 100000.0
CANTIDAD_DIA_MIN = -1000.0
CANTIDAD_DIA_MAX = 1000.0

CUANTIL_PRECIO = 0.999
RECENCY_FILL = 99

LAGS = (1, 2, 3, 4, 5, 6, 12)

# Hiperparámetros LightGBM para clasificacion y regresion
LGBM_CLASSIFIER_PARAMS = {
    "n_estimators": 6000,
    "learning_rate": 0.03,
    "num_leaves": 256,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": SEMILLA,
    "verbosity": -1,
}

LGBM_REGRESSOR_PARAMS = {
    "n_estimators": 8000,
    "learning_rate": 0.03,
    "num_leaves": 256,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": SEMILLA,
    "objective": "regression",
    "verbosity": -1,
}
