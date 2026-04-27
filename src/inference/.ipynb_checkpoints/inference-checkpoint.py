"""
inference.py

Ejecuta inferencia en tiempo real con el modelo entrenado.

Entradas
--------
- /opt/ml/model/model.joblib

Salida
------
- Predicciones vía endpoint HTTP de SageMaker:
  - GET /ping
  - POST /invocations
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, Response, request

from src.config import TARGET_MAX, TARGET_MIN


MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
MODEL_PATH = MODEL_DIR / "model.joblib"

app = Flask(__name__)

_MODEL_PAYLOAD: dict[str, Any] | None = None


def get_service_logger() -> logging.Logger:
    """
    Crea/configura logger para serving en SageMaker.
    """
    logger = logging.getLogger(__name__)

    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    return logger


def require_file(path: Path) -> None:
    """
    Verifica que exista un archivo requerido.
    """
    if not path.exists():
        raise FileNotFoundError(f"Archivo requerido no encontrado: {path.as_posix()}")


def cargar_modelo(model_path: Path) -> dict[str, Any]:
    """
    Carga el modelo entrenado desde /opt/ml/model/model.joblib.
    """
    require_file(model_path)
    payload: dict[str, Any] = joblib.load(model_path)
    if "bundle" not in payload:
        raise ValueError("Archivo de modelo inválido: se esperaba la llave 'bundle'.")
    return payload


def obtener_modelo() -> dict[str, Any]:
    """
    Carga el modelo una sola vez y lo mantiene en memoria.
    """
    global _MODEL_PAYLOAD

    if _MODEL_PAYLOAD is None:
        logger = get_service_logger()
        logger.info("Cargando modelo desde %s", MODEL_PATH.as_posix())
        _MODEL_PAYLOAD = cargar_modelo(MODEL_PATH)
        logger.info("Modelo cargado correctamente.")

    return _MODEL_PAYLOAD


def predecir(payload: dict[str, Any], test_features: pd.DataFrame) -> np.ndarray:
    """
    Genera predicciones usando el modelo en dos etapas.
    """
    bundle = payload["bundle"]
    feature_cols = bundle["feature_cols"]

    missing = [col for col in feature_cols if col not in test_features.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas para inferencia: {missing}")

    x_test = test_features[feature_cols]
    prob = bundle["clf"].predict_proba(x_test)[:, 1].astype(np.float32)
    mu = bundle["reg"].predict(x_test).astype(np.float32)

    return np.clip(prob * mu, TARGET_MIN, TARGET_MAX)


def _log_basic_df_info(logger: logging.Logger, name: str, df: pd.DataFrame) -> None:
    """
    Loggea información básica de un DataFrame (shape + NaNs) sin imprimir datos.
    """
    logger.info("%s: filas=%d cols=%d", name, len(df), df.shape[1])
    na_total = int(df.isna().sum().sum())
    if na_total > 0:
        logger.warning("%s: total_NA=%d", name, na_total)


def cargar_request_as_dataframe(content_type: str, body: bytes) -> pd.DataFrame:
    """
    Convierte el body del request a DataFrame.

    Content types soportados
    ------------------------
    - application/json
    - text/csv
    """
    if "application/json" in content_type:
        payload = json.loads(body.decode("utf-8"))

        if isinstance(payload, dict):
            if "instances" in payload:
                return pd.DataFrame(payload["instances"])
            if "inputs" in payload:
                return pd.DataFrame(payload["inputs"])
            return pd.DataFrame([payload])

        if isinstance(payload, list):
            return pd.DataFrame(payload)

        raise ValueError("JSON inválido para inferencia.")

    if "text/csv" in content_type:
        return pd.read_csv(io.StringIO(body.decode("utf-8")))

    raise ValueError(
        "Content-Type no soportado. Usa application/json o text/csv."
    )


@app.get("/ping")
def ping() -> Response:
    """
    Endpoint de health check requerido por SageMaker.
    """
    logger = get_service_logger()

    try:
        obtener_modelo()
        logger.info("Health check OK.")
        return Response(response="OK", status=200, mimetype="text/plain")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fallo en /ping: %s", str(exc))
        return Response(response=str(exc), status=500, mimetype="text/plain")


@app.post("/invocations")
def invocations() -> Response:
    """
    Endpoint de inferencia requerido por SageMaker.
    """
    logger = get_service_logger()
    start = time.perf_counter()
    logger.info("Iniciando inferencia en tiempo real.")

    try:
        content_type = request.content_type or ""
        df = cargar_request_as_dataframe(content_type, request.data)
        _log_basic_df_info(logger, "request_df", df)

        payload = obtener_modelo()
        preds = predecir(payload, df)

        logger.info(
            "Preds: n=%d min=%.4f p50=%.4f max=%.4f",
            len(preds),
            float(np.min(preds)),
            float(np.median(preds)),
            float(np.max(preds)),
        )

        response_body = json.dumps({"predictions": preds.tolist()})
        return Response(response=response_body, status=200, mimetype="application/json")

    except FileNotFoundError as exc:
        logger.exception("Archivo requerido no encontrado: %s", str(exc))
        return Response(
            response=json.dumps({"error": str(exc)}),
            status=500,
            mimetype="application/json",
        )
    except (ValueError, KeyError) as exc:
        logger.exception("Error de validación/estructura de datos: %s", str(exc))
        return Response(
            response=json.dumps({"error": str(exc)}),
            status=400,
            mimetype="application/json",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error inesperado en inferencia: %s", str(exc))
        return Response(
            response=json.dumps({"error": str(exc)}),
            status=500,
            mimetype="application/json",
        )
    finally:
        duration = time.perf_counter() - start
        logger.info("Inferencia terminada. duracion_seg=%.2f", duration)
