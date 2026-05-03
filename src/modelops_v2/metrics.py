"""Métricas robustas para forecasting de ventas con muchos ceros.

El objetivo de este módulo es centralizar las métricas que se usan para elegir
el champion del catálogo de modelos. Se usan métricas complementarias porque el
problema tiene demanda intermitente: muchos pares tienda-producto venden cero y
unos pocos concentran el volumen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


EPSILON = 1e-9


@dataclass(frozen=True)
class ForecastMetrics:
    """Métricas de evaluación para un modelo de pronóstico."""

    rmse: float
    mae: float
    wape: float
    smape: float
    bias: float
    nonzero_recall: float
    nonzero_precision: float
    pred_mean: float
    true_mean: float
    n: int
    beats_naive_mae: bool | None = None
    beats_naive_wape: bool | None = None

    def to_dict(self) -> dict[str, float | int | bool | None]:
        """Convierte las métricas a diccionario serializable."""
        return asdict(self)


def _as_float_array(values: pd.Series | np.ndarray | list[float]) -> np.ndarray:
    """Convierte una serie/lista a arreglo float sin NaN."""
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def clip_sales_prediction(values: pd.Series | np.ndarray, lower: float = 0.0, upper: float = 20.0) -> np.ndarray:
    """Limita predicciones al rango operacional usado por el dataset de 1C."""
    arr = _as_float_array(values)
    return np.clip(arr, lower, upper)


def rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Root mean squared error."""
    true = _as_float_array(y_true)
    pred = _as_float_array(y_pred)
    return float(math.sqrt(np.mean((true - pred) ** 2)))


def mae(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Mean absolute error."""
    true = _as_float_array(y_true)
    pred = _as_float_array(y_pred)
    return float(np.mean(np.abs(true - pred)))


def wape(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Weighted absolute percentage error.

    WAPE = sum(abs(y - yhat)) / sum(abs(y)). Si el denominador es cero, regresa
    0 cuando el error también es cero; de lo contrario regresa un valor grande.
    """
    true = _as_float_array(y_true)
    pred = _as_float_array(y_pred)
    denominator = float(np.sum(np.abs(true)))
    numerator = float(np.sum(np.abs(true - pred)))
    if denominator <= EPSILON:
        return 0.0 if numerator <= EPSILON else float("inf")
    return numerator / denominator


def smape(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Symmetric mean absolute percentage error."""
    true = _as_float_array(y_true)
    pred = _as_float_array(y_pred)
    denominator = (np.abs(true) + np.abs(pred)) / 2.0
    valid = denominator > EPSILON
    if not np.any(valid):
        return 0.0
    return float(np.mean(np.abs(true[valid] - pred[valid]) / denominator[valid]))


def bias(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Sesgo medio: positivo implica sobrepronóstico."""
    true = _as_float_array(y_true)
    pred = _as_float_array(y_pred)
    return float(np.mean(pred - true))


def nonzero_recall(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray, threshold: float = 0.1) -> float:
    """Recall de casos con venta real positiva."""
    true = _as_float_array(y_true)
    pred = _as_float_array(y_pred)
    actual_positive = true > 0
    if not np.any(actual_positive):
        return 0.0
    predicted_positive = pred >= threshold
    return float(np.mean(predicted_positive[actual_positive]))


def nonzero_precision(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray, threshold: float = 0.1) -> float:
    """Precisión de casos pronosticados como venta positiva."""
    true = _as_float_array(y_true)
    pred = _as_float_array(y_pred)
    predicted_positive = pred >= threshold
    if not np.any(predicted_positive):
        return 0.0
    actual_positive = true > 0
    return float(np.mean(actual_positive[predicted_positive]))


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    naive_pred: pd.Series | np.ndarray | None = None,
) -> ForecastMetrics:
    """Calcula métricas de un modelo y, opcionalmente, comparación contra naive."""
    true = _as_float_array(y_true)
    pred = clip_sales_prediction(y_pred)

    model_metrics = ForecastMetrics(
        rmse=rmse(true, pred),
        mae=mae(true, pred),
        wape=wape(true, pred),
        smape=smape(true, pred),
        bias=bias(true, pred),
        nonzero_recall=nonzero_recall(true, pred),
        nonzero_precision=nonzero_precision(true, pred),
        pred_mean=float(np.mean(pred)),
        true_mean=float(np.mean(true)),
        n=int(len(true)),
    )

    if naive_pred is None:
        return model_metrics

    naive = clip_sales_prediction(naive_pred)
    naive_mae = mae(true, naive)
    naive_wape = wape(true, naive)

    return ForecastMetrics(
        **{
            **model_metrics.to_dict(),
            "beats_naive_mae": bool(model_metrics.mae < naive_mae),
            "beats_naive_wape": bool(model_metrics.wape < naive_wape),
        }
    )
