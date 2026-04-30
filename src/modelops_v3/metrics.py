"""Metrics for retail demand model comparison."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def clip_sales(values: pd.Series | np.ndarray, lower: float = 0.0, upper: float = 20.0) -> np.ndarray:
    return np.clip(pd.Series(values).astype(float).fillna(0).to_numpy(), lower, upper)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum(np.abs(y_true)))
    if denom <= 0:
        return mae(y_true, y_pred)
    return float(np.sum(np.abs(y_pred - y_true)) / denom)


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred - y_true))


def nonzero_recall(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.05) -> float:
    mask = y_true > 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(y_pred[mask] > threshold))


def build_metrics(
    model_id: str,
    model_name: str,
    model_family: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_naive: np.ndarray,
) -> dict:
    """Build a normalized metric row.

    The model decision metric is MAE. WAPE is still reported because it helps
    business users understand relative error, but it is not the promotion metric.
    """
    y_true = clip_sales(y_true)
    y_pred = clip_sales(y_pred)
    y_naive = clip_sales(y_naive)
    model_mae = mae(y_true, y_pred)
    naive_mae = mae(y_true, y_naive)
    return {
        "model_id": model_id,
        "model_name": model_name,
        "model_family": model_family,
        "mae": model_mae,
        "rmse": rmse(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "wape": wape(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "nonzero_recall": nonzero_recall(y_true, y_pred),
        "naive_mae": naive_mae,
        "naive_rmse": rmse(y_true, y_naive),
        "beats_naive_mae": bool(model_mae <= naive_mae),
        "selection_metric": "mae",
    }
