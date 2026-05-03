"""
Utilidades de métricas.
"""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula la raíz del error cuadrático medio (RMSE).

    Parámetros
    ----------
    y_true:
        Valores reales.
    y_pred:
        Valores predichos.

    Regresa
    -------
    float
        Valor de RMSE.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))
