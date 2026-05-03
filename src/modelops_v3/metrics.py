"""Metrics for model comparison.

Selection policy in this version:
1. primary metric = RMSE
2. tie-breaker = MAE
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class ModelRunResult:
    model_id: str
    model_name: str
    model_family: str
    rmse: float
    mae: float
    smape: float
    wape: float
    bias: float
    nonzero_recall: float
    n_valid: int
    status: str = "challenger"
    is_champion: bool = False
    model_scope: str = "global"
    model_description: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def clip_target(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.clip(pd.to_numeric(pd.Series(values), errors="coerce").fillna(0).values, 0, 20)


def compute_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    y_true_arr = clip_target(y_true)
    y_pred_arr = clip_target(y_pred)
    error = y_pred_arr - y_true_arr
    abs_error = np.abs(error)
    denom = np.abs(y_true_arr) + np.abs(y_pred_arr)
    smape = np.where(denom > 0, 2 * abs_error / denom, 0.0)
    total_true = float(np.sum(np.abs(y_true_arr)))
    return {
        "rmse": float(math.sqrt(np.mean(error**2))),
        "mae": float(np.mean(abs_error)),
        "smape": float(np.mean(smape)),
        "wape": float(np.sum(abs_error) / total_true) if total_true > 0 else float(np.mean(abs_error)),
        "bias": float(np.mean(error)),
        "nonzero_recall": float(np.mean(y_pred_arr[y_true_arr > 0] > 0.05)) if np.any(y_true_arr > 0) else 0.0,
    }


def select_champion(results: list[ModelRunResult], incumbent_id: str | None = None, min_improvement: float = 0.0) -> list[ModelRunResult]:
    if not results:
        return results
    for r in results:
        r.is_champion = False
        r.status = "challenger"
    sorted_results = sorted(results, key=lambda r: (r.rmse, r.mae))
    best = sorted_results[0]
    if incumbent_id:
        incumbent = next((r for r in results if r.model_id == incumbent_id), None)
        if incumbent is not None and best.model_id != incumbent.model_id:
            threshold = incumbent.rmse * (1 - min_improvement)
            if best.rmse >= threshold:
                best = incumbent
    best.is_champion = True
    best.status = "champion"
    return results


def results_df(results: list[ModelRunResult]) -> pd.DataFrame:
    df = pd.DataFrame([r.as_dict() for r in results])
    if df.empty:
        return df
    return df.sort_values(["is_champion", "rmse", "mae"], ascending=[False, True, True]).reset_index(drop=True)
