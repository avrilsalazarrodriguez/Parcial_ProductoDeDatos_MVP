from __future__ import annotations

import numpy as np

from src.modelops_v3.metrics import build_metrics


def test_mae_is_selection_metric_and_beats_naive():
    y_true = np.array([0.0, 1.0, 2.0])
    y_pred = np.array([0.0, 1.1, 2.2])
    y_naive = np.array([0.0, 0.0, 0.0])
    row = build_metrics("m", "model", "family", y_true, y_pred, y_naive)
    assert row["selection_metric"] == "mae"
    assert row["mae"] < row["naive_mae"]
    assert row["beats_naive_mae"] is True
