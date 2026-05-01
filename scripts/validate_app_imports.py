"""Validate the clean app files before running Streamlit.

This script does not import `frontend.app` because importing a Streamlit app
executes the page. It validates backend loaders and parses app.py for syntax.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_MODELOPS_S3", "false")
os.environ.setdefault("MODELOPS_LOCAL_ROOT", "modelops_outputs")
os.environ.setdefault("DISABLE_RDS_WRITES", "true")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import modelops_s3  # noqa: E402


REQUIRED_LOADERS = [
    "load_forecast_detail",
    "load_forecast_summary_by_category",
    "load_forecast_summary_by_shop_segment",
    "load_evaluation_by_segment",
    "load_evaluation_by_item",
    "load_evaluation_detail",
    "load_evaluation_curves_by_model",
    "load_model_metrics",
    "load_champion",
    "load_model_runs",
    "load_review_suggestions",
    "load_overestimated_products",
    "load_underestimated_products",
    "modelops_healthcheck",
]


def main() -> None:
    app_path = Path("frontend/app.py")
    ast.parse(app_path.read_text(encoding="utf-8"))
    print("frontend/app.py syntax OK")

    missing = [name for name in REQUIRED_LOADERS if not hasattr(modelops_s3, name)]
    if missing:
        raise RuntimeError(f"Missing loaders in backend.modelops_s3: {missing}")
    print("backend.modelops_s3 loaders OK")

    # Data-dependent checks are optional: print status but do not fail when local
    # outputs are not generated yet.
    for loader_name in ["load_champion", "load_model_runs", "load_model_metrics", "load_evaluation_curves_by_model"]:
        loader = getattr(modelops_s3, loader_name)
        try:
            value = loader()
            shape = getattr(value, "shape", None)
            print(f"{loader_name}: OK", f"shape={shape}" if shape is not None else "")
        except Exception as exc:  # noqa: BLE001
            print(f"{loader_name}: optional data not available yet -> {type(exc).__name__}: {exc}")

    print("imports OK")


if __name__ == "__main__":
    main()
