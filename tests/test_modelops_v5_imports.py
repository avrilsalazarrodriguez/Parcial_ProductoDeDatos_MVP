from frontend.ui_enhancements import attach_catalog_labels, render_forecast_distribution
from backend.modelops_s3 import load_model_metrics, modelops_healthcheck


def test_imports_exist():
    assert callable(attach_catalog_labels)
    assert callable(render_forecast_distribution)
    assert callable(load_model_metrics)
    assert callable(modelops_healthcheck)
