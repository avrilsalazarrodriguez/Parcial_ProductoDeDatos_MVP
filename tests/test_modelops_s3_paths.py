import os

from backend.modelops_s3 import build_key, s3_uri_for


def test_build_key_uses_modelops_prefix(monkeypatch):
    monkeypatch.setenv("MODEL_BUCKET", "demo-bucket")
    monkeypatch.setenv("MODELOPS_PREFIX", "modelops/latest")

    assert build_key("predictions/forecast_detail.parquet") == (
        "modelops/latest/predictions/forecast_detail.parquet"
    )


def test_s3_uri_for(monkeypatch):
    monkeypatch.setenv("MODEL_BUCKET", "demo-bucket")
    monkeypatch.setenv("MODELOPS_PREFIX", "modelops/latest")

    assert s3_uri_for("evaluation/model_metrics.json") == (
        "s3://demo-bucket/modelops/latest/evaluation/model_metrics.json"
    )
