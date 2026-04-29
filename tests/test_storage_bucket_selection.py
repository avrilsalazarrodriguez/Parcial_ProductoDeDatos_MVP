from backend.storage import get_export_bucket, get_export_prefix


def test_export_bucket_prefers_app_export_bucket(monkeypatch):
    monkeypatch.setenv("APP_EXPORT_BUCKET", "exports-bucket")
    monkeypatch.setenv("MODEL_BUCKET", "model-bucket")
    assert get_export_bucket() == "exports-bucket"


def test_export_bucket_falls_back_to_model_bucket(monkeypatch):
    monkeypatch.delenv("APP_EXPORT_BUCKET", raising=False)
    monkeypatch.setenv("MODEL_BUCKET", "model-bucket")
    assert get_export_bucket() == "model-bucket"


def test_export_prefix_default(monkeypatch):
    monkeypatch.delenv("APP_EXPORT_PREFIX", raising=False)
    assert get_export_prefix() == "app/batch_exports"
