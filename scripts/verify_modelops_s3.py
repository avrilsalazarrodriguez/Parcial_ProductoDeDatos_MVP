"""Verify that Streamlit can read ModelOps S3 outputs."""

from __future__ import annotations

from backend.modelops_s3 import (
    load_evaluation_by_item,
    load_evaluation_by_segment,
    load_forecast_detail,
    load_model_metrics,
    modelops_healthcheck,
)


def main() -> None:
    print("Healthcheck:")
    print(modelops_healthcheck())

    forecast = load_forecast_detail()
    print("forecast_detail:", forecast.shape)

    segment = load_evaluation_by_segment()
    print("evaluation_by_segment:", segment.shape)

    item = load_evaluation_by_item()
    print("evaluation_by_item:", item.shape)

    metrics = load_model_metrics()
    print("model_metrics keys:", list(metrics.keys()))


if __name__ == "__main__":
    main()
