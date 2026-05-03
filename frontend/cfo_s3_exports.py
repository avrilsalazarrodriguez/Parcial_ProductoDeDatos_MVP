from __future__ import annotations

"""S3 export helper for normal CFO batch files.

This module is intentionally small and isolated from the Streamlit design. It is
used only to choose a clearer S3 partition for normal CFO exports:

- shop exports -> app/batch_exports/todos_los_productos_de_una_tienda/shop_<id>/
- category exports -> app/batch_exports/segmento_categoria/category_<id>/
- full catalog -> app/batch_exports/catalogo_completo/all/

Uploaded-batch predictions are handled by frontend/batch_upload_inference.py and
remain in app/batch_uploads/predictions/ without category partitions.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import os

import pandas as pd


def _resolve_bucket() -> tuple[str | None, str]:
    for name in ["MODEL_BUCKET", "MODELOPS_BUCKET", "MODEL_OPS_BUCKET", "S3_BUCKET", "BUCKET_NAME"]:
        value = os.getenv(name)
        if value:
            return value, f"env:{name}"
    try:
        from backend import modelops_s3  # type: ignore

        for attr in ["CONFIG", "config", "get_config", "get_modelops_config"]:
            candidate = getattr(modelops_s3, attr, None)
            cfg = candidate() if callable(candidate) else candidate
            if cfg is not None and getattr(cfg, "bucket", None):
                return getattr(cfg, "bucket"), f"backend.modelops_s3.{attr}.bucket"
    except Exception:
        pass
    return None, "not_found"


def _s3_client():
    import boto3

    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
    return boto3.client("s3", region_name=region)


def _safe_value(value: Any, prefix: str) -> str:
    if value is None:
        raw = "unknown"
    else:
        try:
            raw = "unknown" if pd.isna(value) else str(value)
        except Exception:
            raw = str(value)
    clean = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in raw.strip())
    clean = clean.strip("_") or "unknown"
    return f"{prefix}_{clean}"


def _scope_folder(scope: str) -> str:
    normalized = str(scope).strip().lower()
    if "tienda" in normalized:
        return "todos_los_productos_de_una_tienda"
    if "segmento" in normalized or "categor" in normalized:
        return "segmento_categoria"
    if "catalog" in normalized or "catálogo" in normalized:
        return "catalogo_completo"
    return _safe_value(scope, "scope")


def _infer_category_id(df: pd.DataFrame, category_id: Any | None = None) -> Any | None:
    if category_id is not None and str(category_id).strip() not in {"", "None", "nan"}:
        return category_id
    for col in ["item_category_id", "segment_id", "category_id"]:
        if col in df.columns:
            values = df[col].dropna().drop_duplicates().tolist()
            if len(values) == 1:
                return values[0]
    return None


def upload_cfo_dataframe_partitioned_to_s3(
    df: pd.DataFrame,
    *,
    scope: str,
    shop_id: Any | None = None,
    category_id: Any | None = None,
    filename_prefix: str = "predictions",
) -> str:
    """Upload normal CFO export to a clear partitioned S3 path."""
    bucket, bucket_source = _resolve_bucket()
    if not bucket:
        raise RuntimeError("No encontré bucket S3. Define MODEL_BUCKET o MODELOPS_BUCKET en el entorno.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    scope_folder = _scope_folder(scope)

    if scope_folder == "todos_los_productos_de_una_tienda":
        partition = _safe_value(shop_id, "shop")
    elif scope_folder == "segmento_categoria":
        category_id = _infer_category_id(df, category_id)
        partition = _safe_value(category_id, "category")
    else:
        partition = "all"

    key = f"app/batch_exports/{scope_folder}/{partition}/{filename_prefix}_{timestamp}.csv"
    body = df.to_csv(index=False).encode("utf-8")
    _s3_client().put_object(Bucket=bucket, Key=key, Body=body, ContentType="text/csv")
    return f"s3://{bucket}/{key}"
