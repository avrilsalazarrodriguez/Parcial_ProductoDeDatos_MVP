"""
backend/storage.py

Funciones para guardar archivos generados por la app en Amazon S3.

En el MVP usamos S3 como ubicaciónpara los archivos batch
que pediría Finanzas. La app genera el CSV, lo sube a S3 y conserva el URI.
"""

from __future__ import annotations

from io import StringIO
from time import gmtime, strftime

import boto3
import pandas as pd


def get_default_bucket() -> str:
    """Obtenemos el bucket default de SageMaker/AWS usado en el proyecto."""
    account_id = boto3.client("sts").get_caller_identity()["Account"]
    region = boto3.session.Session().region_name or "us-east-1"
    return f"sagemaker-{region}-{account_id}"


def upload_batch_dataframe_to_s3(
    df: pd.DataFrame,
    scope: str,
    shop_id: int | None = None,
) -> str:
    """
    Sube un DataFrame batch como CSV a S3 y regresa su S3 URI.

    scope indica si el archivo corresponde a catálogo completo,
    tienda específica o archivo cargado por usuario.
    """
    bucket = get_default_bucket()
    timestamp = strftime("%Y%m%d-%H%M%S", gmtime())

    safe_scope = scope.lower().replace(" ", "_")
    shop_part = f"shop_{shop_id}" if shop_id is not None else "all"

    key = f"pfs-mvp/batch_exports/{safe_scope}/{shop_part}/predictions_{timestamp}.csv"

    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)

    s3_client = boto3.client("s3")
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=csv_buffer.getvalue().encode("utf-8"),
        ContentType="text/csv",
    )

    return f"s3://{bucket}/{key}"
