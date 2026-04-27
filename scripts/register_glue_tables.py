"""
register_glue_tables.py

Registro de tablas del MVP en Amazon S3 y AWS Glue Data Catalog.

Glue sirve como catálogo de datos para que las tablas puedan consultarse
después desde Athena.

Para este MVP registramos tablas pequeñas y útiles para negocio:
- forecast_results: pronósticos del mes futuro
- problem_products: productos con errores altos en validación
- model_metadata: metadata y métricas del modelo activo
"""

from __future__ import annotations

from pathlib import Path

import boto3
import pandas as pd
import sagemaker
import awswrangler as wr


APP_DATA_DIR = Path("data/app")
DATABASE_NAME = "pfs_mvp"


def get_default_bucket() -> str:
    """Usamos el bucket default de SageMaker para no hardcodear nombres."""
    session = sagemaker.Session()
    return session.default_bucket()


def upload_table_to_glue(
    df: pd.DataFrame,
    bucket: str,
    table_name: str,
) -> None:
    """
    Sube un DataFrame a S3 en formato Parquet y lo registra en Glue.

    mode='overwrite' hace que el script sea idempotente:
    se puede correr varias veces sin duplicar datos.
    """
    s3_path = f"s3://{bucket}/pfs-mvp/glue/{table_name}/"

    wr.s3.to_parquet(
        df=df,
        path=s3_path,
        dataset=True,
        database=DATABASE_NAME,
        table=table_name,
        mode="overwrite",
    )

    print(f"{table_name} registrada en Glue: {s3_path}")


def main() -> None:
    """Crea la base de datos Glue y registra las tablas principales."""
    bucket = get_default_bucket()

    print(f"Bucket usado: {bucket}")

    wr.catalog.create_database(name=DATABASE_NAME, exist_ok=True)

    forecast_results = pd.read_csv(APP_DATA_DIR / "forecast_results.csv")
    problem_products = pd.read_csv(APP_DATA_DIR / "problem_products.csv")
    model_metadata = pd.read_csv(APP_DATA_DIR / "model_metadata.csv")

    upload_table_to_glue(forecast_results, bucket, "forecast_results")
    upload_table_to_glue(problem_products, bucket, "problem_products")
    upload_table_to_glue(model_metadata, bucket, "model_metadata")

    print("Tablas registradas correctamente en Glue.")


if __name__ == "__main__":
    main()
