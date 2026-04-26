"""
backend/db.py

Funciones para conectar Streamlit con Amazon RDS PostgreSQL.

La aplicación no guarda credenciales en código, sino que ee:
- endpoint desde CloudFormation
- usuario/password/dbname desde Secrets Manager

Este módulo se usa para:
- guardar feedback de negocio
- listar feedback ya capturado
- listar productos problemáticos cargados a RDS
"""

from __future__ import annotations

import json
from functools import lru_cache

import boto3
import pandas as pd
from sqlalchemy import create_engine, text


STACK_NAME = "pfs-mvp-rds"
SECRET_NAME = "itam/rds/pfs-mvp/credentials"


@lru_cache(maxsize=1)
def get_stack_outputs() -> dict[str, str]:
    """Recuperamos los outputs del stack para obtener el endpoint de RDS."""
    cf_client = boto3.client("cloudformation")
    response = cf_client.describe_stacks(StackName=STACK_NAME)
    outputs = response["Stacks"][0]["Outputs"]
    return {item["OutputKey"]: item["OutputValue"] for item in outputs}


@lru_cache(maxsize=1)
def get_db_credentials() -> dict[str, str]:
    """Leemos las credenciales desde Secrets Manager."""
    secrets_client = boto3.client("secretsmanager")
    response = secrets_client.get_secret_value(SecretId=SECRET_NAME)
    return json.loads(response["SecretString"])


@lru_cache(maxsize=1)
def get_engine():
    """Creamos un engine reusable de SQLAlchemy para conectarnos a RDS."""
    outputs = get_stack_outputs()
    creds = get_db_credentials()

    connection_url = (
        f"postgresql+psycopg2://{creds['username']}:{creds['password']}"
        f"@{outputs['RdsEndpoint']}:{creds['port']}/{creds['dbname']}"
    )

    return create_engine(connection_url)


def insert_business_feedback(
    shop_id: int,
    item_id: int,
    issue_type: str,
    comment: str,
    analyst_name: str | None = None,
) -> None:
    """Inserta una observación de negocio en RDS."""
    query = text(
        """
        INSERT INTO business_feedback
            (shop_id, item_id, issue_type, comment, analyst_name)
        VALUES
            (:shop_id, :item_id, :issue_type, :comment, :analyst_name)
        """
    )

    with get_engine().begin() as connection:
        connection.execute(
            query,
            {
                "shop_id": int(shop_id),
                "item_id": int(item_id),
                "issue_type": issue_type,
                "comment": comment,
                "analyst_name": analyst_name,
            },
        )


def read_business_feedback(limit: int = 100) -> pd.DataFrame:
    """Lee las observaciones más recientes capturadas por negocio."""
    query = f"""
        SELECT
            feedback_id,
            shop_id,
            item_id,
            issue_type,
            comment,
            analyst_name,
            created_at
        FROM business_feedback
        ORDER BY created_at DESC
        LIMIT {int(limit)}
    """
    return pd.read_sql(query, get_engine())


def read_problem_products(limit: int = 100) -> pd.DataFrame:
    """Lee productos problemáticos precargados desde validación."""
    query = f"""
        SELECT
            problem_id,
            shop_id,
            item_id,
            y_true,
            prediction,
            abs_error,
            reason,
            created_at
        FROM problem_products
        ORDER BY abs_error DESC
        LIMIT {int(limit)}
    """
    return pd.read_sql(query, get_engine())


def insert_batch_export(
    scope: str,
    shop_id: int | None,
    records_count: int,
    total_prediction: float,
    s3_uri: str | None = None,
) -> None:
    """Registra un archivo batch generado para negocio."""
    query = text(
        """
        INSERT INTO batch_exports
            (scope, shop_id, records_count, total_prediction, s3_uri, status)
        VALUES
            (:scope, :shop_id, :records_count, :total_prediction, :s3_uri, 'generated')
        """
    )

    with get_engine().begin() as connection:
        connection.execute(
            query,
            {
                "scope": scope,
                "shop_id": shop_id,
                "records_count": int(records_count),
                "total_prediction": float(total_prediction),
                "s3_uri": s3_uri,
            },
        )


def read_batch_exports(limit: int = 50) -> pd.DataFrame:
    """Lee el historial de archivos batch generados."""
    query = f"""
        SELECT
            export_id,
            scope,
            shop_id,
            records_count,
            total_prediction,
            s3_uri,
            status,
            created_at
        FROM batch_exports
        ORDER BY created_at DESC
        LIMIT {int(limit)}
    """
    return pd.read_sql(query, get_engine())
