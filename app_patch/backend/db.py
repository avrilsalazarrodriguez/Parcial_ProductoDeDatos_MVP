"""backend/db.py

Funciones para conectar Streamlit con Amazon RDS PostgreSQL.

La aplicación no guarda credenciales en código. Lee:
- endpoint desde CloudFormation;
- usuario/password/dbname desde Secrets Manager.

Este módulo se usa para:
- guardar feedback de negocio;
- listar feedback capturado;
- listar productos problemáticos;
- registrar batch exports;
- registrar eventos de uso de la app.

Notas de operación:
- En local puedes usar DISABLE_RDS_WRITES=true para probar la UI aunque RDS
  no exista, no esté accesible desde tu red, o el stack tenga otro nombre.
- En ECS/Fargate debes configurar RDS_STACK_NAME y RDS_SECRET_NAME con los
  valores reales del despliegue.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Any

import boto3
import pandas as pd
from botocore.exceptions import BotoCoreError, ClientError
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# Valores default originales del proyecto. Se pueden sobreescribir por variables
# de entorno sin cambiar código.
STACK_NAME = os.getenv("RDS_STACK_NAME", "pfs-mvp-rds")
SECRET_NAME = os.getenv("RDS_SECRET_NAME", "itam/rds/pfs-mvp/credentials")
AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# Útil para pruebas locales: evita que la app falle si RDS no está disponible.
DISABLE_RDS_WRITES = os.getenv("DISABLE_RDS_WRITES", "false").lower() == "true"

# Si está en true, algunos errores de RDS se propagan. En local normalmente déjalo
# en false; en una prueba estricta de ECS puedes ponerlo en true.
RDS_FAIL_FAST = os.getenv("RDS_FAIL_FAST", "false").lower() == "true"


class RDSConfigurationError(RuntimeError):
    """Error de configuración de RDS/CloudFormation/Secrets Manager."""


def _empty_dataframe(columns: list[str]) -> pd.DataFrame:
    """Crea un DataFrame vacío con columnas estables para la UI."""
    return pd.DataFrame(columns=columns)


def _boto3_client(service_name: str):
    """Crea clientes boto3 usando la región configurada."""
    return boto3.client(service_name, region_name=AWS_REGION)


def _first_present(mapping: dict[str, Any], candidates: list[str]) -> Any | None:
    """Devuelve el primer valor disponible entre varias llaves candidatas."""
    for key in candidates:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def _safe_log_rds_error(action: str, exc: Exception) -> None:
    """Registra errores RDS sin exponer credenciales ni connection strings."""
    logger.warning(
        "action=%s status=warning error_type=%s error_message=%s",
        action,
        type(exc).__name__,
        str(exc)[:300],
        exc_info=True,
    )


@lru_cache(maxsize=1)
def get_stack_outputs() -> dict[str, str]:
    """Recupera outputs de CloudFormation para obtener el endpoint de RDS.

    El nombre del stack se toma de RDS_STACK_NAME. Si no existe, esta función
    lanza RDSConfigurationError con un mensaje útil.
    """
    try:
        cf_client = _boto3_client("cloudformation")
        response = cf_client.describe_stacks(StackName=STACK_NAME)
        outputs = response["Stacks"][0].get("Outputs", [])
        return {item["OutputKey"]: item["OutputValue"] for item in outputs}
    except (ClientError, BotoCoreError) as exc:
        raise RDSConfigurationError(
            f"No se pudieron leer outputs de CloudFormation para stack "
            f"'{STACK_NAME}'. Revisa RDS_STACK_NAME o usa "
            f"DISABLE_RDS_WRITES=true en pruebas locales."
        ) from exc


@lru_cache(maxsize=1)
def get_db_credentials() -> dict[str, Any]:
    """Lee credenciales desde Secrets Manager.

    El nombre del secret se toma de RDS_SECRET_NAME. El secret debe incluir,
    como mínimo, usuario, password, puerto y nombre de base.
    """
    try:
        secrets_client = _boto3_client("secretsmanager")
        response = secrets_client.get_secret_value(SecretId=SECRET_NAME)
        return json.loads(response["SecretString"])
    except (ClientError, BotoCoreError, json.JSONDecodeError) as exc:
        raise RDSConfigurationError(
            f"No se pudo leer el secret '{SECRET_NAME}'. Revisa "
            f"RDS_SECRET_NAME y permisos de Secrets Manager."
        ) from exc


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Crea un engine reusable de SQLAlchemy para conectarse a RDS.

    Soporta variaciones comunes de nombres de outputs/secret keys para evitar
    romperse si CloudFormation usa nombres ligeramente diferentes.
    """
    if DISABLE_RDS_WRITES:
        raise RDSConfigurationError(
            "RDS está desactivado por DISABLE_RDS_WRITES=true."
        )

    outputs = get_stack_outputs()
    creds = get_db_credentials()

    endpoint = _first_present(
        outputs,
        ["RdsEndpoint", "RDSEndpoint", "DBEndpoint", "DatabaseEndpoint", "Endpoint"],
    )
    if endpoint is None:
        raise RDSConfigurationError(
            "No encontré endpoint de RDS en los outputs de CloudFormation. "
            "Revisa que exista un output como RdsEndpoint/RDSEndpoint."
        )

    username = _first_present(creds, ["username", "user", "Username"])
    password = _first_present(creds, ["password", "Password"])
    port = _first_present(creds, ["port", "Port"]) or 5432
    dbname = _first_present(creds, ["dbname", "database", "db_name", "DBName"])

    if not username or not password or not dbname:
        raise RDSConfigurationError(
            "El secret de RDS no contiene username/password/dbname con nombres esperados."
        )

    connection_url = (
        f"postgresql+psycopg2://{username}:{password}@{endpoint}:{port}/{dbname}"
    )
    return create_engine(connection_url, pool_pre_ping=True)


def insert_business_feedback(
    shop_id: int,
    item_id: int,
    issue_type: str,
    comment: str,
    analyst_name: str | None = None,
) -> None:
    """Inserta una observación de negocio en RDS.

    En local, si DISABLE_RDS_WRITES=true, no escribe y solo registra un warning.
    """
    if DISABLE_RDS_WRITES:
        logger.warning("action=insert_business_feedback status=skipped reason=rds_disabled")
        return

    query = text(
        """
        INSERT INTO business_feedback
            (shop_id, item_id, issue_type, comment, analyst_name)
        VALUES
            (:shop_id, :item_id, :issue_type, :comment, :analyst_name)
        """
    )

    try:
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
    except Exception as exc:
        _safe_log_rds_error("insert_business_feedback", exc)
        raise


def read_business_feedback(limit: int = 100) -> pd.DataFrame:
    """Lee las observaciones más recientes capturadas por negocio."""
    columns = [
        "feedback_id",
        "shop_id",
        "item_id",
        "issue_type",
        "comment",
        "analyst_name",
        "created_at",
    ]
    if DISABLE_RDS_WRITES:
        return _empty_dataframe(columns)

    query = f"""
        SELECT feedback_id, shop_id, item_id, issue_type,
               comment, analyst_name, created_at
        FROM business_feedback
        ORDER BY created_at DESC
        LIMIT {int(limit)}
    """

    try:
        return pd.read_sql(query, get_engine())
    except Exception as exc:
        _safe_log_rds_error("read_business_feedback", exc)
        return _empty_dataframe(columns)


def read_problem_products(limit: int = 100) -> pd.DataFrame:
    """Lee productos problemáticos precargados desde validación."""
    columns = [
        "problem_id",
        "shop_id",
        "item_id",
        "y_true",
        "prediction",
        "abs_error",
        "reason",
        "created_at",
    ]
    if DISABLE_RDS_WRITES:
        return _empty_dataframe(columns)

    query = f"""
        SELECT problem_id, shop_id, item_id, y_true, prediction,
               abs_error, reason, created_at
        FROM problem_products
        ORDER BY abs_error DESC
        LIMIT {int(limit)}
    """

    try:
        return pd.read_sql(query, get_engine())
    except Exception as exc:
        _safe_log_rds_error("read_problem_products", exc)
        return _empty_dataframe(columns)


def insert_batch_export(
    scope: str,
    shop_id: int | None,
    records_count: int,
    total_prediction: float,
    s3_uri: str | None = None,
) -> None:
    """Registra en RDS un archivo batch generado para negocio."""
    if DISABLE_RDS_WRITES:
        logger.warning("action=insert_batch_export status=skipped reason=rds_disabled")
        return

    query = text(
        """
        INSERT INTO batch_exports
            (scope, shop_id, records_count, total_prediction, s3_uri, status)
        VALUES
            (:scope, :shop_id, :records_count, :total_prediction, :s3_uri, 'generated')
        """
    )

    try:
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
    except Exception as exc:
        _safe_log_rds_error("insert_batch_export", exc)
        if RDS_FAIL_FAST:
            raise


def read_batch_exports(limit: int = 50) -> pd.DataFrame:
    """Lee el historial de archivos batch generados."""
    columns = [
        "export_id",
        "scope",
        "shop_id",
        "records_count",
        "total_prediction",
        "s3_uri",
        "status",
        "created_at",
    ]
    if DISABLE_RDS_WRITES:
        return _empty_dataframe(columns)

    query = f"""
        SELECT export_id, scope, shop_id, records_count,
               total_prediction, s3_uri, status, created_at
        FROM batch_exports
        ORDER BY created_at DESC
        LIMIT {int(limit)}
    """

    try:
        return pd.read_sql(query, get_engine())
    except Exception as exc:
        _safe_log_rds_error("read_batch_exports", exc)
        return _empty_dataframe(columns)


def insert_usage_event(
    event_type: str,
    shop_id: int | None = None,
    item_id: int | None = None,
    records_count: int | None = None,
    status: str = "success",
    message: str | None = None,
) -> None:
    """Registra eventos de uso de la app para monitoreo operacional.

    Esta función nunca debe tirar la app. Si RDS no existe localmente, si está
    apagado o si las credenciales fallan, solo registra warning y continúa.
    """
    if DISABLE_RDS_WRITES:
        logger.info(
            "action=insert_usage_event status=skipped reason=rds_disabled event_type=%s",
            event_type,
        )
        return

    query = text(
        """
        INSERT INTO app_usage_events
            (event_type, shop_id, item_id, records_count, status, message)
        VALUES
            (:event_type, :shop_id, :item_id, :records_count, :status, :message)
        """
    )

    try:
        with get_engine().begin() as connection:
            connection.execute(
                query,
                {
                    "event_type": event_type,
                    "shop_id": shop_id,
                    "item_id": item_id,
                    "records_count": records_count,
                    "status": status,
                    "message": message,
                },
            )
    except Exception as exc:
        _safe_log_rds_error("insert_usage_event", exc)


def read_usage_events(limit: int = 100) -> pd.DataFrame:
    """Lee eventos recientes de uso de la app."""
    columns = [
        "event_id",
        "event_type",
        "shop_id",
        "item_id",
        "records_count",
        "status",
        "message",
        "created_at",
    ]
    if DISABLE_RDS_WRITES:
        return _empty_dataframe(columns)

    query = f"""
        SELECT event_id, event_type, shop_id, item_id, records_count,
               status, message, created_at
        FROM app_usage_events
        ORDER BY created_at DESC
        LIMIT {int(limit)}
    """

    try:
        return pd.read_sql(query, get_engine())
    except Exception as exc:
        _safe_log_rds_error("read_usage_events", exc)
        return _empty_dataframe(columns)
