"""
load_rds_tables.py

Carga las tablas operacionales del MVP en Amazon RDS PostgreSQL.

Este script usa:
- Secrets Manager para leer usuario, password, puerto y nombre de base.
- CloudFormation Outputs para leer el endpoint de RDS.
- SQLAlchemy para conectarse a PostgreSQL.
- pandas para cargar CSVs generados previamente en data/app/.

No se guardan credenciales en código.
"""

from __future__ import annotations

import json
from pathlib import Path

import boto3
import pandas as pd
from sqlalchemy import create_engine, text


STACK_NAME = "pfs-mvp-rds"
SECRET_NAME = "itam/rds/pfs-mvp/credentials"
APP_DATA_DIR = Path("data/app")
SCHEMA_PATH = Path("infra/sql/01_schema.sql")


def get_stack_outputs(stack_name: str) -> dict[str, str]:
    """Leemos los outputs de CloudFormation para recuperar el endpoint de RDS."""
    cf_client = boto3.client("cloudformation")
    response = cf_client.describe_stacks(StackName=stack_name)
    outputs = response["Stacks"][0]["Outputs"]
    return {item["OutputKey"]: item["OutputValue"] for item in outputs}


def get_db_credentials(secret_name: str) -> dict[str, str]:
    """Leemos credenciales desde Secrets Manager para no hardcodear passwords."""
    secrets_client = boto3.client("secretsmanager")
    response = secrets_client.get_secret_value(SecretId=secret_name)
    return json.loads(response["SecretString"])


def build_engine():
    """Construimos el engine de SQLAlchemy para conectarnos a PostgreSQL."""
    outputs = get_stack_outputs(STACK_NAME)
    creds = get_db_credentials(SECRET_NAME)

    endpoint = outputs["RdsEndpoint"]
    username = creds["username"]
    password = creds["password"]
    dbname = creds["dbname"]
    port = creds["port"]

    connection_url = (
        f"postgresql+psycopg2://{username}:{password}"
        f"@{endpoint}:{port}/{dbname}"
    )

    return create_engine(connection_url)


def run_schema(engine) -> None:
    """Ejecutamos el schema SQL que crea las tablas del MVP."""
    schema_sql = SCHEMA_PATH.read_text()

    with engine.begin() as connection:
        for statement in schema_sql.split(";"):
            clean_statement = statement.strip()
            if clean_statement:
                connection.execute(text(clean_statement))


def load_csv_table(engine, csv_path: Path, table_name: str) -> None:
    """Cargamos un CSV en una tabla de RDS usando pandas."""
    df = pd.read_csv(csv_path)
    df.to_sql(table_name, engine, if_exists="append", index=False)
    print(f"{table_name}: {len(df):,} filas cargadas")


def main() -> None:
    """Crea schema y carga tablas operacionales en RDS."""
    engine = build_engine()

    print("Conectando a RDS...")
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    print("Conexión a RDS exitosa.")

    print("Creando tablas...")
    run_schema(engine)

    print("Cargando datos...")
    load_csv_table(engine, APP_DATA_DIR / "forecast_results.csv", "forecast_results")
    load_csv_table(engine, APP_DATA_DIR / "problem_products.csv", "problem_products")
    load_csv_table(engine, APP_DATA_DIR / "model_metadata.csv", "model_metadata")

    print("Carga completa.")


if __name__ == "__main__":
    main()
