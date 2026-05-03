"""Initialize the RDS PostgreSQL schema for the Streamlit MVP.

Reads the RDS endpoint from CloudFormation and credentials from Secrets Manager.
Then executes sql/01_schema_app.sql.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import boto3
import psycopg2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-name", default="pfs-mvp-rds")
    parser.add_argument("--secret-name", default="itam/rds/pfs-mvp/credentials")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--sql-path", default="sql/01_schema_app.sql")
    return parser.parse_args()


def get_stack_outputs(stack_name: str, region: str) -> dict[str, str]:
    response = boto3.client("cloudformation", region_name=region).describe_stacks(
        StackName=stack_name
    )
    outputs = response["Stacks"][0].get("Outputs", [])
    return {item["OutputKey"]: item["OutputValue"] for item in outputs}


def get_secret(secret_name: str, region: str) -> dict[str, str]:
    response = boto3.client("secretsmanager", region_name=region).get_secret_value(
        SecretId=secret_name
    )
    return json.loads(response["SecretString"])


def main() -> None:
    args = parse_args()
    outputs = get_stack_outputs(args.stack_name, args.region)
    creds = get_secret(args.secret_name, args.region)
    sql = Path(args.sql_path).read_text()

    conn = psycopg2.connect(
        host=outputs["RdsEndpoint"],
        port=int(creds.get("port", 5432)),
        dbname=creds["dbname"],
        user=creds["username"],
        password=creds["password"],
        connect_timeout=20,
    )
    conn.autocommit = True
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql)
    finally:
        conn.close()

    print("RDS schema initialized successfully.")
    print(f"Endpoint: {outputs['RdsEndpoint']}")
    print(f"Database: {creds['dbname']}")


if __name__ == "__main__":
    main()
