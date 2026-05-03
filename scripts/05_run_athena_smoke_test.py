"""Run a small Athena smoke test against Glue-catalogued ModelOps tables."""

from __future__ import annotations

import argparse
import time

import boto3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True)
    parser.add_argument("--workgroup", default="pfs-modelops-athena")
    parser.add_argument("--query", default="SHOW TABLES")
    parser.add_argument("--region", default="us-east-1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    athena = boto3.client("athena", region_name=args.region)
    start = athena.start_query_execution(
        QueryString=args.query,
        QueryExecutionContext={"Database": args.database},
        WorkGroup=args.workgroup,
    )
    query_id = start["QueryExecutionId"]
    while True:
        result = athena.get_query_execution(QueryExecutionId=query_id)
        state = result["QueryExecution"]["Status"]["State"]
        if state in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            break
        time.sleep(2)

    print(f"query_id={query_id} state={state}")
    if state != "SUCCEEDED":
        print(result["QueryExecution"]["Status"])
        raise SystemExit(1)

    rows = athena.get_query_results(QueryExecutionId=query_id)["ResultSet"]["Rows"]
    for row in rows[:20]:
        print([cell.get("VarCharValue", "") for cell in row.get("Data", [])])


if __name__ == "__main__":
    main()
