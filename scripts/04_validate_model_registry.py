"""Validate model registry files and print a compact model catalog."""

from __future__ import annotations

import argparse
import json

import boto3
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    s3 = boto3.client("s3")
    champion_obj = s3.get_object(Bucket=args.bucket, Key="modelops/registry/champion.json")
    champion = json.loads(champion_obj["Body"].read().decode("utf-8"))
    runs_obj = s3.get_object(Bucket=args.bucket, Key="modelops/registry/model_runs.csv")
    runs = pd.read_csv(runs_obj["Body"])
    print("Champion:")
    print(json.dumps(champion, indent=2, default=str))
    print("\nModel runs:")
    print(runs.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
