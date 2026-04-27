"""Append a newly arrived sales CSV to the current sales_train.csv.

This supports a simple manual ingestion demo:
    historical_base_sales.csv + incoming_batch.csv -> updated_sales_train.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

EXPECTED_COLUMNS = [
    "date",
    "date_block_num",
    "shop_id",
    "item_id",
    "item_price",
    "item_cnt_day",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-sales-path", required=True)
    parser.add_argument("--incoming-sales-path", required=True)
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


def validate_columns(df: pd.DataFrame, name: str) -> None:
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def main() -> None:
    args = parse_args()
    base = pd.read_csv(args.base_sales_path)
    incoming = pd.read_csv(args.incoming_sales_path)

    validate_columns(base, "base_sales")
    validate_columns(incoming, "incoming_sales")

    combined = pd.concat([base[EXPECTED_COLUMNS], incoming[EXPECTED_COLUMNS]], ignore_index=True)
    combined = combined.drop_duplicates()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    print(f"Base rows: {len(base):,}")
    print(f"Incoming rows: {len(incoming):,}")
    print(f"Combined rows: {len(combined):,}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
