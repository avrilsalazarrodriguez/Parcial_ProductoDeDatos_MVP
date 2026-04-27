"""Create a realistic incoming batch from historical sales data.

Use this for demos instead of synthetic data. It splits one historical month
from sales_train.csv and lets you simulate a new production CSV arrival.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sales-path", required=True)
    parser.add_argument("--date-block-num", type=int, default=33)
    parser.add_argument("--base-output-path", required=True)
    parser.add_argument("--incoming-output-path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sales = pd.read_csv(args.sales_path)

    base = sales.loc[sales["date_block_num"] < args.date_block_num].copy()
    incoming = sales.loc[sales["date_block_num"] == args.date_block_num].copy()

    if base.empty:
        raise ValueError("Base dataset is empty. Check --date-block-num.")
    if incoming.empty:
        raise ValueError(f"No incoming rows found for date_block_num={args.date_block_num}")

    base_path = Path(args.base_output_path)
    incoming_path = Path(args.incoming_output_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    incoming_path.parent.mkdir(parents=True, exist_ok=True)

    base.to_csv(base_path, index=False)
    incoming.to_csv(incoming_path, index=False)

    print(f"Base rows: {len(base):,} -> {base_path}")
    print(f"Incoming rows: {len(incoming):,} -> {incoming_path}")


if __name__ == "__main__":
    main()
