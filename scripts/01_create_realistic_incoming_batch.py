"""Create a realistic incoming sales batch from a historical month.

This avoids claiming model improvement with synthetic data. The generated CSV is
real historical data that is held out and used only to demonstrate the new-data
path: S3 incoming -> ingestion -> retraining -> evaluation -> promotion.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sales-path", default="data/raw/sales_train.csv")
    parser.add_argument("--date-block-num", type=int, default=33)
    parser.add_argument("--output-path", default="data/incoming/sales_incremental_block_33.csv")
    parser.add_argument("--sample-frac", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sales = pd.read_csv(args.sales_path)
    incoming = sales.loc[sales["date_block_num"] == args.date_block_num].copy()
    if incoming.empty:
        raise ValueError(f"No rows found for date_block_num={args.date_block_num}")

    if args.sample_frac < 1.0:
        incoming = incoming.sample(frac=args.sample_frac, random_state=args.random_state).copy()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    incoming.to_csv(output_path, index=False)
    print(f"Wrote incoming batch: {output_path}")
    print(f"rows={len(incoming):,}")


if __name__ == "__main__":
    main()
