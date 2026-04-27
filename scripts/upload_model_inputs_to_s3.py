"""Upload translated Kaggle input CSVs to the ModelOps S3 bucket."""
from __future__ import annotations

import argparse

from modelops.io import copy_local_to_s3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", default="raw/current")
    parser.add_argument("--sales-path", default="data/raw/sales_train.csv")
    parser.add_argument("--test-path", default="data/raw/test.csv")
    parser.add_argument("--items-path", default="data/raw/items_en.csv")
    parser.add_argument("--categories-path", default="data/raw/item_categories_en.csv")
    parser.add_argument("--shops-path", default="data/raw/shops_en.csv")
    parser.add_argument("--sample-submission-path", default="data/raw/sample_submission.csv")
    return parser.parse_args()


def upload_if_exists(local_path: str | None, s3_uri: str, label: str) -> None:
    """Upload one optional file if its path exists."""
    if not local_path:
        return
    try:
        copy_local_to_s3(local_path, s3_uri)
        print(f"Uploaded {label}: {s3_uri}")
    except FileNotFoundError:
        print(f"Skipped {label}; file not found: {local_path}")


def main() -> None:
    args = parse_args()
    base = f"s3://{args.bucket}/{args.prefix.strip('/')}"
    upload_if_exists(args.sales_path, f"{base}/sales_train.csv", "sales")
    upload_if_exists(args.test_path, f"{base}/test.csv", "test")
    upload_if_exists(args.items_path, f"{base}/items_en.csv", "items")
    upload_if_exists(args.categories_path, f"{base}/item_categories_en.csv", "categories")
    upload_if_exists(args.shops_path, f"{base}/shops_en.csv", "shops")
    upload_if_exists(args.sample_submission_path, f"{base}/sample_submission.csv", "sample_submission")
    print(f"Done. Inputs prefix: {base}")


if __name__ == "__main__":
    main()
