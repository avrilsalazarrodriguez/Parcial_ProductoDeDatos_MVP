"""Run the complete model flow locally.

Run this before moving to AWS. It uses real Kaggle files and can incorporate
translated metadata: items_en.csv, item_categories_en.csv, shops_en.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from modelops.feature_builder import build_features
from modelops.score_batch import score_batch
from modelops.train_segmented import train_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sales-path", default="data/raw/sales_train.csv")
    parser.add_argument("--test-path", default="data/raw/test.csv")
    parser.add_argument("--items-path", default="data/raw/items_en.csv")
    parser.add_argument("--categories-path", default="data/raw/item_categories_en.csv")
    parser.add_argument("--shops-path", default="data/raw/shops_en.csv")
    parser.add_argument("--work-dir", default="data/modelops")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir)
    features_dir = work_dir / "features"
    model_dir = work_dir / "model"
    evaluation_dir = work_dir / "evaluation"
    predictions_dir = work_dir / "predictions"

    print("1/3 Building features...")
    build_features(
        sales_path=args.sales_path,
        test_path=args.test_path,
        items_path=args.items_path,
        categories_path=args.categories_path,
        shops_path=args.shops_path,
        output_dir=str(features_dir),
    )

    print("2/3 Training/evaluating models...")
    train_models(
        train_path=str(features_dir / "train.parquet"),
        valid_path=str(features_dir / "valid.parquet"),
        model_output_path=str(model_dir / "model_bundle.joblib"),
        evaluation_output_dir=str(evaluation_dir),
    )

    print("3/3 Scoring inference batch...")
    score_batch(
        model_path=str(model_dir / "model_bundle.joblib"),
        inference_features_path=str(features_dir / "inference_features.parquet"),
        inference_pairs_path=str(features_dir / "inference_pairs.parquet"),
        output_dir=str(predictions_dir),
    )

    print(f"Done. Outputs written to {work_dir}")


if __name__ == "__main__":
    main()
