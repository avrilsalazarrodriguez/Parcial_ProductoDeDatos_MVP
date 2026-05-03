"""Run the ModelOps flow in AWS as three sequential SageMaker Processing jobs.

This is the safest path for the exam if the dashboard is being built separately.
It runs entirely in AWS once the input CSVs are in S3.

Requirements:
- A SageMaker execution role with access to the ModelOps S3 bucket.
- The repo contains `requirements.txt` at the root so the FrameworkProcessor
  installs pyarrow/scikit-learn/joblib.
"""
from __future__ import annotations

import argparse

import boto3
import sagemaker
from sagemaker.processing import FrameworkProcessor, ProcessingInput, ProcessingOutput
from sagemaker.sklearn.estimator import SKLearn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--instance-type", default="ml.m5.large")
    parser.add_argument("--base-prefix", default="modelops")
    parser.add_argument("--sales-key", default="raw/current/sales_train.csv")
    parser.add_argument("--test-key", default="raw/current/test.csv")
    parser.add_argument("--items-key", default="raw/current/items_en.csv")
    parser.add_argument("--categories-key", default="raw/current/item_categories_en.csv")
    parser.add_argument("--shops-key", default="raw/current/shops_en.csv")
    return parser.parse_args()


def make_processor(args: argparse.Namespace) -> FrameworkProcessor:
    boto_session = boto3.Session(region_name=args.region)
    session = sagemaker.Session(
        boto_session=boto_session,
        default_bucket=args.bucket,
    )
    return FrameworkProcessor(
        estimator_cls=SKLearn,
        framework_version="1.2-1",
        role=args.role_arn,
        instance_type=args.instance_type,
        instance_count=1,
        base_job_name="pfs-modelops",
        sagemaker_session=session,
    )


def main() -> None:
    args = parse_args()
    processor = make_processor(args)

    bucket = args.bucket
    base = f"s3://{bucket}/{args.base_prefix.strip('/')}"
    sales_s3 = f"s3://{bucket}/{args.sales_key}"
    test_s3 = f"s3://{bucket}/{args.test_key}"
    items_s3 = f"s3://{bucket}/{args.items_key}"
    categories_s3 = f"s3://{bucket}/{args.categories_key}"
    shops_s3 = f"s3://{bucket}/{args.shops_key}"

    features_s3 = f"{base}/features"
    model_s3 = f"{base}/model"
    evaluation_s3 = f"{base}/evaluation"
    predictions_s3 = f"{base}/predictions"

    print("1/3 Build features in SageMaker Processing")
    processor.run(
        code="scripts/sagemaker_entrypoints/build_features.py",
        source_dir=".",
        inputs=[
            ProcessingInput(source=sales_s3, destination="/opt/ml/processing/input/sales"),
            ProcessingInput(source=test_s3, destination="/opt/ml/processing/input/test"),
            ProcessingInput(source=items_s3, destination="/opt/ml/processing/input/items"),
            ProcessingInput(source=categories_s3, destination="/opt/ml/processing/input/categories"),
            ProcessingInput(source=shops_s3, destination="/opt/ml/processing/input/shops"),
        ],
        outputs=[
            ProcessingOutput(source="/opt/ml/processing/output", destination=features_s3),
        ],
        arguments=[
            "--sales-path", "/opt/ml/processing/input/sales/sales_train.csv",
            "--test-path", "/opt/ml/processing/input/test/test.csv",
            "--items-path", "/opt/ml/processing/input/items/items_en.csv",
            "--categories-path", "/opt/ml/processing/input/categories/item_categories_en.csv",
            "--shops-path", "/opt/ml/processing/input/shops/shops_en.csv",
            "--output-dir", "/opt/ml/processing/output",
        ],
        wait=True,
        logs=True,
    )

    print("2/3 Train segmented models in SageMaker Processing")
    processor.run(
        code="scripts/sagemaker_entrypoints/train_segmented.py",
        source_dir=".",
        inputs=[
            ProcessingInput(source=features_s3, destination="/opt/ml/processing/features"),
        ],
        outputs=[
            ProcessingOutput(source="/opt/ml/processing/model", destination=model_s3),
            ProcessingOutput(source="/opt/ml/processing/evaluation", destination=evaluation_s3),
        ],
        arguments=[
            "--train-path", "/opt/ml/processing/features/train.parquet",
            "--valid-path", "/opt/ml/processing/features/valid.parquet",
            "--model-output-path", "/opt/ml/processing/model/model_bundle.joblib",
            "--evaluation-output-dir", "/opt/ml/processing/evaluation",
        ],
        wait=True,
        logs=True,
    )

    print("3/3 Score batch in SageMaker Processing")
    processor.run(
        code="scripts/sagemaker_entrypoints/score_batch.py",
        source_dir=".",
        inputs=[
            ProcessingInput(source=model_s3, destination="/opt/ml/processing/model"),
            ProcessingInput(source=features_s3, destination="/opt/ml/processing/features"),
        ],
        outputs=[
            ProcessingOutput(source="/opt/ml/processing/predictions", destination=predictions_s3),
        ],
        arguments=[
            "--model-path", "/opt/ml/processing/model/model_bundle.joblib",
            "--inference-features-path", "/opt/ml/processing/features/inference_features.parquet",
            "--inference-pairs-path", "/opt/ml/processing/features/inference_pairs.parquet",
            "--output-dir", "/opt/ml/processing/predictions",
        ],
        wait=True,
        logs=True,
    )

    print("Done.")
    print(f"Features: {features_s3}")
    print(f"Model: {model_s3}")
    print(f"Evaluation: {evaluation_s3}")
    print(f"Predictions: {predictions_s3}")


if __name__ == "__main__":
    main()
