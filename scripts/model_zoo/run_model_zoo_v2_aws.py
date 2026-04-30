"""Run the clean/friendly sparse-demand model zoo in SageMaker Processing.

The job reads existing feature files and raw dimensions from S3:

* modelops/features/train.parquet
* modelops/features/valid.parquet
* modelops/features/inference_features.parquet
* modelops/features/inference_pairs.parquet
* raw/current/shops_en.csv
* raw/current/items_en.csv
* raw/current/item_categories_en.csv

Outputs are written to:

* modelops/model-zoo-runs/<run_id>/
* modelops/latest/
* modelops/registry/
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import boto3
import sagemaker
from sagemaker.processing import FrameworkProcessor, ProcessingInput, ProcessingOutput
from sagemaker.sklearn.estimator import SKLearn


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the SageMaker model zoo job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--instance-type", default="ml.m5.2xlarge")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--features-prefix", default="modelops/features")
    parser.add_argument("--raw-prefix", default="raw/current")
    parser.add_argument("--runs-prefix", default="modelops/model-zoo-runs")
    parser.add_argument("--latest-prefix", default="modelops/latest")
    parser.add_argument("--registry-prefix", default="modelops/registry")
    parser.add_argument("--max-train-rows", type=int, default=500_000)
    parser.add_argument("--include-slow-models", action="store_true")
    return parser.parse_args()


def make_processor(args: argparse.Namespace) -> FrameworkProcessor:
    """Create a SageMaker FrameworkProcessor for the model zoo job."""
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
        base_job_name="pfs-clean-model-zoo",
        sagemaker_session=session,
    )


def main() -> None:
    """Launch the model zoo job in SageMaker Processing."""
    args = parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    processor = make_processor(args)

    bucket = args.bucket
    features = f"s3://{bucket}/{args.features_prefix.strip('/')}"
    raw = f"s3://{bucket}/{args.raw_prefix.strip('/')}"
    run_outputs = f"s3://{bucket}/{args.runs_prefix.strip('/')}/{run_id}"
    latest = f"s3://{bucket}/{args.latest_prefix.strip('/')}"
    registry = f"s3://{bucket}/{args.registry_prefix.strip('/')}"
    base_s3_uri = f"s3://{bucket}/modelops"

    arguments = [
        "--train-path",
        "/opt/ml/processing/features/train.parquet",
        "--valid-path",
        "/opt/ml/processing/features/valid.parquet",
        "--inference-features-path",
        "/opt/ml/processing/features/inference_features.parquet",
        "--inference-pairs-path",
        "/opt/ml/processing/features/inference_pairs.parquet",
        "--shops-path",
        "/opt/ml/processing/dimensions/shops_en.csv",
        "--items-path",
        "/opt/ml/processing/dimensions/items_en.csv",
        "--categories-path",
        "/opt/ml/processing/dimensions/item_categories_en.csv",
        "--output-dir",
        "/opt/ml/processing/output",
        "--latest-dir",
        "/opt/ml/processing/latest",
        "--registry-dir",
        "/opt/ml/processing/registry",
        "--run-id",
        run_id,
        "--base-s3-uri",
        base_s3_uri,
        "--max-train-rows",
        str(args.max_train_rows),
    ]

    if args.include_slow_models:
        arguments.append("--include-slow-models")

    print(f"run_id={run_id}")
    print(f"features={features}")
    print(f"raw_dimensions={raw}")
    print(f"run_outputs={run_outputs}")
    print(f"latest={latest}")
    print(f"registry={registry}")

    processor.run(
        code="scripts/sagemaker_entrypoints/train_model_zoo_v2.py",
        source_dir=".",
        inputs=[
            ProcessingInput(
                input_name="features",
                source=features,
                destination="/opt/ml/processing/features",
            ),
            ProcessingInput(
                input_name="raw_dimensions",
                source=f"{raw}/",
                destination="/opt/ml/processing/dimensions",
            ),
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output",
                destination=run_outputs,
            ),
            ProcessingOutput(
                source="/opt/ml/processing/latest",
                destination=latest,
            ),
            ProcessingOutput(
                source="/opt/ml/processing/registry",
                destination=registry,
            ),
        ],
        arguments=arguments,
        wait=True,
        logs=True,
    )

    print("Done.")
    print(f"RUN_ID={run_id}")


if __name__ == "__main__":
    main()