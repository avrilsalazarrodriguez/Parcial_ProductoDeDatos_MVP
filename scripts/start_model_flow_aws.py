"""Create or update and start a SageMaker Pipeline for the model flow.

This uses managed SageMaker Processing jobs with the sklearn processing image.
It is intentionally simpler than BYOC so it can be added quickly to the MVP
without breaking the existing dashboard branch.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import boto3
import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--pipeline-name", default="pfs-segmented-modelops")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--instance-type", default="ml.m5.large")
    parser.add_argument("--base-prefix", default="modelops")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session = sagemaker.Session(boto_session=boto3.Session(region_name=args.region))
    code_dir = Path("modelops").resolve()

    sales_s3_uri = ParameterString("SalesS3Uri", default_value=f"s3://{args.bucket}/raw/current/sales_train.csv")
    test_s3_uri = ParameterString("TestS3Uri", default_value=f"s3://{args.bucket}/raw/current/test.csv")
    items_s3_uri = ParameterString("ItemsS3Uri", default_value="")

    base_s3 = f"s3://{args.bucket}/{args.base_prefix}"
    features_s3 = f"{base_s3}/features"
    model_s3 = f"{base_s3}/model"
    evaluation_s3 = f"{base_s3}/evaluation"
    predictions_s3 = f"{base_s3}/predictions"

    processor = ScriptProcessor(
        image_uri=sagemaker.image_uris.retrieve("sklearn", args.region, version="1.2-1"),
        command=["python3"],
        role=args.role_arn,
        instance_type=args.instance_type,
        instance_count=1,
        base_job_name="pfs-modelops",
        sagemaker_session=session,
    )

    feature_step = ProcessingStep(
        name="BuildFeatures",
        processor=processor,
        code="modelops/feature_builder.py",
        inputs=[
            ProcessingInput(source=sales_s3_uri, destination="/opt/ml/processing/input/sales"),
            ProcessingInput(source=test_s3_uri, destination="/opt/ml/processing/input/test"),
        ],
        outputs=[
            ProcessingOutput(output_name="features", source="/opt/ml/processing/output", destination=features_s3),
        ],
        job_arguments=[
            "--sales-path", "/opt/ml/processing/input/sales/sales_train.csv",
            "--test-path", "/opt/ml/processing/input/test/test.csv",
            "--output-dir", "/opt/ml/processing/output",
        ],
    )

    train_step = ProcessingStep(
        name="TrainSegmentedModels",
        processor=processor,
        code="modelops/train_segmented.py",
        inputs=[
            ProcessingInput(source=features_s3, destination="/opt/ml/processing/features"),
        ],
        outputs=[
            ProcessingOutput(output_name="model", source="/opt/ml/processing/model", destination=model_s3),
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation", destination=evaluation_s3),
        ],
        job_arguments=[
            "--train-path", "/opt/ml/processing/features/train.parquet",
            "--valid-path", "/opt/ml/processing/features/valid.parquet",
            "--model-output-path", "/opt/ml/processing/model/model_bundle.joblib",
            "--evaluation-output-dir", "/opt/ml/processing/evaluation",
        ],
        depends_on=[feature_step],
    )

    score_step = ProcessingStep(
        name="ScoreBatch",
        processor=processor,
        code="modelops/score_batch.py",
        inputs=[
            ProcessingInput(source=model_s3, destination="/opt/ml/processing/model"),
            ProcessingInput(source=features_s3, destination="/opt/ml/processing/features"),
        ],
        outputs=[
            ProcessingOutput(output_name="predictions", source="/opt/ml/processing/predictions", destination=predictions_s3),
        ],
        job_arguments=[
            "--model-path", "/opt/ml/processing/model/model_bundle.joblib",
            "--inference-features-path", "/opt/ml/processing/features/inference_features.parquet",
            "--inference-pairs-path", "/opt/ml/processing/features/inference_pairs.parquet",
            "--output-dir", "/opt/ml/processing/predictions",
        ],
        depends_on=[train_step],
    )

    pipeline = Pipeline(
        name=args.pipeline_name,
        parameters=[sales_s3_uri, test_s3_uri, items_s3_uri],
        steps=[feature_step, train_step, score_step],
        sagemaker_session=session,
    )
    pipeline.upsert(role_arn=args.role_arn)
    execution = pipeline.start()
    print(f"Started pipeline: {execution.arn}")
    print(f"Outputs prefix: {base_s3}")


if __name__ == "__main__":
    main()
