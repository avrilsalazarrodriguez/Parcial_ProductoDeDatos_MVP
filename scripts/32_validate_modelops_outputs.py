from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path

import boto3
import pandas as pd


def read_local(root: Path, rel: str):
    path = root / rel
    if rel.endswith('.json'):
        return json.loads(path.read_text(encoding='utf-8'))
    if rel.endswith('.csv'):
        return pd.read_csv(path)
    return pd.read_parquet(path)


def read_s3(bucket: str, key: str, region: str):
    body = boto3.client('s3', region_name=region).get_object(Bucket=bucket, Key=key)['Body'].read()
    if key.endswith('.json'):
        return json.loads(body.decode('utf-8'))
    if key.endswith('.csv'):
        return pd.read_csv(BytesIO(body))
    return pd.read_parquet(BytesIO(body))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-root', default=None)
    parser.add_argument('--bucket', default=None)
    parser.add_argument('--latest-prefix', default='modelops/latest')
    parser.add_argument('--registry-prefix', default='modelops/registry')
    parser.add_argument('--region', default='us-east-1')
    args = parser.parse_args()

    if args.local_root:
        root = Path(args.local_root)
        champion = read_local(root / 'registry', 'champion.json')
        runs = read_local(root / 'registry', 'model_runs.csv')
        curves = read_local(root / 'latest', 'evaluation/evaluation_curves_by_model.parquet')
        review = read_local(root / 'latest', 'review/review_suggestions.parquet')
        over = read_local(root / 'latest', 'review/overestimated_products.parquet')
    else:
        champion = read_s3(args.bucket, f"{args.registry_prefix}/champion.json", args.region)
        runs = read_s3(args.bucket, f"{args.registry_prefix}/model_runs.csv", args.region)
        curves = read_s3(args.bucket, f"{args.latest_prefix}/evaluation/evaluation_curves_by_model.parquet", args.region)
        review = read_s3(args.bucket, f"{args.latest_prefix}/review/review_suggestions.parquet", args.region)
        over = read_s3(args.bucket, f"{args.latest_prefix}/review/overestimated_products.parquet", args.region)

    assert not runs.empty
    assert {'model_id', 'rmse', 'mae', 'is_champion'}.issubset(runs.columns)
    assert not curves.empty
    assert {'model_id', 'series', 'demand_bucket', 'mean_value'}.issubset(curves.columns)
    assert not review.empty
    assert not over.empty
    print('VALIDATION OK')
    print('champion:', champion.get('champion', {}).get('model_id', 'N/D'))
    print('runs:', runs[['model_id', 'rmse', 'mae', 'is_champion']].to_string(index=False))
    print('curves:', curves.shape)
    print('review:', review.shape)
    print('overestimated:', over.shape)


if __name__ == '__main__':
    main()
