#!/usr/bin/env python3
from __future__ import annotations

import os
import boto3
import pandas as pd


def resolve_bucket() -> str:
    for name in ['MODEL_BUCKET', 'MODELOPS_BUCKET', 'MODEL_OPS_BUCKET', 'S3_BUCKET', 'BUCKET_NAME']:
        value = os.getenv(name)
        if value:
            return value
    raise RuntimeError('Define MODEL_BUCKET o MODELOPS_BUCKET antes de ejecutar este script.')


def main() -> None:
    bucket = resolve_bucket()
    region = os.getenv('AWS_REGION') or os.getenv('AWS_DEFAULT_REGION') or 'us-east-1'
    predictions_prefix = os.getenv('UPLOADED_BATCH_S3_PREFIX', 'app/batch_uploads/predictions').strip('/') + '/'
    history_key = os.getenv('UPLOADED_BATCH_HISTORY_KEY', 'app/batch_uploads/history/uploaded_predictions_history.csv')
    s3 = boto3.client('s3', region_name=region)

    print('bucket:', bucket)
    print('predictions prefix:', f's3://{bucket}/{predictions_prefix}')
    print('history:', f's3://{bucket}/{history_key}')

    try:
        obj = s3.get_object(Bucket=bucket, Key=history_key)
        history = pd.read_csv(obj['Body'])
        print('\nHistorial exclusivo de predicciones cargadas:')
        print(history.head(20).to_string(index=False))
    except Exception as exc:
        print('\nNo pude leer historial CSV:', exc)

    print('\nObjetos bajo prefijo exclusivo:')
    response = s3.list_objects_v2(Bucket=bucket, Prefix=predictions_prefix)
    for obj in response.get('Contents', [])[-20:]:
        print(obj.get('LastModified'), obj.get('Size'), f"s3://{bucket}/{obj.get('Key')}")


if __name__ == '__main__':
    main()
