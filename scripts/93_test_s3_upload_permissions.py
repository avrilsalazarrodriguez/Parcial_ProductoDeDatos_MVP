#!/usr/bin/env python3
from __future__ import annotations

import os
from datetime import datetime, timezone
import boto3


def resolve_bucket() -> str:
    for name in ['MODEL_BUCKET', 'MODELOPS_BUCKET', 'MODEL_OPS_BUCKET', 'S3_BUCKET', 'BUCKET_NAME']:
        value = os.getenv(name)
        if value:
            return value
    raise RuntimeError('Define MODEL_BUCKET o MODELOPS_BUCKET.')


def main() -> None:
    bucket = resolve_bucket()
    region = os.getenv('AWS_REGION') or os.getenv('AWS_DEFAULT_REGION') or 'us-east-1'
    s3 = boto3.client('s3', region_name=region)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    key = f'app/batch_uploads/_diagnostics/put_test_{timestamp}.txt'
    body = f'put test {timestamp}\n'.encode('utf-8')
    print('bucket:', bucket)
    print('region:', region)
    print('put:', f's3://{bucket}/{key}')
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType='text/plain')
    obj = s3.get_object(Bucket=bucket, Key=key)
    print('read bytes:', len(obj['Body'].read()))
    print('OK: PutObject/GetObject funciona para este perfil/rol.')


if __name__ == '__main__':
    main()
