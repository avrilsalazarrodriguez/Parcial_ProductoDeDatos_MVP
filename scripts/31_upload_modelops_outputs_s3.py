from __future__ import annotations

import argparse
from pathlib import Path

import boto3


def upload_tree(local_root: Path, bucket: str, prefix: str, region: str) -> None:
    s3 = boto3.client("s3", region_name=region)
    for path in local_root.rglob("*"):
        if path.is_file():
            key = f"{prefix.strip('/')}/{path.relative_to(local_root).as_posix()}"
            s3.upload_file(str(path), bucket, key)
            print(f"uploaded s3://{bucket}/{key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--local-root", default="modelops_outputs")
    parser.add_argument("--latest-prefix", default="modelops/latest")
    parser.add_argument("--registry-prefix", default="modelops/registry")
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()

    root = Path(args.local_root)
    upload_tree(root / "latest", args.bucket, args.latest_prefix, args.region)
    upload_tree(root / "registry", args.bucket, args.registry_prefix, args.region)


if __name__ == "__main__":
    main()
