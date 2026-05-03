#!/usr/bin/env python3
from __future__ import annotations

"""Increase ALB idle timeout to reduce WebSocket disconnects for Streamlit."""

import argparse

import boto3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alb-arn", default="")
    parser.add_argument("--name-contains", default="pfs-mvp")
    parser.add_argument("--idle-timeout", default="120")
    parser.add_argument("--region", default=None)
    return parser.parse_args()


def find_alb_arn(elbv2, name_contains: str) -> str:
    lbs = elbv2.describe_load_balancers()["LoadBalancers"]
    matches = [lb for lb in lbs if name_contains in lb.get("LoadBalancerName", "") or name_contains in lb.get("DNSName", "")]
    if not matches:
        raise RuntimeError(f"No encontré ALB que contenga: {name_contains}")
    if len(matches) > 1:
        print("Múltiples ALB encontrados. Uso el primero:")
        for lb in matches:
            print("-", lb["LoadBalancerName"], lb["DNSName"], lb["LoadBalancerArn"])
    return matches[0]["LoadBalancerArn"]


def main() -> None:
    args = parse_args()
    elbv2 = boto3.client("elbv2", region_name=args.region)
    alb_arn = args.alb_arn or find_alb_arn(elbv2, args.name_contains)
    elbv2.modify_load_balancer_attributes(
        LoadBalancerArn=alb_arn,
        Attributes=[{"Key": "idle_timeout.timeout_seconds", "Value": str(args.idle_timeout)}],
    )
    print("Updated ALB idle timeout")
    print("alb_arn:", alb_arn)
    print("idle_timeout.timeout_seconds:", args.idle_timeout)


if __name__ == "__main__":
    main()
