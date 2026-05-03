#!/usr/bin/env python3
from __future__ import annotations

"""Register a new ECS Fargate task definition revision with larger CPU/memory.

Usage:
  PYTHONPATH=. uv run python scripts/83_update_ecs_fargate_resources.py \
    --cluster pfs-mvp-cluster \
    --service pfs-mvp \
    --cpu 2048 \
    --memory 4096 \
    --region us-east-1
"""

import argparse
import copy
from typing import Any

import boto3

TASK_DEF_KEYS = {
    "family",
    "taskRoleArn",
    "executionRoleArn",
    "networkMode",
    "containerDefinitions",
    "volumes",
    "placementConstraints",
    "requiresCompatibilities",
    "cpu",
    "memory",
    "tags",
    "pidMode",
    "ipcMode",
    "proxyConfiguration",
    "inferenceAccelerators",
    "ephemeralStorage",
    "runtimePlatform",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", default="pfs-mvp-cluster")
    parser.add_argument("--service", default="pfs-mvp")
    parser.add_argument("--cpu", default="2048", help="Task CPU units. 1024=1 vCPU, 2048=2 vCPU.")
    parser.add_argument("--memory", default="4096", help="Task memory in MiB. 4096=4GB.")
    parser.add_argument("--region", default=None)
    parser.add_argument("--container-memory", default=None, help="Optional hard memory limit for first container in MiB.")
    parser.add_argument("--container-cpu", default=None, help="Optional CPU units for first container.")
    return parser.parse_args()


def clean_task_definition(task_def: dict[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in task_def.items() if key in TASK_DEF_KEYS and value not in (None, [], {})}


def main() -> None:
    args = parse_args()
    ecs = boto3.client("ecs", region_name=args.region)

    svc = ecs.describe_services(cluster=args.cluster, services=[args.service])["services"][0]
    current_td_arn = svc["taskDefinition"]
    current = ecs.describe_task_definition(taskDefinition=current_td_arn, include=["TAGS"])
    task_def = clean_task_definition(current["taskDefinition"])

    task_def["cpu"] = str(args.cpu)
    task_def["memory"] = str(args.memory)

    if task_def.get("containerDefinitions"):
        container = task_def["containerDefinitions"][0]
        if args.container_cpu is not None:
            container["cpu"] = int(args.container_cpu)
        if args.container_memory is not None:
            container["memory"] = int(args.container_memory)

    # Keep tags if present from describe_task_definition include=[TAGS]
    if current.get("tags"):
        task_def["tags"] = current["tags"]

    registered = ecs.register_task_definition(**task_def)
    new_td_arn = registered["taskDefinition"]["taskDefinitionArn"]
    ecs.update_service(cluster=args.cluster, service=args.service, taskDefinition=new_td_arn, forceNewDeployment=True)

    print("Updated ECS service with new task definition")
    print("cluster:", args.cluster)
    print("service:", args.service)
    print("old_task_definition:", current_td_arn)
    print("new_task_definition:", new_td_arn)
    print("cpu:", args.cpu)
    print("memory:", args.memory)


if __name__ == "__main__":
    main()
