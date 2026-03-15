from __future__ import annotations

import json
import logging
import os
from typing import Any

import boto3


logger = logging.getLogger(__name__)


def load_aws_secret(secret_name: str, region_name: str = "us-east-1") -> dict[str, Any]:
    client = boto3.client("secretsmanager", region_name=region_name)
    response = client.get_secret_value(SecretId=secret_name)
    secret = response.get("SecretString")
    if not secret:
        return {}
    payload = json.loads(secret)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Secret {secret_name} did not contain a JSON object.")
    return payload


def apply_secret_env(secret_name: str | None, region_name: str | None = None) -> bool:
    if not secret_name:
        return False
    resolved_region = region_name or os.getenv("AWS_REGION", "us-east-1")
    payload = load_aws_secret(secret_name, resolved_region)
    for key, value in payload.items():
        if value is None:
            continue
        os.environ.setdefault(str(key), str(value))
    logger.info("Loaded application secrets from AWS Secrets Manager: %s", secret_name)
    return True
