"""
Cleanup script — elimina todos los recursos de LocalStack creados por deploy_localstack.py:
  - Función Lambda
  - Bucket S3 (documentos + índice ChromaDB + ZIP de Lambda)
  - REST API Gateway
  - Artifact local function.zip

Ejecutar con LocalStack corriendo:
    python scripts/cleanup_localstack.py
"""

import sys
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

import boto3
from botocore.exceptions import ClientError
from src.config import get_settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

LAMBDA_FUNCTION_NAME = "technical-docs-assistant"
API_GATEWAY_NAME     = "docs-assistant-api"


def _boto3_client(service: str, settings):
    kwargs = {"service_name": service, "region_name": settings.aws_region}
    if settings.aws_endpoint_url:
        kwargs["endpoint_url"] = settings.aws_endpoint_url
    if settings.aws_access_key_id:
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
    return boto3.client(**kwargs)


def delete_s3_bucket(s3_client, bucket_name: str):
    """Vacía el bucket (documentos, índice ChromaDB, ZIP de Lambda) y lo elimina."""
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket_name):
            objects = page.get("Contents", [])
            if objects:
                s3_client.delete_objects(
                    Bucket=bucket_name,
                    Delete={"Objects": [{"Key": o["Key"]} for o in objects]},
                )
        s3_client.delete_bucket(Bucket=bucket_name)
        logger.info(f"S3 bucket '{bucket_name}' deleted.")
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchBucket":
            logger.info(f"S3 bucket '{bucket_name}' does not exist — skipping.")
        else:
            raise


def delete_lambda(lambda_client):
    try:
        lambda_client.delete_function(FunctionName=LAMBDA_FUNCTION_NAME)
        logger.info(f"Lambda function '{LAMBDA_FUNCTION_NAME}' deleted.")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceNotFoundException":
            logger.info(f"Lambda '{LAMBDA_FUNCTION_NAME}' does not exist — skipping.")
        else:
            raise


def delete_api_gateway(apigw_client):
    try:
        apis = apigw_client.get_rest_apis().get("items", [])
        api = next((a for a in apis if a["name"] == API_GATEWAY_NAME), None)
        if api:
            apigw_client.delete_rest_api(restApiId=api["id"])
            logger.info(f"API Gateway '{API_GATEWAY_NAME}' (id={api['id']}) deleted.")
        else:
            logger.info(f"API Gateway '{API_GATEWAY_NAME}' does not exist — skipping.")
    except ClientError as e:
        raise


def delete_local_artifact():
    zip_path = root_path / "function.zip"
    if zip_path.exists():
        zip_path.unlink()
        logger.info(f"Local artifact {zip_path} deleted.")
    else:
        logger.info("No local function.zip found — skipping.")


def main():
    settings = get_settings()

    if settings.environment != "localstack":
        logger.error(
            "ENVIRONMENT must be 'localstack'. Set it in your .env:\n"
            "  ENVIRONMENT=localstack"
        )
        sys.exit(1)

    logger.info(f"Cleaning up LocalStack resources at {settings.aws_endpoint_url}")

    s3_client     = _boto3_client("s3", settings)
    lambda_client = _boto3_client("lambda", settings)
    apigw_client  = _boto3_client("apigateway", settings)

    delete_s3_bucket(s3_client, settings.s3_bucket_name)
    delete_lambda(lambda_client)
    delete_api_gateway(apigw_client)
    delete_local_artifact()

    logger.info("Cleanup complete.")


if __name__ == "__main__":
    main()
