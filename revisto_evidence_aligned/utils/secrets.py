"""AWS Secrets Manager utilities"""

import json
import os
from typing import Dict, Optional, Any
from functools import lru_cache

import boto3
from botocore.exceptions import ClientError

from .logging import get_logger

logger = get_logger(__name__)


class SecretsManager:
    """Manage secrets from AWS Secrets Manager"""
    
    def __init__(self, region_name: Optional[str] = None):
        """
        Initialize Secrets Manager client
        
        Args:
            region_name: AWS region (defaults to AWS_DEFAULT_REGION env var or us-east-1)
        """
        self.region_name = region_name or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        self._client = None
    
    @property
    def client(self):
        """Lazy load boto3 client"""
        if self._client is None:
            self._client = boto3.client("secretsmanager", region_name=self.region_name)
        return self._client
    
    def get_secret(self, secret_name: str) -> Dict[str, Any]:
        """
        Retrieve secret from AWS Secrets Manager
        
        Args:
            secret_name: Name of the secret in AWS
            
        Returns:
            Dict containing the secret values
            
        Raises:
            ClientError: If secret cannot be retrieved
        """
        try:
            logger.info(f"Retrieving secret: {secret_name}")
            response = self.client.get_secret_value(SecretId=secret_name)
            
            # Secrets Manager returns either SecretString or SecretBinary
            if "SecretString" in response:
                secret = response["SecretString"]
                # Try to parse as JSON, otherwise return as dict with single key
                try:
                    return json.loads(secret)
                except json.JSONDecodeError:
                    return {"value": secret}
            else:
                # Binary secrets not supported for this use case
                raise ValueError(f"Binary secret not supported: {secret_name}")
                
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "ResourceNotFoundException":
                logger.error(f"Secret not found: {secret_name}")
            elif error_code == "AccessDeniedException":
                logger.error(f"Access denied to secret: {secret_name}")
            else:
                logger.error(f"Error retrieving secret {secret_name}: {e}")
            raise
    
    @lru_cache(maxsize=32)
    def get_cached_secret(self, secret_name: str) -> Dict[str, Any]:
        """
        Get secret with caching to avoid repeated AWS calls
        
        Args:
            secret_name: Name of the secret
            
        Returns:
            Cached secret dict
        """
        return self.get_secret(secret_name)


# Global instance
_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager() -> SecretsManager:
    """Get or create global secrets manager instance"""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager


def get_elasticsearch_config() -> Dict[str, str]:
    """
    Get Elasticsearch configuration from AWS Secrets Manager
    
    Expected secret format:
    {
        "ES_URL": "https://elasticsearch.example.com:9200",
        "API_KEY": "your-api-key"
    }
    
    Returns:
        Dict with es_url and api_key
    """
    secret_name = os.getenv("ES_SECRET_NAME", "develop")
    
    try:
        manager = get_secrets_manager()
        secret_data = manager.get_cached_secret(secret_name)
        
        # Validate required fields (using uppercase keys as stored in AWS)
        if "ES_URL" not in secret_data or "API_KEY" not in secret_data:
            raise ValueError("Secret must contain 'ES_URL' and 'API_KEY' fields")
        
        return {
            "es_url": secret_data["ES_URL"],
            "api_key": secret_data["API_KEY"]
        }
        
    except Exception as e:
        logger.error(f"Failed to get Elasticsearch config from secrets: {e}")
        
        # Fall back to environment variables if secrets fail
        es_url = os.getenv("ES_URL")
        api_key = os.getenv("API_KEY")
        
        if es_url and api_key:
            logger.warning("Using environment variables as fallback")
            return {"es_url": es_url, "api_key": api_key}

        raise RuntimeError("Unable to get Elasticsearch configuration from secrets or environment")


def get_landingai_api_key() -> str:
    """
    Get LandingAI API key from AWS Secrets Manager

    Expected secret format:
    {
        "LANDING_AI_API_KEY": "..."
    }

    Returns:
        LandingAI API key string
    """
    # First check environment variable
    env_key = os.getenv("LANDING_AI_API_KEY")
    if env_key:
        return env_key

    secret_name = os.getenv("ES_SECRET_NAME", "develop")

    try:
        manager = get_secrets_manager()
        secret_data = manager.get_cached_secret(secret_name)

        if "LANDING_AI_API_KEY" in secret_data:
            return secret_data["LANDING_AI_API_KEY"]

        logger.warning(f"LANDING_AI_API_KEY not found in secret '{secret_name}'")
        return None

    except Exception as e:
        logger.error(f"Failed to get LandingAI API key from secrets: {e}")
        return None


def get_anthropic_api_key() -> str:
    """
    Get Anthropic API key from AWS Secrets Manager

    Expected secret format (in the same secret as ES config):
    {
        "ES_URL": "...",
        "API_KEY": "...",
        "ANTHROPIC_API_KEY": "sk-ant-..."
    }

    Returns:
        Anthropic API key string
    """
    # First check environment variable
    env_key = os.getenv("ANTHROPIC_API_KEY")
    if env_key:
        return env_key

    secret_name = os.getenv("ES_SECRET_NAME", "develop")

    try:
        manager = get_secrets_manager()
        secret_data = manager.get_cached_secret(secret_name)

        if "ANTHROPIC_API_KEY" in secret_data:
            return secret_data["ANTHROPIC_API_KEY"]

        logger.warning(f"ANTHROPIC_API_KEY not found in secret '{secret_name}'")
        return None

    except Exception as e:
        logger.error(f"Failed to get Anthropic API key from secrets: {e}")
        return None


def get_openai_api_key() -> str:
    """
    Get OpenAI API key from AWS Secrets Manager

    Expected secret format (in the same secret as ES config):
    {
        "ES_URL": "...",
        "API_KEY": "...",
        "OPENAI_API_KEY": "sk-..."
    }

    Returns:
        OpenAI API key string
    """
    # First check environment variable
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key

    secret_name = os.getenv("ES_SECRET_NAME", "develop")

    try:
        manager = get_secrets_manager()
        secret_data = manager.get_cached_secret(secret_name)

        if "OPENAI_API_KEY" in secret_data:
            return secret_data["OPENAI_API_KEY"]

        logger.warning(f"OPENAI_API_KEY not found in secret '{secret_name}'")
        return None

    except Exception as e:
        logger.error(f"Failed to get OpenAI API key from secrets: {e}")
        return None


# S3 utilities for LandingAI zero data retention mode
_s3_client = None


def get_s3_client():
    """Get or create S3 client"""
    global _s3_client
    if _s3_client is None:
        from botocore.config import Config
        region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        # Use signature v4 for presigned URLs (required by many services)
        config = Config(signature_version='s3v4')
        _s3_client = boto3.client("s3", region_name=region, config=config)
    return _s3_client


def generate_presigned_upload_url(bucket: str, key: str, expiration: int = 3600, content_type: str = None) -> str:
    """
    Generate a presigned URL for uploading to S3.

    Args:
        bucket: S3 bucket name
        key: Object key (path) in the bucket
        expiration: URL expiration time in seconds (default 1 hour)
        content_type: Optional content type for the upload

    Returns:
        Presigned URL for PUT upload
    """
    s3 = get_s3_client()
    params = {"Bucket": bucket, "Key": key}
    if content_type:
        params["ContentType"] = content_type
    url = s3.generate_presigned_url(
        "put_object",
        Params=params,
        ExpiresIn=expiration
    )
    return url


def generate_presigned_download_url(bucket: str, key: str, expiration: int = 3600) -> str:
    """
    Generate a presigned URL for downloading from S3.

    Args:
        bucket: S3 bucket name
        key: Object key (path) in the bucket
        expiration: URL expiration time in seconds (default 1 hour)

    Returns:
        Presigned URL for GET download
    """
    s3 = get_s3_client()
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expiration
    )
    return url


def download_from_s3(bucket: str, key: str) -> bytes:
    """
    Download object content from S3.

    Args:
        bucket: S3 bucket name
        key: Object key (path) in the bucket

    Returns:
        Object content as bytes
    """
    s3 = get_s3_client()
    response = s3.get_object(Bucket=bucket, Key=key)
    return response["Body"].read()


def upload_to_s3(bucket: str, key: str, file_path: str) -> None:
    """
    Upload a file to S3.

    Args:
        bucket: S3 bucket name
        key: Object key (path) in the bucket
        file_path: Local file path to upload
    """
    s3 = get_s3_client()
    s3.upload_file(file_path, bucket, key)


def get_landingai_output_bucket() -> Optional[str]:
    """
    Get S3 bucket for LandingAI visual_claims storage.

    Returns:
        Bucket name or None if not configured
    """
    return os.getenv("LANDINGAI_OUTPUT_BUCKET")