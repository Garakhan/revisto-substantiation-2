import json
import os
from typing import Any, Dict, Tuple, Type

import boto3
from botocore.exceptions import ClientError, NoCredentialsError, NoRegionError
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

SECRET_KEY = os.environ.get("SECRET_KEY") or "key"
REGION = os.environ.get("REGION") or "us-east-1"
PROFILE = os.environ.get("DEV_PROFILE") or None

ENVIRONMENT = os.getenv("ENVIRONMENT", "production")


def get_secret():
    if ENVIRONMENT == "local":
        return {}

    session = boto3.session.Session(region_name=REGION, profile_name=PROFILE)
    client = session.client(
        service_name="secretsmanager",
        region_name=REGION,
    )

    try:
        get_secret_value_response = client.get_secret_value(SecretId=SECRET_KEY)
    except ClientError as e:
        print(f"Error getting secret: {e}")
        return {}
    except NoCredentialsError as e:
        print(f"Error getting secret: {e}")
        return {}
    except NoRegionError as e:
        print(f"Error getting secret: {e}")
        return {}

    return json.loads(get_secret_value_response["SecretString"])


_cached_secrets = None


def get_cached_secrets():
    global _cached_secrets
    if _cached_secrets is None:
        _cached_secrets = get_secret()
    return _cached_secrets


class SecretManagerSource(PydanticBaseSettingsSource):
    def get_field_value(self, field: FieldInfo, field_name: str) -> Tuple[Any, str, bool]:
        secrets = get_cached_secrets()
        field_value = secrets.get(field_name)
        return field_value, field_name, False

    def prepare_field_value(
        self,
        field_name: str,
        field: FieldInfo,
        value: Any,
        value_is_complex: bool,
    ) -> Any:
        return value

    def __call__(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}

        for field_name, field in self.settings_cls.model_fields.items():
            field_value, field_key, value_is_complex = self.get_field_value(field, field_name)
            field_value = self.prepare_field_value(field_name, field, field_value, value_is_complex)
            if field_value is not None:
                d[field_key] = field_value
        return d


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file_encoding="utf-8")

    # Server settings
    SERVICE_DEFAULT_HOST: str = "0.0.0.0"
    SERVICE_DEFAULT_PORT: int = 50051

    LOCAL_MODE: bool = True
    GRPC_MAX_WORKERS: int = 10
    GRACEFUL_SHUTDOWN_SECONDS: int = 5

    # SSL/TLS settings
    SERVER_CERTIFICATE_KEY: str = ""
    SERVER_CERTIFICATE: str = ""
    CLIENT_CERTIFICATE: str = ""

    # Embedding model settings
    EMBEDDING_MODEL: str = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_NORMALIZE: bool = True
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_ENABLED: bool = True

    # Named entity recognition model settings
    # NER_MODEL: str = "en_ner_bc5cdr_md" # d4data/biomedical-ner-all
    NER_MODEL: str = "d4data/biomedical-ner-all"
    NER_BATCH_SIZE: int = 16
    NER_ENABLED: bool = True

    # Sentenciser settings (Stanza)
    STANZA_LANG: str = "en"
    SENTENCISER_ENABLED: bool = True

    # Numeric entity recognition settings
    NUMERIC_NER_ENABLED: bool = True

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (
            env_settings,
            SecretManagerSource(settings_cls),
            file_secret_settings,
            init_settings,
        )


SETTINGS = Settings()
