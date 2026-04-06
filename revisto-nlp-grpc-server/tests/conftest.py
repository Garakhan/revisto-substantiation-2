import grpc
import pytest

from generated.nlp import nlp_pb2, nlp_pb2_grpc
from settings import SETTINGS


async def _is_server_ready(host: str, port: int) -> bool:
    """Check if a gRPC server is running AND healthy (all enabled models ready)."""
    try:
        async with grpc.aio.insecure_channel(f"{host}:{port}") as channel:
            stub = nlp_pb2_grpc.NLPServiceStub(channel)
            resp = await stub.Health(nlp_pb2.HealthRequest(), timeout=2)
            return resp.healthy
    except Exception:
        return False


@pytest.fixture(scope="session")
def server_address():
    return f"localhost:{SETTINGS.SERVICE_DEFAULT_PORT}"
