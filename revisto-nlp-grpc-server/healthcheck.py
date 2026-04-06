import asyncio
import os
import time

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

from common.logger import logger
from generated.nlp import nlp_pb2, nlp_pb2_grpc


def _get_client_credentials():
    """Load cert"""
    return grpc.ssl_channel_credentials()


def create_channel():
    """Builds an async secure or insecure channel."""
    target = os.getenv("GRPC_TARGET", "localhost:50051")
    use_tls = os.getenv("GRPC_USE_TLS", "false").lower() == "true"

    if use_tls:
        creds = _get_client_credentials()
        logger.info(f"[gRPC] Connecting securely to {target}")
        return grpc.aio.secure_channel(target, creds)

    logger.info(f"[gRPC] Connecting insecurely to {target}")
    return grpc.aio.insecure_channel(target)


async def nlp_health_call(stub: nlp_pb2_grpc.NLPServiceStub) -> bool:
    """Call the NLP service's Health RPC to verify models are ready."""
    try:
        resp = await stub.Health(nlp_pb2.HealthRequest(), timeout=3)
        if resp.healthy:
            logger.info("NLP service health: healthy")
            return True
        logger.warning(f"NLP service health: unhealthy (embedding={resp.embedding.ready}, ner={resp.ner.ready})")
        return False
    except grpc.aio.AioRpcError as e:
        logger.error(f"NLP health call failed: {e.code()} {e.details()}")
        return False


async def grpc_health_call(stub: health_pb2_grpc.HealthStub) -> bool:
    """Call the standard gRPC health service."""
    request = health_pb2.HealthCheckRequest(service="nlp.NLPService")

    try:
        resp = await stub.Check(request, timeout=2)
    except grpc.aio.AioRpcError as e:
        logger.error(f"HealthCheck error: {e.code()} {e.details()}")
        return False

    if resp.status == health_pb2.HealthCheckResponse.SERVING:
        logger.info("Health status: SERVING")
        return True

    logger.warning(f"Health status: {resp.status}")
    return False


async def run():
    channel = create_channel()
    nlp_stub = nlp_pb2_grpc.NLPServiceStub(channel)
    health_stub = health_pb2_grpc.HealthStub(channel)

    timeout = int(os.getenv("HC_TIMEOUT", "30"))
    interval = float(os.getenv("HC_INTERVAL", "1"))

    deadline = time.time() + timeout

    while time.time() < deadline:
        grpc_ok = await grpc_health_call(health_stub)
        nlp_ok = await nlp_health_call(nlp_stub)

        if grpc_ok and nlp_ok:
            logger.info("Server is healthy.")
            await channel.close()
            return

        await asyncio.sleep(interval)

    logger.error("Server did NOT become healthy in time.")
    await channel.close()
    exit(1)


if __name__ == "__main__":
    asyncio.run(run())
