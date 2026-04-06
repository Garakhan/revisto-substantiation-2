#!/usr/bin/env python3
"""Main entry point for the NLP gRPC service."""

import argparse
import asyncio
import signal
import threading
from concurrent import futures
from time import sleep

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from common.logger import logger
from generated.nlp import nlp_pb2_grpc
from services.nlp_service import NLPService
from settings import SETTINGS


def _get_server_credentials():
    """Get SSL credentials for secure connection."""
    creds_chain = ((SETTINGS.SERVER_CERTIFICATE_KEY.encode(), SETTINGS.SERVER_CERTIFICATE.encode()),)
    return grpc.ssl_server_credentials(creds_chain)


def _toggle_health(health_servicer: health.HealthServicer, service: str):
    """Toggle health status for the service."""
    next_status = health_pb2.HealthCheckResponse.SERVING
    while True:
        if next_status == health_pb2.HealthCheckResponse.SERVING:
            next_status = health_pb2.HealthCheckResponse.NOT_SERVING
        else:
            next_status = health_pb2.HealthCheckResponse.SERVING

        health_servicer.set(service, next_status)
        sleep(SETTINGS.TOGGLE_HEALTH_SLEEP_TIME)


def _configure_health_server(server: grpc.aio.Server):
    """Configure health checking for the server."""
    health_servicer = health.HealthServicer(
        experimental_non_blocking=True,
        experimental_thread_pool=futures.ThreadPoolExecutor(max_workers=SETTINGS.GRPC_MAX_WORKERS),
    )
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    # Set initial status as serving
    health_servicer.set("nlp.NLPService", health_pb2.HealthCheckResponse.SERVING)


async def serve(host: str, port: int):
    """Start the gRPC server."""
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=SETTINGS.GRPC_MAX_WORKERS))

    # Create and register the NLP service
    nlp_service = NLPService()

    # Preload models if configured
    if SETTINGS.PRELOAD_MODELS:
        logger.info("Preloading models...")
        nlp_service.preload_models()
        logger.info("Models preloaded successfully")

    nlp_pb2_grpc.add_NLPServiceServicer_to_server(nlp_service, server)

    listen_addr = f"{host}:{port}"
    if SETTINGS.LOCAL_MODE:
        server.add_insecure_port(listen_addr)
    else:
        server.add_secure_port(listen_addr, _get_server_credentials())

    _configure_health_server(server)

    await server.start()
    logger.info(f"NLP gRPC server started, listening on {listen_addr}")
    logger.info(f"  Embedding model: {SETTINGS.EMBEDDING_MODEL}")
    logger.info(f"  NER model: {SETTINGS.NER_MODEL}")
    logger.info(f"  Local mode: {SETTINGS.LOCAL_MODE}")

    # Wait for shutdown signal
    shutdown_event = asyncio.Event()

    # Register signal handlers
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_event.set)

    await shutdown_event.wait()

    logger.info("Shutting down gracefully...")
    await server.stop(SETTINGS.GRACEFUL_SHUTDOWN_SECONDS)
    logger.info("Shutdown complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLP gRPC Server")
    parser.add_argument("--host", default=SETTINGS.SERVICE_DEFAULT_HOST, help="Host to bind to")
    parser.add_argument("--port", type=int, default=SETTINGS.SERVICE_DEFAULT_PORT, help="Port to bind to")
    args = parser.parse_args()

    asyncio.run(serve(args.host, args.port))
