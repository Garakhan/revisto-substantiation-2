import argparse
import asyncio
import signal
from concurrent import futures

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from common.logger import logger
from generated.nlp import nlp_pb2_grpc
from services.nlp_services import NLPService
from settings import SETTINGS

HEALTH_CHECK_INTERVAL = 30


def _get_server_credentials():
    creds_chain = ((SETTINGS.SERVER_CERTIFICATE_KEY.encode(), SETTINGS.SERVER_CERTIFICATE.encode()),)
    return grpc.ssl_server_credentials(creds_chain)


def _register_health_service(server: grpc.aio.Server) -> health.HealthServicer:
    """Register the gRPC health service on the server (must be called before start)."""
    health_servicer = health.HealthServicer(
        experimental_non_blocking=True,
        experimental_thread_pool=futures.ThreadPoolExecutor(max_workers=10),
    )
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    health_servicer.set("nlp.NLPService", health_pb2.HealthCheckResponse.SERVING)
    return health_servicer


def _start_health_polling(health_servicer: health.HealthServicer, nlp_service: NLPService):
    """Start a background task that polls model readiness (call after server.start)."""

    async def _update_health():
        while True:
            models_ok = nlp_service.is_ready()
            status = health_pb2.HealthCheckResponse.SERVING if models_ok else health_pb2.HealthCheckResponse.NOT_SERVING
            health_servicer.set("nlp.NLPService", status)
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)

    asyncio.create_task(_update_health())


async def serve(host: str, port: int):
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=SETTINGS.GRPC_MAX_WORKERS))
    nlp_service = NLPService()
    await nlp_service.preload_models()
    nlp_pb2_grpc.add_NLPServiceServicer_to_server(nlp_service, server)

    listen_addr = f"{host}:{port}"
    if SETTINGS.LOCAL_MODE:
        server.add_insecure_port(listen_addr)
    else:
        server.add_secure_port(listen_addr, _get_server_credentials())

    health_servicer = _register_health_service(server)

    await server.start()
    _start_health_polling(health_servicer, nlp_service)
    logger.info(f"gRPC server started, listening on {listen_addr}")

    # Wait for shutdown signal
    shutdown_event = asyncio.Event()

    # Register signal handlers
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_event.set)

    await shutdown_event.wait()

    logger.info("Shutting down gracefully...")
    await server.stop(SETTINGS.GRACEFUL_SHUTDOWN_SECONDS)  # allow up to 5s for graceful shutdown
    logger.info("Shutdown complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gRPC Server")
    parser.add_argument("--host", default=SETTINGS.SERVICE_DEFAULT_HOST, help="Host to listen on")
    parser.add_argument("--port", type=int, default=SETTINGS.SERVICE_DEFAULT_PORT, help="Port to listen on")
    args = parser.parse_args()

    asyncio.run(serve(args.host, args.port))
