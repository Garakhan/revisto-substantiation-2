"""Main FastAPI application"""

import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from ..config import get_config
from ..storage import ESClient
from ..nlp import load_embedder, NERExtractor
from ..nlp.tokenization import StanzaNumericExtractor
from ..grpc_client import GrpcEmbeddingModel, GrpcNERExtractor, GrpcSentenciser, GrpcNumericExtractor
from ..utils.table_linearizer import ClaudeLinearizer
from ..utils.logging import get_logger
from .routers import health, indexing, search
from .models import ErrorResponse
from .dependencies import resources

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    logger.info("Starting up application...")
    
    # Load configuration
    config = get_config()
    resources["config"] = config
    
    # Initialize Elasticsearch
    try:
        es_client = ESClient(config.es)
        # Test connection
        _ = es_client.client
        resources["es_client"] = es_client
        logger.info("Connected to Elasticsearch")
    except Exception as e:
        logger.error(f"Failed to connect to Elasticsearch: {e}")
        raise
    
    # Load models based on NLP mode
    logger.info(f"Loading models (mode: {config.nlp_mode})...")

    if config.nlp_mode == "grpc":
        # Use gRPC clients for NLP models
        try:
            embedder = GrpcEmbeddingModel(config.embed, grpc_address=config.grpc.address)
            resources["embedder"] = embedder
            logger.info(f"Connected to gRPC embedding service at {config.grpc.address}")
        except Exception as e:
            logger.error(f"Failed to connect to gRPC embedding service: {e}")
            raise

        if config.ner.enabled:
            try:
                ner_extractor = GrpcNERExtractor(config.ner, grpc_address=config.grpc.address)
                resources["ner_extractor"] = ner_extractor
                logger.info(f"Connected to gRPC NER service at {config.grpc.address}")
            except Exception as e:
                logger.warning(f"Failed to connect to gRPC NER service: {e}")
                resources["ner_extractor"] = None
        else:
            resources["ner_extractor"] = None

        # Sentenciser via gRPC
        try:
            sentenciser = GrpcSentenciser(grpc_address=config.grpc.address)
            resources["sentenciser"] = sentenciser
            logger.info(f"Connected to gRPC sentenciser service at {config.grpc.address}")
        except Exception as e:
            logger.warning(f"Failed to connect to gRPC sentenciser service: {e}")
            resources["sentenciser"] = None

        # Numeric extractor via gRPC (uses Stanza NER on server)
        try:
            numeric_extractor = GrpcNumericExtractor(grpc_address=config.grpc.address)
            resources["numeric_extractor"] = numeric_extractor
            logger.info(f"Connected to gRPC numeric extractor service at {config.grpc.address}")
        except Exception as e:
            logger.warning(f"Failed to connect to gRPC numeric extractor service: {e}")
            resources["numeric_extractor"] = None
    else:
        # Load models locally (default)
        try:
            embedder = load_embedder(config.embed)
            resources["embedder"] = embedder
            logger.info("Loaded embedding model locally")
        except Exception as e:
            logger.error(f"Failed to load embedder: {e}")
            raise

        if config.ner.enabled:
            try:
                ner_extractor = NERExtractor(config.ner)
                resources["ner_extractor"] = ner_extractor
                logger.info("Loaded NER model locally")
            except Exception as e:
                logger.warning(f"Failed to load NER model: {e}")
                resources["ner_extractor"] = None
        else:
            resources["ner_extractor"] = None

        # Local mode uses stanza directly in indexing.py, no sentenciser resource needed
        resources["sentenciser"] = None

        # Numeric extractor using local Stanza NER
        try:
            numeric_extractor = StanzaNumericExtractor()
            resources["numeric_extractor"] = numeric_extractor
            logger.info("Initialized local Stanza numeric extractor")
        except Exception as e:
            logger.warning(f"Failed to initialize numeric extractor: {e}")
            resources["numeric_extractor"] = None

    # Initialize table linearizer (uses Claude API - requires ANTHROPIC_API_KEY)
    try:
        table_linearizer = ClaudeLinearizer()
        resources["table_linearizer"] = table_linearizer
        logger.info("Initialized Claude table linearizer")
    except Exception as e:
        logger.warning(f"Failed to initialize table linearizer (table support disabled): {e}")
        resources["table_linearizer"] = None

    logger.info("Application startup complete")
    
    yield
    
    # Cleanup
    logger.info("Shutting down application...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="Revisto Evidence Aligned API",
        description="API for evidence alignment and claim extraction",
        version="0.3.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add request ID middleware
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(time.time()))
        request.state.request_id = request_id
        
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        
        error_response = ErrorResponse(
            error="Internal server error",
            detail=str(exc) if app.debug else None,
            request_id=getattr(request.state, "request_id", None)
        )
        
        return JSONResponse(
            status_code=500,
            content=error_response.model_dump()
        )
    
    # Include routers
    app.include_router(health.router, prefix="/api/health", tags=["health"])
    app.include_router(indexing.router, prefix="/api/index", tags=["indexing"])
    app.include_router(search.router, prefix="/api/search", tags=["search"])
    
    # Add Prometheus instrumentation
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
    
    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "message": "Revisto Evidence Aligned API",
            "version": "0.3.0",
            "docs": "/api/docs"
        }
    
    return app


# Make resources accessible
