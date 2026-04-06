"""Health check endpoints"""

from fastapi import APIRouter, Depends
from typing import Dict, Any

from ..dependencies import get_resources
from ..models import HealthResponse
from ...utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/", response_model=HealthResponse)
async def health_check():
    """Basic health check"""
    resources = get_resources()
    
    # Check Elasticsearch
    es_status = {"connected": False}
    try:
        es_client = resources.get("es_client")
        if es_client:
            info = es_client.client.info()
            es_status = {
                "connected": True,
                "cluster_name": info.get("cluster_name", "unknown"),
                "version": info.get("version", {}).get("number", "unknown")
            }
    except Exception as e:
        es_status["error"] = str(e)
    
    # Check models
    models_status = {
        "embedder": resources.get("embedder") is not None,
        "ner": resources.get("ner_extractor") is not None
    }
    
    return HealthResponse(
        status="healthy" if es_status["connected"] else "degraded",
        version="0.3.0",
        elasticsearch=es_status,
        models_loaded=models_status
    )


@router.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes"""
    resources = get_resources()
    
    # Check if essential resources are loaded
    es_client = resources.get("es_client")
    embedder = resources.get("embedder")
    
    if not es_client or not embedder:
        return {"ready": False, "reason": "Resources not loaded"}
    
    # Check ES connection
    try:
        es_client.client.ping()
    except:
        return {"ready": False, "reason": "Elasticsearch not reachable"}
    
    return {"ready": True}


@router.get("/live")
async def liveness_check():
    """Liveness check for Kubernetes"""
    return {"alive": True}