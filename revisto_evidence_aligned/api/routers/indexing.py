"""Indexing endpoints"""

import os
import time
from pathlib import Path
from typing import List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from tempfile import TemporaryDirectory

from ..dependencies import get_resources
from ..models import IndexRequest, IndexResponse
from ...core import index_references
from ...core.indexing import SegmentationLevel
from ...utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Default cache directory for parsed documents
PARSE_CACHE_DIR = os.getenv("PARSE_CACHE_DIR", "/app/data/parse_cache")


@router.post("/references", response_model=IndexResponse)
async def index_reference_files(
    files: List[UploadFile] = File(..., description="PDF files to index"),
    org_id: int = Form(..., description="Organization ID", gt=0),
    brand_id: int = Form(..., description="Brand ID", gt=0),
    index_name: str = Form(None, description="Custom index name. If not provided, uses default from config"),
    batch_size: int = Form(500, description="Batch size for indexing", gt=0, le=5000),
    enable_ner: bool = Form(True, description="Enable NER extraction"),
    index_text: bool = Form(True, description="Index text content"),
    index_tables: bool = Form(True, description="Linearize and index table content"),
    index_figures: bool = Form(True, description="Process and index figure content"),
    segmentation_level: SegmentationLevel = Form("sentence", description="Segmentation level: 'sentence' for fine-grained, 'block' for paragraph-level")
):
    """Index reference PDF files"""
    start_time = time.time()
    resources = get_resources()

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # Validate file types
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.filename}. Only PDF files are supported."
            )

    # Create request-scoped config copy
    config = resources["config"]
    if not enable_ner:
        from dataclasses import replace
        config = replace(config, ner=replace(config.ner, enabled=False))

    # Get table linearizer only if indexing tables (linearization + indexing are coupled)
    table_linearizer = resources.get("table_linearizer") if index_tables else None

    # Get figure processor only if indexing figures (processing + indexing are coupled)
    figure_processor = resources.get("figure_processor") if index_figures else None

    # Process files in temporary directory
    with TemporaryDirectory() as temp_dir:
        pdf_paths = []

        # Save uploaded files
        for file in files:
            file_path = Path(temp_dir) / file.filename
            content = await file.read()

            with open(file_path, 'wb') as f:
                f.write(content)

            pdf_paths.append(file_path)
            logger.info(f"Saved uploaded file: {file.filename}")

        # Index files
        try:
            results = await index_references(
                pdf_paths=pdf_paths,
                org_id=org_id,
                brand_id=brand_id,
                es_client=resources["es_client"],
                config=config,
                index_name=index_name,
                batch_size=batch_size,
                segmentation_level=segmentation_level,
                cache_dir=PARSE_CACHE_DIR,
                embedder=resources.get("embedder"),
                ner_extractor=resources.get("ner_extractor"),
                sentenciser=resources.get("sentenciser"),
                table_linearizer=table_linearizer,
                figure_processor=figure_processor,
                numeric_extractor=resources.get("numeric_extractor"),
                index_text=index_text,
                index_tables=index_tables,
                index_figures=index_figures
            )

            processing_time = (time.time() - start_time) * 1000

            return IndexResponse(
                status="success",
                documents_processed=results["document_count"],
                documents_indexed=results["index_stats"]["success"],
                errors=results["index_stats"]["errors"],
                processing_time_ms=processing_time,
                index_name=index_name or config.es.index_name
            )

        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@router.post("/references/async", status_code=202)
async def index_references_async(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="PDF files to index"),
    org_id: int = Form(..., description="Organization ID", gt=0),
    brand_id: int = Form(..., description="Brand ID", gt=0),
    index_name: str = Form(None, description="Custom index name. If not provided, uses default from config"),
    batch_size: int = Form(500, description="Batch size for indexing", gt=0, le=5000),
    enable_ner: bool = Form(True, description="Enable NER extraction"),
    index_text: bool = Form(True, description="Index text content"),
    index_tables: bool = Form(True, description="Linearize and index table content"),
    index_figures: bool = Form(True, description="Process and index figure content"),
    segmentation_level: SegmentationLevel = Form("sentence", description="Segmentation level: 'sentence' for fine-grained, 'block' for paragraph-level")
):
    """Index reference files asynchronously"""
    # TODO: Implement proper async job tracking with job IDs

    # For now, just acknowledge the request
    return {
        "message": "Indexing job accepted",
        "job_id": f"job_{int(time.time())}",
        "files": [f.filename for f in files]
    }


@router.get("/status/{job_id}")
async def get_indexing_status(job_id: str):
    """Get status of an indexing job"""
    # TODO: Implement job status tracking

    return {
        "job_id": job_id,
        "status": "pending",
        "message": "Job status tracking not yet implemented"
    }
