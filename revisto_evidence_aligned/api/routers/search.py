"""Search endpoints"""

import time
import asyncio
from dataclasses import replace
from typing import List
from fastapi import APIRouter, HTTPException

from ..dependencies import get_resources
from ..models import (
    SearchRequest,
    SearchResponse,
    SentenceSearchResult,
    SentenceSearchResponse,
    BatchSearchRequest,
    BatchSearchResponse,
    EvidenceResult,
)
from ...core import search_single_claim, search_claim_by_sentences, search_claim_combined_by_sentences
from ...core.scoring import ScoreCalculator
from ...utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


def create_request_config(base_config, request: SearchRequest):
    """Create a request-scoped config copy with overridden parameters.

    This avoids mutating the shared config object which is unsafe
    in concurrent/multi-threaded contexts.
    """
    # Create copies with overridden values
    score_config = replace(
        base_config.score,
        threshold=request.threshold,
        topk=request.top_k
    )
    ner_config = replace(
        base_config.ner,
        enabled=request.enable_ner
    )
    # Create new config with updated sub-configs
    return replace(base_config, score=score_config, ner=ner_config)


@router.post("/claim", response_model=SearchResponse)
async def search_claim(request: SearchRequest):
    """Search for evidence matching a single claim"""
    logger.info(f"Started search")
    start_time = time.time()
    resources = get_resources()

    # Create request-scoped config (thread-safe, doesn't mutate shared config)
    config = create_request_config(resources["config"], request)

    # Get resources
    es_client = resources["es_client"]
    embedder = resources["embedder"]
    ner_extractor = resources.get("ner_extractor") if request.enable_ner else None
    score_calculator = ScoreCalculator(config.score)
    
    # Convert claim to dict format
    claim_dict = request.claim.model_dump(by_alias=True)

    logger.info(f"Collected metadata")
    
    try:
        # Search for evidence
        results = search_single_claim(
            claim=claim_dict,
            org_id=request.org_id,
            brand_id=request.brand_id,
            es_client=es_client,
            config=config,
            index_name=request.index_name,
            embedder=embedder,
            ner_extractor=ner_extractor,
            score_calculator=score_calculator,
            numeric_extractor=resources.get("numeric_extractor")
        )
        
        # Convert to response models
        evidence_results = []
        for result in results:
            evidence = EvidenceResult(
                ref_id=result["ref_id"],
                ref_title=result["ref_title"],
                text=result["text"],
                relevance_score=result["relevance_score"],
                page=result["page"],
                table_source=result.get("table_source"),
                paragraph_number=result.get("paragraph_number"),
                sentence_number=result.get("sentence_number"),
                bbox=result.get("bbox"),
                score_breakdown=result["score_breakdown"],
                entities=result.get("entities", []),
                numeric_tokens=result.get("numeric_tokens", [])
            )
            evidence_results.append(evidence)
        
        processing_time = (time.time() - start_time) * 1000

        return SearchResponse(
            evidence=evidence_results,
            evidence_count=len(evidence_results),
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/claim/sentences", response_model=SentenceSearchResponse)
async def search_claim_sentences(request: SearchRequest):
    """Search for evidence matching each sentence of a claim separately.

    The claim is first split into sentences using Stanza, then each sentence
    is searched independently. Results are returned per sentence.
    """
    logger.info(f"Started sentence-level search")
    start_time = time.time()
    resources = get_resources()

    # Create request-scoped config (thread-safe, doesn't mutate shared config)
    config = create_request_config(resources["config"], request)

    # Get resources
    es_client = resources["es_client"]
    embedder = resources["embedder"]
    ner_extractor = resources.get("ner_extractor") if request.enable_ner else None
    score_calculator = ScoreCalculator(config.score)
    sentenciser = resources.get("sentenciser")

    # Convert claim to dict format
    claim_dict = request.claim.model_dump(by_alias=True)

    logger.info(f"Collected metadata for sentence-level search")

    try:
        # Search for evidence by sentences
        results = search_claim_by_sentences(
            claim=claim_dict,
            org_id=request.org_id,
            brand_id=request.brand_id,
            es_client=es_client,
            config=config,
            index_name=request.index_name,
            embedder=embedder,
            ner_extractor=ner_extractor,
            score_calculator=score_calculator,
            numeric_extractor=resources.get("numeric_extractor"),
            sentenciser=sentenciser
        )

        # Convert to response models
        sentence_results = []
        total_evidence = 0
        for result in results:
            evidence_list = []
            for ev in result["evidence"]:
                evidence = EvidenceResult(
                    ref_id=ev["ref_id"],
                    ref_title=ev["ref_title"],
                    text=ev["text"],
                    relevance_score=ev["relevance_score"],
                    page=ev["page"],
                    table_source=ev.get("table_source"),
                    paragraph_number=ev.get("paragraph_number"),
                    sentence_number=ev.get("sentence_number"),
                    bbox=ev.get("bbox"),
                    score_breakdown=ev["score_breakdown"],
                    entities=ev.get("entities", []),
                    numeric_tokens=ev.get("numeric_tokens", [])
                )
                evidence_list.append(evidence)

            sentence_result = SentenceSearchResult(
                sentence_index=result["sentence_index"],
                sentence_text=result["sentence_text"],
                evidence=evidence_list,
                evidence_count=result["evidence_count"]
            )
            sentence_results.append(sentence_result)
            total_evidence += result["evidence_count"]

        processing_time = (time.time() - start_time) * 1000

        return SentenceSearchResponse(
            sentence_results=sentence_results,
            total_sentences=len(sentence_results),
            total_evidence=total_evidence,
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Sentence search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Sentence search failed: {str(e)}")


@router.post("/claim/combined", response_model=SentenceSearchResponse)
async def search_claim_combined(request: SearchRequest):
    """Search for evidence using combined block and sentence-level queries.

    Performs two searches:
    1. Block search: Full claim text as a single query
    2. Sentence search: Each sentence of the claim searched separately

    Results are merged and deduplicated by (ref_id, page, paragraph, sentence).
    Higher-scored duplicates are kept. Returns results grouped by sentence.
    """
    logger.info(f"Started combined (block + sentence) search")
    start_time = time.time()
    resources = get_resources()

    # Create request-scoped config (thread-safe, doesn't mutate shared config)
    config = create_request_config(resources["config"], request)

    # Get resources
    es_client = resources["es_client"]
    embedder = resources["embedder"]
    ner_extractor = resources.get("ner_extractor") if request.enable_ner else None
    score_calculator = ScoreCalculator(config.score)
    sentenciser = resources.get("sentenciser")

    # Convert claim to dict format
    claim_dict = request.claim.model_dump(by_alias=True)

    logger.info(f"Collected metadata for combined search")

    try:
        # Search using combined approach
        results = search_claim_combined_by_sentences(
            claim=claim_dict,
            org_id=request.org_id,
            brand_id=request.brand_id,
            es_client=es_client,
            config=config,
            index_name=request.index_name,
            embedder=embedder,
            ner_extractor=ner_extractor,
            score_calculator=score_calculator,
            numeric_extractor=resources.get("numeric_extractor"),
            sentenciser=sentenciser
        )

        # Convert to response models
        sentence_results = []
        total_evidence = 0
        for result in results:
            evidence_list = []
            for ev in result["evidence"]:
                evidence = EvidenceResult(
                    ref_id=ev["ref_id"],
                    ref_title=ev["ref_title"],
                    text=ev["text"],
                    relevance_score=ev["relevance_score"],
                    page=ev["page"],
                    table_source=ev.get("table_source"),
                    paragraph_number=ev.get("paragraph_number"),
                    sentence_number=ev.get("sentence_number"),
                    bbox=ev.get("bbox"),
                    score_breakdown=ev["score_breakdown"],
                    entities=ev.get("entities", []),
                    numeric_tokens=ev.get("numeric_tokens", [])
                )
                evidence_list.append(evidence)

            sentence_result = SentenceSearchResult(
                sentence_index=result["sentence_index"],
                sentence_text=result["sentence_text"],
                evidence=evidence_list,
                evidence_count=result["evidence_count"]
            )
            sentence_results.append(sentence_result)
            total_evidence += result["evidence_count"]

        processing_time = (time.time() - start_time) * 1000

        return SentenceSearchResponse(
            sentence_results=sentence_results,
            total_sentences=len(sentence_results),
            total_evidence=total_evidence,
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Combined search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Combined search failed: {str(e)}")


@router.post("/batch", response_model=BatchSearchResponse)
async def search_batch(request: BatchSearchRequest):
    """Search for evidence matching multiple claims"""
    start_time = time.time()
    
    if not request.claims:
        raise HTTPException(status_code=400, detail="No claims provided")
    
    if len(request.claims) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 claims per batch")
    
    # Process claims
    if request.parallel:
        # Process in parallel using asyncio
        tasks = []
        for claim in request.claims:
            search_req = SearchRequest(
                claim=claim,
                org_id=request.org_id,
                brand_id=request.brand_id,
                index_name=request.index_name,
                threshold=request.threshold,
                top_k=request.top_k,
                enable_ner=request.enable_ner
            )
            tasks.append(search_claim(search_req))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results and exceptions
        search_results = []
        errors = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing claim {i}: {result}")
                errors += 1
                # Add empty result
                search_results.append(SearchResponse(
                    evidence=[],
                    evidence_count=0,
                    processing_time_ms=0
                ))
            else:
                search_results.append(result)
    else:
        # Process sequentially
        search_results = []
        errors = 0
        for i, claim in enumerate(request.claims):
            try:
                search_req = SearchRequest(
                    claim=claim,
                    org_id=request.org_id,
                    brand_id=request.brand_id,
                    threshold=request.threshold,
                    top_k=request.top_k,
                    enable_ner=request.enable_ner
                )
                result = await search_claim(search_req)
                search_results.append(result)
            except Exception as e:
                logger.error(f"Error processing claim {i}: {e}")
                errors += 1
                search_results.append(SearchResponse(
                    evidence=[],
                    evidence_count=0,
                    processing_time_ms=0
                ))
    
    # Calculate summary
    total_claims = len(request.claims)
    claims_with_evidence = sum(1 for r in search_results if r.evidence_count > 0)
    total_evidence = sum(r.evidence_count for r in search_results)
    
    summary = {
        "total_claims": total_claims,
        "claims_with_evidence": claims_with_evidence,
        "total_evidence": total_evidence,
        "coverage": claims_with_evidence / total_claims if total_claims > 0 else 0,
        "avg_evidence_per_claim": total_evidence / total_claims if total_claims > 0 else 0,
        "errors": errors
    }
    
    processing_time = (time.time() - start_time) * 1000
    
    return BatchSearchResponse(
        results=search_results,
        summary=summary,
        total_processing_time_ms=processing_time
    )