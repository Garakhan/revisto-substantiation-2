"""Response models for API"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class EvidenceResult(BaseModel):
    """Single evidence result"""
    model_config = ConfigDict(extra="allow")

    ref_id: str = Field(..., description="Reference document ID")
    ref_title: str = Field(..., description="Reference document title")
    text: str = Field(..., description="Evidence text")
    relevance_score: float = Field(..., description="Overall relevance score")
    page: int = Field(..., description="Page number in reference")
    table_source: Optional[str] = Field(None, description="Table location (e.g., 'Page 5, Table 1; row 1; column 1') or None for text")
    paragraph_number: Optional[int] = Field(None, description="Paragraph number within the page")
    sentence_number: Optional[int] = Field(None, description="Sentence number within the paragraph")
    bbox: Optional[Dict[str, float]] = Field(None, description="Bounding box coordinates")
    score_breakdown: Dict[str, float] = Field(..., description="Detailed score components")
    entities: List[str] = Field(default_factory=list, description="Extracted entities")
    numeric_tokens: List[str] = Field(default_factory=list, description="Extracted numbers")


class SearchResponse(BaseModel):
    """Response for search requests"""
    model_config = ConfigDict(extra="forbid")

    evidence: List[EvidenceResult] = Field(..., description="List of evidence results")
    evidence_count: int = Field(..., description="Number of evidence found")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class SentenceSearchResult(BaseModel):
    """Search results for a single sentence"""
    model_config = ConfigDict(extra="forbid")

    sentence_index: int = Field(..., description="0-based index of the sentence in the claim")
    sentence_text: str = Field(..., description="The sentence text")
    evidence: List[EvidenceResult] = Field(..., description="List of evidence results for this sentence")
    evidence_count: int = Field(..., description="Number of evidence found for this sentence")


class SentenceSearchResponse(BaseModel):
    """Response for sentence-level search requests"""
    model_config = ConfigDict(extra="forbid")

    sentence_results: List[SentenceSearchResult] = Field(..., description="Results for each sentence")
    total_sentences: int = Field(..., description="Total number of sentences in claim")
    total_evidence: int = Field(..., description="Total evidence across all sentences")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class BatchSearchResponse(BaseModel):
    """Response for batch search requests"""
    model_config = ConfigDict(extra="forbid")

    results: List[SearchResponse] = Field(..., description="Results for each claim")
    summary: Dict[str, Any] = Field(..., description="Summary statistics")
    total_processing_time_ms: float = Field(..., description="Total processing time")


class IndexResponse(BaseModel):
    """Response for indexing requests"""
    model_config = ConfigDict(extra="forbid")
    
    status: str = Field(..., description="Indexing status")
    documents_processed: int = Field(..., description="Number of documents processed")
    documents_indexed: int = Field(..., description="Number of documents indexed")
    errors: int = Field(0, description="Number of errors")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    index_name: str = Field(..., description="Elasticsearch index name")


class HealthResponse(BaseModel):
    """Health check response"""
    model_config = ConfigDict(extra="forbid")
    
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(..., description="API version")
    elasticsearch: Dict[str, Any] = Field(..., description="Elasticsearch status")
    models_loaded: Dict[str, bool] = Field(..., description="Model loading status")


class ErrorResponse(BaseModel):
    """Error response"""
    model_config = ConfigDict(extra="forbid")
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = Field(None, description="Request ID for tracking")