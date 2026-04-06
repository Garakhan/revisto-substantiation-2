"""Request models for API"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict


class ClaimInput(BaseModel):
    """Single claim for searching"""
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., description="The claim text to search for evidence")


class IndexRequest(BaseModel):
    """Request for indexing reference documents"""
    model_config = ConfigDict(extra="forbid")
    
    org_id: int = Field(..., description="Organization ID", gt=0)
    brand_id: int = Field(..., description="Brand ID", gt=0)
    batch_size: int = Field(500, description="Batch size for indexing", gt=0, le=5000)
    enable_ner: bool = Field(True, description="Enable NER extraction")
    segmentation_level: Literal["sentence", "block"] = Field(
        "sentence",
        description="Segmentation level: 'sentence' for fine-grained, 'block' for paragraph-level"
    )


class SearchRequest(BaseModel):
    """Request for searching a single claim"""
    model_config = ConfigDict(extra="forbid")
    
    claim: ClaimInput = Field(..., description="Claim to search for evidence")
    org_id: int = Field(..., description="Organization ID", gt=0)
    brand_id: int = Field(..., description="Brand ID", gt=0)
    index_name: Optional[str] = Field(None, description="Custom index name. If not provided, uses default from config")
    threshold: float = Field(0.4, description="Minimum relevance score", ge=0.0, le=1.0)
    top_k: int = Field(20, description="Maximum number of results", gt=0, le=100)
    enable_ner: bool = Field(True, description="Enable NER extraction")


class BatchSearchRequest(BaseModel):
    """Request for searching multiple claims"""
    model_config = ConfigDict(extra="forbid")
    
    claims: List[ClaimInput] = Field(..., description="List of claims to search")
    org_id: int = Field(..., description="Organization ID", gt=0)
    brand_id: int = Field(..., description="Brand ID", gt=0)
    index_name: Optional[str] = Field(None, description="Custom index name. If not provided, uses default from config")
    threshold: float = Field(0.4, description="Minimum relevance score", ge=0.0, le=1.0)
    top_k: int = Field(20, description="Maximum results per claim", gt=0, le=100)
    enable_ner: bool = Field(True, description="Enable NER extraction")
    parallel: bool = Field(True, description="Process claims in parallel")