"""Core business logic modules"""

from .indexing import index_references, index_reference_directory
from .searching import (
    search_claims,
    search_single_claim,
    search_claim_by_sentences,
    search_claim_combined,
    search_claim_combined_by_sentences,
    deduplicate_evidence,
)
from .scoring import ScoreCalculator

__all__ = [
    "index_references",
    "index_reference_directory",
    "search_claims",
    "search_single_claim",
    "search_claim_by_sentences",
    "search_claim_combined",
    "search_claim_combined_by_sentences",
    "deduplicate_evidence",
    "ScoreCalculator",
]