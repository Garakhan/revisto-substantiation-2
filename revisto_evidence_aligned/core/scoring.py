"""Scoring algorithms for evidence matching.

Search pipeline:
1. RETRIEVAL: Elasticsearch query using semantic similarity + numeric token boosting (3x)
2. SCORING: Use ES score directly (already includes numeric boost)

Note: Numeric and entity overlap are calculated and stored in breakdown for reference,
but the final score is the ES score which already includes numeric boosting.
"""

from typing import Dict, List, Any, Optional, Tuple

from rapidfuzz import fuzz

from ..config import ScoreConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ScoreCalculator:
    """Calculate relevance scores for evidence matching"""

    def __init__(self, score_config: ScoreConfig):
        self.score_config = score_config

    def calculate_score(
            self,
            es_score: float,
            numeric_overlap: float,
            entity_overlap: float
    ) -> float:
        """Calculate final relevance score.

        Uses ES score directly (which already includes numeric boosting from query).

        Note: numeric_overlap and entity_overlap are kept in breakdown for reference
        but not used in final score calculation.

        Args:
            es_score: Normalized Elasticsearch score (includes numeric boost) [0, 1]
            numeric_overlap: Jaccard overlap of numeric tokens (for reference only)
            entity_overlap: Jaccard overlap of biomedical entities (for reference only)

        Returns:
            ES score directly
        """
        # return (es_score + numeric_overlap) / 2.0
        return es_score

    def calculate_numeric_fuzzy_overlap(
        self,
        tokens1: List[str],
        tokens2: List[str]    ) -> float:
        """Calculate average fuzzy overlap between token lists"""
        if not tokens1 or not tokens2:
            return 0.0

        def find_candidate(token, candidates):
            best_score = 0
            bes_index = -1
            for idx, candidate in enumerate(candidates):
                score = fuzz.partial_ratio(token.lower(), candidate.lower())
                if score > best_score:
                    best_score = score
                    bes_index = idx
            return best_score, bes_index
        
        first, second = (tokens1, tokens2) if len(tokens1) <= len(tokens2) else (tokens2, tokens1)

        total_score = 0.0
        for token in first:
            score, idx = find_candidate(token, second)
            if idx >= 0:
                total_score += score

        return (total_score / len(second) / 100.0) if second else 0.0

    def calculate_overlap(
        self,
        tokens1: List[str],
        tokens2: List[str]
    ) -> float:
        """Calculate Jaccard overlap between token lists"""
        if not tokens1 or not tokens2:
            return 0.0

        set1 = set(t.lower() for t in tokens1)
        set2 = set(t.lower() for t in tokens2)

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def normalize_bm25_score(self, bm25_score: float) -> float:
        """Normalize BM25 score to [0, 1] range"""
        return bm25_score / (bm25_score + self.score_config.bm25_norm)

    def score_batch(
        self,
        claim: Dict[str, Any],
        candidates: List[Dict[str, Any]]
    ) -> List[Tuple[Dict[str, Any], float, Dict[str, float]]]:
        """Score a batch of candidates for a claim"""
        results = []

        # Extract claim features
        claim_numbers = set(t.lower() for t in claim.get("numeric_tokens", []))
        claim_entities = set(e.lower() for e in claim.get("entities", []))

        # Process candidates
        for candidate in candidates:
            # Get scores from ES result and normalize
            raw_es_score = candidate.get("_score", 0.0)
            es_score = self.normalize_bm25_score(raw_es_score)

            # Calculate overlaps
            cand_numbers = candidate.get("numeric_tokens", [])
            cand_entities = candidate.get("entities", [])

            # numeric_overlap = self.calculate_numeric_fuzzy_overlap(
            #     list(claim_numbers),
            #     cand_numbers
            # )
            numeric_overlap = self.calculate_overlap(
                list(claim_numbers),
                cand_numbers
            )

            entity_overlap = self.calculate_overlap(
                list(claim_entities),
                cand_entities
            )

            # Calculate final score
            final_score = self.calculate_score(
                es_score,
                numeric_overlap,
                entity_overlap
            )

            # Store detailed scores
            score_breakdown = {
                "es_score": es_score,
                "numeric": numeric_overlap,
                "entity": entity_overlap,
                "final_score": final_score
            }

            results.append((candidate, final_score, score_breakdown))

        # Sort by final score
        results.sort(key=lambda x: x[1], reverse=True)

        return results