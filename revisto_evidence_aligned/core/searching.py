"""Claim searching pipeline"""

from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import json

from ..config import get_config, Config
from ..nlp import load_embedder, extract_numbers, extract_entities, NERExtractor
from ..storage import ESClient
from ..core.scoring import ScoreCalculator
from ..utils.logging import get_logger

logger = get_logger(__name__)


def sentencize_text(text: str, sentenciser=None) -> List[str]:
    """Split text into sentences using Stanza or provided sentenciser."""
    if not text or not text.strip():
        return []

    if sentenciser:
        # Use gRPC sentenciser
        return sentenciser.sentencise(text)
    else:
        # Use local Stanza
        import stanza
        nlp = stanza.Pipeline("en", processors="tokenize", verbose=False)
        doc = nlp(text)
        return [s.text for s in doc.sentences]


def build_search_query(
        embedding: List[float],
        text: str,
        org_id: int,
        brand_id: int,
        numeric_tokens: List[str] = None,
        size: int = 50
) -> Dict[str, Any]:
    """Build ES query using lexical + semantic similarity with numeric boosting.

    Args:
        embedding: Vector embedding for semantic search
        text: Query text for lexical (BM25) search
        org_id: Organization ID filter
        brand_id: Brand ID filter
        numeric_tokens: List of numeric token strings from Stanza NER (e.g., ["42%", "51%"])
        size: Number of results to return

    Scoring formula:
        final_score = (lexical_weight * BM25) + (semantic_weight * cosine_sim) * numeric_boost
    """
    if numeric_tokens is None:
        numeric_tokens = []

    # Normalize tokens for comparison (lowercase, no spaces)
    normalized_tokens = [t.lower().replace(' ', '') for t in numeric_tokens]

    # Weights for combining lexical and semantic scores
    LEXICAL_WEIGHT = 0.4
    SEMANTIC_WEIGHT = 0.6
    NUMERIC_BOOST = 0.5  # Multiplier when numeric tokens match

    # Script for semantic similarity + numeric boost
    script_source = """
        double semantic = cosineSimilarity(params.embedding, 'vector') + 1.0;
        boolean numeric_match = false;

        if (doc.containsKey('numeric_tokens') && doc['numeric_tokens'].size() > 0) {
            for (def doc_token : doc['numeric_tokens']) {
                String normalized_doc = doc_token.toLowerCase().replace(' ', '');
                for (def claim_token : params.claim_numeric_tokens) {
                    if (normalized_doc.equals(claim_token)) {
                        numeric_match = true;
                        break;
                    }
                }
                if (numeric_match) break;
            }
        }

        double score = params.semantic_weight * semantic;
        if (numeric_match) {
            score = score * params.numeric_boost;
        }
        return score;
    """

    return {
        "query": {
            "function_score": {
                "query": {
                    "bool": {
                        "should": [
                            {
                                "match": {
                                    "text": {
                                        "query": text,
                                        "boost": LEXICAL_WEIGHT
                                    }
                                }
                            }
                        ],
                        "filter": [
                            {"term": {"doc_type": "segment"}},
                            {"term": {"org_id": org_id}},
                            {"term": {"brand_id": brand_id}}
                        ],
                        "minimum_should_match": 0  # Don't require lexical match
                    }
                },
                "functions": [
                    {
                        "script_score": {
                            "script": {
                                "source": script_source,
                                "params": {
                                    "embedding": embedding,
                                    "claim_numeric_tokens": normalized_tokens,
                                    "semantic_weight": SEMANTIC_WEIGHT,
                                    "numeric_boost": NUMERIC_BOOST
                                }
                            }
                        }
                    }
                ],
                "score_mode": "sum",  # How to combine multiple functions (we have one)
                "boost_mode": "sum"  # Add query score (lexical) + function score (semantic)
            }
        },
        "size": size,
        "_source": {"excludes": ["vector"]}
    }



def search_single_claim(
    claim: Dict[str, Any],
    org_id: int,
    brand_id: int,
    es_client: ESClient,
    config: Config,
    index_name: Optional[str] = None,
    embedder = None,
    ner_extractor: Optional[NERExtractor] = None,
    score_calculator: Optional[ScoreCalculator] = None,
    numeric_extractor = None
) -> List[Dict[str, Any]]:
    """Search for evidence matching a single claim"""

    claim_text = claim.get("text", claim.get("claim_text", ""))

    if not claim_text:
        logger.warning("Empty claim text")
        return []

    # Generate embedding
    embedding = embedder.encode_single(claim_text).tolist()

    # Extract numeric tokens using Stanza NER (for query boosting and re-ranking)
    if numeric_extractor:
        numeric_tokens = numeric_extractor.extract(claim_text)
    else:
        numeric_tokens = extract_numbers(claim_text)

    # Extract entities for re-ranking
    entities = []
    if ner_extractor and config.ner.enabled:
        ent_results = ner_extractor.extract(claim_text)
        entities = [e["text"] for e in ent_results]

    logger.info(f"Embedded and extracted: entities={entities}, numeric_tokens={numeric_tokens}")

    # ==========================================================
    # STAGE 1: RETRIEVAL (semantic similarity with numeric boosting)
    # Retrieve more candidates than needed, then re-rank
    # ==========================================================
    retrieval_size = max(config.score.topk * 3, 50)  # Get 3x candidates for re-ranking

    query = build_search_query(
        embedding=embedding,
        text=claim_text,
        org_id=org_id,
        brand_id=brand_id,
        numeric_tokens=numeric_tokens,
        size=retrieval_size
    )

    target_index = index_name or config.es.index_name

    try:
        response = es_client.client.search(
            index=target_index,
            body=query
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        return []

    # Extract candidates with ES scores
    candidates = []
    for hit in response["hits"]["hits"]:
        candidate = hit["_source"]
        candidate["_score"] = hit["_score"]
        candidates.append(candidate)

    logger.info(f"Stage 1 (retrieval): {len(candidates)} candidates")

    # ==========================================================
    # STAGE 2: RE-RANKING (numeric + entity overlap)
    # ==========================================================
    if score_calculator is None:
        score_calculator = ScoreCalculator(config.score)

    # Prepare claim for scoring
    scoring_claim = {
        "text": claim_text,
        "numeric_tokens": numeric_tokens,
        "entities": entities
    }

    # Re-rank candidates using numeric + entity overlap
    scored_results = score_calculator.score_batch(scoring_claim, candidates)

    # Filter by threshold, minimum word count, and limit to topk
    filtered_results = []
    min_words = config.score.min_evidence_words

    for candidate, score, breakdown in scored_results:
        if score >= config.score.threshold:
            text = candidate.get("text", "")
            word_count = len(text.split())
            if word_count < min_words:
                logger.debug(f"Skipping short evidence ({word_count} words): {text[:50]}...")
                continue

            result = {
                **candidate,
                "relevance_score": score,
                "score_breakdown": breakdown
            }
            filtered_results.append(result)

            if len(filtered_results) >= config.score.topk:
                break

    logger.info(f"Stage 2 (re-ranking): {len(filtered_results)} results after filtering")

    # Attach doc_metadata from separate metadata documents
    if filtered_results:
        ref_ids = list(set(r.get("ref_id") for r in filtered_results if r.get("ref_id")))
        if ref_ids:
            metadata_map = es_client.get_ref_metadata(target_index, ref_ids)
            for result in filtered_results:
                ref_id = result.get("ref_id")
                if ref_id and ref_id in metadata_map:
                    result["doc_metadata"] = metadata_map[ref_id]

    return filtered_results


def search_claim_by_sentences(
    claim: Dict[str, Any],
    org_id: int,
    brand_id: int,
    es_client: ESClient,
    config: Config,
    index_name: Optional[str] = None,
    embedder=None,
    ner_extractor: Optional[NERExtractor] = None,
    score_calculator: Optional[ScoreCalculator] = None,
    numeric_extractor=None,
    sentenciser=None
) -> List[Dict[str, Any]]:
    """Search for evidence matching each sentence of a claim.

    Returns a list of results, one per sentence, each containing:
    - sentence_index: 0-based index of the sentence
    - sentence_text: the sentence text
    - evidence: list of evidence results for this sentence
    """
    claim_text = claim.get("text", claim.get("claim_text", ""))

    if not claim_text:
        logger.warning("Empty claim text")
        return []

    # Sentencize the claim
    sentences = sentencize_text(claim_text, sentenciser)

    if not sentences:
        logger.warning("No sentences extracted from claim")
        return []

    logger.info(f"Split claim into {len(sentences)} sentences")

    # Search for each sentence
    sentence_results = []
    for idx, sentence in enumerate(sentences):
        # Skip very short sentences
        if len(sentence.strip()) < 5:
            continue

        # Create a claim dict for this sentence
        sentence_claim = {"text": sentence}

        # Search for this sentence
        evidence = search_single_claim(
            claim=sentence_claim,
            org_id=org_id,
            brand_id=brand_id,
            es_client=es_client,
            config=config,
            index_name=index_name,
            embedder=embedder,
            ner_extractor=ner_extractor,
            score_calculator=score_calculator,
            numeric_extractor=numeric_extractor
        )

        sentence_results.append({
            "sentence_index": idx,
            "sentence_text": sentence,
            "evidence": evidence,
            "evidence_count": len(evidence)
        })

    return sentence_results


def deduplicate_evidence(evidence_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate evidence based on ref_id, page, paragraph, sentence.

    Keeps the first occurrence (typically the higher-scored one if pre-sorted).
    """
    seen = set()
    unique = []

    for ev in evidence_list:
        key = (
            ev.get("ref_id"),
            ev.get("page"),
            ev.get("paragraph_number"),
            ev.get("sentence_number")
        )
        if key not in seen:
            seen.add(key)
            unique.append(ev)

    return unique


def search_claim_combined(
    claim: Dict[str, Any],
    org_id: int,
    brand_id: int,
    es_client: ESClient,
    config: Config,
    index_name: Optional[str] = None,
    embedder=None,
    ner_extractor: Optional[NERExtractor] = None,
    score_calculator: Optional[ScoreCalculator] = None,
    numeric_extractor=None,
    sentenciser=None
) -> List[Dict[str, Any]]:
    """Search for evidence using both block-level and sentence-level queries.

    Performs two searches:
    1. Block search: Full claim text as a single query
    2. Sentence search: Each sentence of the claim searched separately

    Results are merged and deduplicated by (ref_id, page, paragraph, sentence).
    Higher-scored duplicates are kept.

    Returns:
        List of evidence results with sentence attribution where applicable.
    """
    claim_text = claim.get("text", claim.get("claim_text", ""))

    if not claim_text:
        logger.warning("Empty claim text")
        return []

    # --- Block-level search (full claim) ---
    logger.info("Searching block-level (full claim)...")
    block_results = search_single_claim(
        claim=claim,
        org_id=org_id,
        brand_id=brand_id,
        es_client=es_client,
        config=config,
        index_name=index_name,
        embedder=embedder,
        ner_extractor=ner_extractor,
        score_calculator=score_calculator,
        numeric_extractor=numeric_extractor
    )

    # Mark block results with source
    for ev in block_results:
        ev["search_source"] = "block"
        ev["source_sentence_index"] = None
        ev["source_sentence_text"] = None

    logger.info(f"  Block search found {len(block_results)} results")

    # --- Sentence-level search ---
    logger.info("Searching sentence-level...")
    sentence_results = search_claim_by_sentences(
        claim=claim,
        org_id=org_id,
        brand_id=brand_id,
        es_client=es_client,
        config=config,
        index_name=index_name,
        embedder=embedder,
        ner_extractor=ner_extractor,
        score_calculator=score_calculator,
        numeric_extractor=numeric_extractor,
        sentenciser=sentenciser
    )

    # Flatten sentence results and mark with source
    sentence_evidence = []
    for sent_result in sentence_results:
        sent_idx = sent_result.get("sentence_index", 0)
        sent_text = sent_result.get("sentence_text", "")
        for ev in sent_result.get("evidence", []):
            ev["search_source"] = "sentence"
            ev["source_sentence_index"] = sent_idx
            ev["source_sentence_text"] = sent_text
            sentence_evidence.append(ev)

    logger.info(f"  Sentence search found {len(sentence_evidence)} results")

    # --- Merge and deduplicate ---
    # Combine all results, sort by score (descending), then deduplicate
    all_results = block_results + sentence_evidence
    all_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

    deduplicated = deduplicate_evidence(all_results)

    logger.info(f"  Combined: {len(all_results)} -> {len(deduplicated)} after deduplication")

    # Limit to topk
    topk = config.score.topk
    final_results = deduplicated[:topk]

    return final_results


def search_claim_combined_by_sentences(
    claim: Dict[str, Any],
    org_id: int,
    brand_id: int,
    es_client: ESClient,
    config: Config,
    index_name: Optional[str] = None,
    embedder=None,
    ner_extractor: Optional[NERExtractor] = None,
    score_calculator: Optional[ScoreCalculator] = None,
    numeric_extractor=None,
    sentenciser=None
) -> List[Dict[str, Any]]:
    """Search using combined approach but return results grouped by sentence.

    This maintains compatibility with search_claim_by_sentences output format,
    but includes block-level results distributed to relevant sentences.

    Returns:
        List of sentence results, each with evidence list.
    """
    claim_text = claim.get("text", claim.get("claim_text", ""))

    if not claim_text:
        logger.warning("Empty claim text")
        return []

    # Get sentences
    sentences = sentencize_text(claim_text, sentenciser)
    if not sentences:
        logger.warning("No sentences extracted from claim")
        return []

    # --- Block-level search (full claim) ---
    logger.info("Searching block-level (full claim)...")
    block_results = search_single_claim(
        claim=claim,
        org_id=org_id,
        brand_id=brand_id,
        es_client=es_client,
        config=config,
        index_name=index_name,
        embedder=embedder,
        ner_extractor=ner_extractor,
        score_calculator=score_calculator,
        numeric_extractor=numeric_extractor
    )
    logger.info(f"  Block search found {len(block_results)} results")

    # --- Sentence-level search ---
    logger.info("Searching sentence-level...")
    sentence_results = []
    for idx, sentence in enumerate(sentences):
        if len(sentence.strip()) < 5:
            continue

        sentence_claim = {"text": sentence}
        evidence = search_single_claim(
            claim=sentence_claim,
            org_id=org_id,
            brand_id=brand_id,
            es_client=es_client,
            config=config,
            index_name=index_name,
            embedder=embedder,
            ner_extractor=ner_extractor,
            score_calculator=score_calculator,
            numeric_extractor=numeric_extractor
        )

        # Mark evidence source
        for ev in evidence:
            ev["search_source"] = "sentence"

        sentence_results.append({
            "sentence_index": idx,
            "sentence_text": sentence,
            "evidence": evidence,
            "evidence_count": len(evidence)
        })

    # --- Add block results to each sentence (will be deduplicated later) ---
    # Block results apply to the claim as a whole, so add to all sentences
    # but mark them as block-sourced
    for sent_result in sentence_results:
        block_evidence_copy = []
        for ev in block_results:
            ev_copy = ev.copy()
            ev_copy["search_source"] = "block"
            block_evidence_copy.append(ev_copy)

        # Merge: sentence evidence first (higher specificity), then block
        combined = sent_result["evidence"] + block_evidence_copy
        combined.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        # Deduplicate within this sentence's evidence
        deduplicated = deduplicate_evidence(combined)

        # Limit to topk per sentence
        sent_result["evidence"] = deduplicated[:config.score.topk]
        sent_result["evidence_count"] = len(sent_result["evidence"])

    total_evidence = sum(s["evidence_count"] for s in sentence_results)
    logger.info(f"  Combined sentence results: {len(sentence_results)} sentences, {total_evidence} total evidence")

    return sentence_results


def search_claims(
    claims_file: Path,
    org_id: int,
    brand_id: int,
    output_file: Optional[Path] = None,
    es_client: Optional[ESClient] = None,
    config: Optional[Config] = None,
    index_name: Optional[str] = None,
    embedder=None,
    ner_extractor=None,
    numeric_extractor=None
) -> Dict[str, Any]:
    """Search for evidence matching claims from a file"""

    if config is None:
        config = get_config()

    if es_client is None:
        es_client = ESClient(config.es)

    # Load claims
    with open(claims_file) as f:
        claims_data = json.load(f)

    # Handle different claim formats
    if isinstance(claims_data, dict):
        claims = claims_data.get("claims", [])
    else:
        claims = claims_data

    logger.info(f"Loaded {len(claims)} claims from {claims_file}")

    # Load models only if not provided (allows gRPC models to be passed in)
    if embedder is None:
        embedder = load_embedder(config.embed)
    if ner_extractor is None and config.ner.enabled:
        ner_extractor = NERExtractor(config.ner)
    score_calculator = ScoreCalculator(config.score)
    
    # Process claims
    all_results = []
    claim_count = 0
    match_count = 0
    
    for claim in claims:
        claim_count += 1
        
        # Search for evidence
        results = search_single_claim(
            claim=claim,
            org_id=org_id,
            brand_id=brand_id,
            es_client=es_client,
            config=config,
            index_name=index_name,
            embedder=embedder,
            ner_extractor=ner_extractor,
            score_calculator=score_calculator,
            numeric_extractor=numeric_extractor
        )
        
        if results:
            match_count += 1
            
        # Store results
        claim_results = {
            "claim": claim,
            "evidence": results,
            "evidence_count": len(results)
        }
        all_results.append(claim_results)
        
        if claim_count % 10 == 0:
            logger.info(f"Processed {claim_count}/{len(claims)} claims")
    
    # Summary statistics
    summary = {
        "total_claims": claim_count,
        "claims_with_evidence": match_count,
        "coverage": match_count / claim_count if claim_count > 0 else 0,
        "avg_evidence_per_claim": sum(r["evidence_count"] for r in all_results) / claim_count if claim_count > 0 else 0
    }
    
    # Final results
    output = {
        "summary": summary,
        "results": all_results,
        "config": {
            "org_id": org_id,
            "brand_id": brand_id,
            "threshold": config.score.threshold,
            "topk": config.score.topk
        }
    }
    
    # Save results if visual_claims file specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results saved to {output_file}")
    
    return output