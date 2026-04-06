#!/usr/bin/env python3
"""Script to substantiate claims from nexobrid_prod.csv"""

import csv
import json
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from revisto_evidence_aligned.config import get_config
from revisto_evidence_aligned.core.searching import search_single_claim
from revisto_evidence_aligned.nlp import load_embedder, NERExtractor
from revisto_evidence_aligned.core.scoring import ScoreCalculator
from revisto_evidence_aligned.storage import ESClient
from revisto_evidence_aligned.utils.logging import get_logger
from revisto_evidence_aligned.cli.search_formatted import (
    format_all_evidence,
    get_page_paragraph_offsets,
    ClaudeEvidenceFilter
)
import asyncio

logger = get_logger(__name__)

# Configuration
ORG_ID = 1
BRAND_ID = 1
INDEX_NAME = "maci_index"
ES_URL = "http://localhost:9200"  # Local Elasticsearch


def parse_claim_row(row: dict) -> dict:
    """Parse a row from nexobrid_prod.csv and extract claim info."""
    properties_json = row.get("properties", "[]")
    identifier = row.get("identifier", "")

    try:
        properties = json.loads(properties_json)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse properties JSON for {identifier}")
        properties = []

    # Extract claim text and category from properties
    claim_text = ""
    category = ""

    for prop in properties:
        label = prop.get("label", "").lower()
        value = prop.get("value", "")

        if "product claim" in label and not claim_text:
            claim_text = value
        elif label == "category":
            category = value

    return {
        "claim_id": identifier,
        "text": claim_text,
        "category": category
    }


def extract_unique_files(evidence_list: list) -> str:
    """Extract unique file names from evidence list, one per line."""
    files = []
    seen = set()

    for ev in evidence_list:
        file_name = ev.get("ref_id", "")
        if file_name and file_name not in seen:
            files.append(file_name)
            seen.add(file_name)

    return "\n".join(files)


def substantiate_claims(
    input_file: Path,
    output_file: Path,
    use_llm_filter: bool = False
):
    """Process claims and generate substantiation visual_claims."""

    config = get_config()
    config.es.index_name = INDEX_NAME
    config.es.url = ES_URL
    config.es.api_key = None  # Don't use API key for local ES
    config.es.verify_certs = False

    # Connect to Elasticsearch
    logger.info(f"Connecting to Elasticsearch at {ES_URL}...")
    es_client = ESClient(config.es)

    # Load models
    logger.info("Loading models...")
    embedder = load_embedder(config.embed)
    ner_extractor = NERExtractor(config.ner) if config.ner.enabled else None
    score_calculator = ScoreCalculator(config.score)

    # Read input CSV
    logger.info(f"Reading claims from {input_file}...")
    claims = []

    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            claim = parse_claim_row(row)
            if claim["text"]:  # Skip empty claims
                claims.append(claim)

    logger.info(f"Loaded {len(claims)} claims")

    # Process each claim
    all_results = []
    max_retries = 3
    delay_between_claims = 0.5  # seconds

    for i, claim in enumerate(claims, 1):
        logger.info(f"Processing claim {i}/{len(claims)}: {claim['claim_id']}")

        # Search for evidence with retry logic
        results = []
        for attempt in range(max_retries):
            try:
                results = search_single_claim(
                    claim=claim,
                    org_id=ORG_ID,
                    brand_id=BRAND_ID,
                    es_client=es_client,
                    config=config,
                    index_name=INDEX_NAME,
                    embedder=embedder,
                    ner_extractor=ner_extractor,
                    score_calculator=score_calculator
                )
                break  # Success, exit retry loop
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, 6s
                    logger.warning(f"Attempt {attempt + 1} failed for {claim['claim_id']}: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All retries failed for {claim['claim_id']}: {e}")
                    results = []

        # Get page offsets for paragraph numbering
        page_offsets = {}
        if results:
            page_offsets = get_page_paragraph_offsets(es_client, INDEX_NAME, results)

        all_results.append({
            "claim": claim,
            "evidence": results,
            "page_offsets": page_offsets
        })

        # Small delay between claims to avoid overwhelming ES
        if i < len(claims):
            time.sleep(delay_between_claims)

    # Apply LLM filtering if enabled
    if use_llm_filter:
        logger.info("Filtering evidence with Claude...")
        evidence_filter = ClaudeEvidenceFilter()

        # Convert to format expected by filter
        filter_input = [
            (r["claim"]["claim_id"], r["claim"]["text"], r["evidence"], r["page_offsets"])
            for r in all_results
        ]

        filtered = asyncio.run(evidence_filter.filter_all_claims(filter_input))

        # Update results with filtered evidence
        for i, (claim_id, claim_text, evidence, page_offsets) in enumerate(filtered):
            all_results[i]["evidence"] = evidence

    # Write visual_claims CSV
    logger.info(f"Writing results to {output_file}...")

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['identifier', 'claim_text', 'category', 'substantiation', 'substantiation_files'])

        for result in all_results:
            claim = result["claim"]
            evidence = result["evidence"]
            page_offsets = result["page_offsets"]

            # Format substantiation
            substantiation = format_all_evidence(evidence, page_offsets)

            # Extract unique file names
            substantiation_files = extract_unique_files(evidence)

            writer.writerow([
                claim["claim_id"],
                claim["text"],
                claim["category"],
                substantiation,
                substantiation_files
            ])

    # Summary
    total_claims = len(all_results)
    claims_with_evidence = sum(1 for r in all_results if r["evidence"])
    total_evidence = sum(len(r["evidence"]) for r in all_results)

    logger.info(f"\nSummary:")
    logger.info(f"  Total claims: {total_claims}")
    logger.info(f"  Claims with evidence: {claims_with_evidence} ({100*claims_with_evidence/total_claims:.1f}%)")
    logger.info(f"  Total evidence items: {total_evidence}")
    logger.info(f"  Avg evidence per claim: {total_evidence/total_claims:.1f}")
    logger.info(f"\nResults saved to {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Substantiate claims from CSV")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("/Users/wafaa/PycharmProjects/claim_extractor/nexobrid_prod.csv"),
        help="Input CSV file with claims"
    )
    parser.add_argument(
        "--visual_claims", "-o",
        type=Path,
        default=Path("/Users/wafaa/PycharmProjects/claim_extractor/nexobrid_substantiated.csv"),
        help="Output CSV file"
    )
    parser.add_argument(
        "--llm-filter",
        action="store_true",
        help="Use Claude to filter and rank evidence"
    )

    args = parser.parse_args()

    substantiate_claims(
        input_file=args.input,
        output_file=args.output,
        use_llm_filter=args.llm_filter
    )


if __name__ == "__main__":
    main()
