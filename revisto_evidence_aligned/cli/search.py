#!/usr/bin/env python3
"""CLI for searching claims against indexed evidence"""

import argparse
import sys
import json
from pathlib import Path

from ..config import get_config
from ..core import search_claims
from ..storage import ESClient
from ..utils.logging import get_logger

logger = get_logger(__name__)


def main():
    """Main entry point for search CLI"""
    parser = argparse.ArgumentParser(
        description="Search for evidence matching claims",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search claims from JSON file
  revisto-search --claims-file claims.json --org-id 1 --brand-id 1 --visual_claims results.json

  # Search with custom threshold
  revisto-search --claims-file claims.json --org-id 1 --brand-id 1 --threshold 0.5
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--claims-file",
        type=Path,
        required=True,
        help="JSON file containing claims to search"
    )
    parser.add_argument(
        "--org-id",
        type=int,
        required=True,
        help="Organization ID"
    )
    parser.add_argument(
        "--brand-id",
        type=int,
        required=True,
        help="Brand ID"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output-file",
        "-o",
        type=Path,
        help="Output file for results (default: print to stdout)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Minimum relevance score threshold"
    )
    parser.add_argument(
        "--topk",
        type=int,
        help="Maximum number of evidence per claim"
    )
    parser.add_argument(
        "--no-ner",
        action="store_true",
        help="Disable NER extraction"
    )
    parser.add_argument(
        "--nlp-grpc",
        action="store_true",
        help="Use NLP gRPC server for embedding/NER instead of loading models locally"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logger.setLevel(args.log_level)
    
    # Validate inputs
    if not args.claims_file.exists():
        logger.error(f"Claims file not found: {args.claims_file}")
        return 1
    
    # Load configuration
    config = get_config()
    
    # Override settings if provided
    if args.threshold is not None:
        config.score.threshold = args.threshold
    
    if args.topk is not None:
        config.score.topk = args.topk

    if args.no_ner:
        config.ner.enabled = False
    
    # Create ES client
    try:
        es_client = ESClient(config.es)
        # Test connection
        _ = es_client.client
    except Exception as e:
        logger.error(f"Failed to connect to Elasticsearch: {e}")
        return 1
    
    # Check index exists
    if not es_client.client.indices.exists(index=config.es.index_name):
        logger.error(f"Index not found: {config.es.index_name}")
        logger.error("Please run indexing first")
        return 1
    
    # Show configuration
    logger.info("=== Configuration ===")
    logger.info(f"Elasticsearch URL: {config.es.url}")
    logger.info(f"Index name: {config.es.index_name}")
    logger.info(f"Embedding model: {config.embed.model_name}")
    logger.info(f"NER enabled: {config.ner.enabled}")
    logger.info(f"Score threshold: {config.score.threshold}")
    logger.info(f"Top K results: {config.score.topk}")
    logger.info(f"Claims file: {args.claims_file}")
    logger.info(f"Organization ID: {args.org_id}")
    logger.info(f"Brand ID: {args.brand_id}")
    logger.info("===================")
    
    # Load models (gRPC or local)
    embedder = None
    ner_extractor = None
    numeric_extractor = None

    if args.nlp_grpc:
        from ..grpc_client.client import GrpcEmbeddingModel, GrpcNERExtractor, GrpcNumericExtractor
        logger.info("Using NLP gRPC server for embedding/NER/numeric extraction")
        embedder = GrpcEmbeddingModel(config=config.embed)
        if config.ner.enabled:
            ner_extractor = GrpcNERExtractor(config=config.ner)
        numeric_extractor = GrpcNumericExtractor()

    # Run search
    try:
        results = search_claims(
            claims_file=args.claims_file,
            org_id=args.org_id,
            brand_id=args.brand_id,
            output_file=args.output_file,
            es_client=es_client,
            config=config,
            embedder=embedder,
            ner_extractor=ner_extractor,
            numeric_extractor=numeric_extractor
        )

        # Show summary
        summary = results["summary"]
        logger.info("\n=== Search Results Summary ===")
        logger.info(f"Total claims: {summary['total_claims']}")
        logger.info(f"Claims with evidence: {summary['claims_with_evidence']}")
        logger.info(f"Coverage: {summary['coverage']:.1%}")
        logger.info(f"Avg evidence per claim: {summary['avg_evidence_per_claim']:.1f}")

        # If no output file, print sample results
        if not args.output_file and results["results"]:
            logger.info("\n=== Sample Results (first 3 claims) ===")
            for i, result in enumerate(results["results"][:3]):
                claim = result["claim"]
                evidence = result["evidence"]
                logger.info(f"\nClaim {i+1}: {claim.get('text', claim.get('claim_text', ''))[:100]}...")
                logger.info(f"Evidence found: {len(evidence)}")
                
                if evidence:
                    top_evidence = evidence[0]
                    logger.info(f"  Top match (score: {top_evidence['relevance_score']:.3f}):")
                    logger.info(f"  {top_evidence['text'][:100]}...")
                    logger.info(f"  From: {top_evidence['ref_title']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())