#!/usr/bin/env python3
"""CLI for indexing reference documents"""

import asyncio
import argparse
import sys
from pathlib import Path

from ..config import get_config
from ..core import index_reference_directory
from ..storage import ESClient
from ..utils.logging import get_logger
from ..utils.table_linearizer import ClaudeLinearizer
from ..utils.figure_processor import FigureProcessor

logger = get_logger(__name__)


async def main():
    """Main entry point for indexing CLI"""
    parser = argparse.ArgumentParser(
        description="Index reference PDFs for evidence alignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index all PDFs in a directory (sentence-level segmentation)
  revisto-index --refs-dir ./references --org-id 1 --brand-id 1

  # Index with block-level segmentation (larger chunks)
  revisto-index --refs-dir ./papers --org-id 1 --brand-id 1 --segmentation-level block

  # Index with custom batch size
  revisto-index --refs-dir ./references --org-id 1 --brand-id 1 --batch-size 1000

  # Disable NER extraction
  revisto-index --refs-dir ./references --org-id 1 --brand-id 1 --no-ner
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--refs-dir",
        type=Path,
        required=True,
        help="Directory containing reference PDFs"
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
        "--segmentation-level",
        type=str,
        choices=["sentence", "block"],
        default="sentence",
        help="Segmentation level: 'sentence' for fine-grained, 'block' for paragraph-level (default: sentence)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for bulk indexing (default: 500)"
    )
    parser.add_argument(
        "--no-ner",
        action="store_true",
        help="Disable NER extraction"
    )

    # Content type indexing flags (controls both processing and indexing)
    parser.add_argument(
        "--index-text",
        action="store_true",
        default=None,
        help="Index text content (default: enabled)"
    )
    parser.add_argument(
        "--no-index-text",
        action="store_true",
        help="Skip indexing text content"
    )
    parser.add_argument(
        "--index-tables",
        action="store_true",
        default=None,
        help="Linearize and index table content (default: enabled)"
    )
    parser.add_argument(
        "--no-index-tables",
        action="store_true",
        help="Skip table linearization and indexing"
    )
    parser.add_argument(
        "--index-figures",
        action="store_true",
        default=None,
        help="Process and index figure content (default: enabled)"
    )
    parser.add_argument(
        "--no-index-figures",
        action="store_true",
        help="Skip figure processing and indexing"
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
    if not args.refs_dir.exists():
        logger.error(f"Directory not found: {args.refs_dir}")
        return 1
    
    if not args.refs_dir.is_dir():
        logger.error(f"Not a directory: {args.refs_dir}")
        return 1
    
    # Load configuration
    config = get_config()
    
    # Override NER if requested
    if args.no_ner:
        config.ner.enabled = False

    # Determine content type indexing flags
    # --no-index-* takes precedence over --index-*
    index_text = not args.no_index_text if args.no_index_text else (args.index_text if args.index_text else config.indexing.index_text)
    index_tables = not args.no_index_tables if args.no_index_tables else (args.index_tables if args.index_tables else config.indexing.index_tables)
    index_figures = not args.no_index_figures if args.no_index_figures else (args.index_figures if args.index_figures else config.indexing.index_figures)

    # Create ES client
    try:
        es_client = ESClient(config.es)
        # Test connection
        _ = es_client.client
    except Exception as e:
        logger.error(f"Failed to connect to Elasticsearch: {e}")
        return 1

    # Initialize table linearizer only if indexing tables (linearization + indexing are coupled)
    table_linearizer = None
    if index_tables:
        try:
            table_linearizer = ClaudeLinearizer()
            logger.info("Table linearization enabled")
        except Exception as e:
            logger.warning(f"Could not initialize table linearizer: {e}")
            logger.warning("Continuing without table support")
            index_tables = False  # Can't index tables without linearizer

    # Initialize figure processor only if indexing figures (processing + indexing are coupled)
    figure_processor = None
    if index_figures:
        try:
            figure_processor = FigureProcessor(model=config.figure.model)
            logger.info("Figure processing enabled")
        except Exception as e:
            logger.warning(f"Could not initialize figure processor: {e}")
            logger.warning("Continuing without figure support")
            index_figures = False  # Can't index figures without processor

    # Show configuration
    logger.info("=== Configuration ===")
    logger.info(f"Elasticsearch URL: {config.es.url}")
    logger.info(f"Index name: {config.es.index_name}")
    logger.info(f"Alias: {config.es.alias}")
    logger.info(f"Embedding model: {config.embed.model_name}")
    logger.info(f"NER enabled: {config.ner.enabled}")
    if config.ner.enabled:
        logger.info(f"NER model: {config.ner.model_name}")
    logger.info(f"Index text: {index_text}")
    logger.info(f"Index tables: {index_tables}")
    logger.info(f"Index figures: {index_figures}")
    logger.info(f"References directory: {args.refs_dir}")
    logger.info(f"Organization ID: {args.org_id}")
    logger.info(f"Brand ID: {args.brand_id}")
    logger.info(f"Segmentation level: {args.segmentation_level}")
    logger.info("===================")

    # Run indexing
    try:
        results = await index_reference_directory(
            refs_dir=args.refs_dir,
            org_id=args.org_id,
            brand_id=args.brand_id,
            es_client=es_client,
            config=config,
            batch_size=args.batch_size,
            segmentation_level=args.segmentation_level,
            table_linearizer=table_linearizer,
            figure_processor=figure_processor,
            index_text=index_text,
            index_tables=index_tables,
            index_figures=index_figures
        )
        
        # Show results
        logger.info("\n=== Indexing Results ===")
        logger.info(f"PDFs processed: {results['pdf_count']}")
        logger.info(f"Documents indexed: {results['document_count']}")
        logger.info(f"Successful: {results['index_stats']['success']}")
        logger.info(f"Errors: {results['index_stats']['errors']}")
        
        # Show index statistics
        doc_count = es_client.count(config.es.index_name)
        logger.info(f"\nTotal documents in index: {doc_count}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Indexing failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    asyncio.run(main())