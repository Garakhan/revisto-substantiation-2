"""Document indexing pipeline"""

from typing import List, Dict, Any, Optional, Literal, Union
from pathlib import Path
import datetime
import re
import numpy as np
from ..utils.landingai_parser import process_pdf_landingai, process_landingai_chunks_async
from ..config import get_config, Config
from ..nlp import (
    load_embedder,
    extract_numbers,
    NERExtractor
)
from ..nlp.tokenization import StanzaNumericExtractor
from ..storage import ESClient, create_index, bulk_index
from ..utils.logging import get_logger


logger = get_logger(__name__)


def extract_year_from_filename(filename: str) -> Optional[str]:
    """Extract a 4-digit year from filename.

    Matches years from 1900-2099 in various filename patterns:
    - "Rigotti JAMA 2023_ORCA-2.pdf" -> "2023"
    - "2019_Yawn_The-Allergy-and-Asthma.pdf" -> "2019"
    - "Rosenberg et al. 2014.pdf" -> "2014"
    - "Singer & Boyce, 2017.pdf" -> "2017"

    Returns the most likely publication year (prefers years at boundaries).
    """
    # Find all 4-digit years (1900-2099)
    years = re.findall(r'\b(19\d{2}|20\d{2})\b', filename)

    if not years:
        return None

    if len(years) == 1:
        return years[0]

    # Multiple years found - prefer the one that looks like a publication year
    # (typically at word boundaries, after author names, or at the end)
    # Return the last one as it's often the publication year
    return years[-1]

SegmentationLevel = Literal["sentence", "block"]

async def process_reference_document(
    pdf_path: Path,
    config: Config,
    embedder,
    ner_extractor: Optional[NERExtractor] = None,
    sentenciser=None,
    table_linearizer=None,
    figure_processor=None,
    segmentation_level: SegmentationLevel = "sentence",
    cache_dir: Optional[str] = None,
    numeric_extractor: Optional[Union[StanzaNumericExtractor, Any]] = None,
    index_text: bool = True,
    index_tables: bool = True,
    index_figures: bool = True
) -> List[Dict[str, Any]]:
    """Process a single reference PDF into indexable documents using publication strategy."""
    logger.info(f"Processing {pdf_path.name} with publication strategy, level: {segmentation_level}")
    logger.info(f"  Content types: text={index_text}, tables={index_tables}, figures={index_figures}")

    # Parse PDF with LandingAI
    chunks, metadata = await process_pdf_landingai(str(pdf_path), extract_metadata=True, cache_dir=cache_dir)

    # Process chunks in parallel (text, tables, figures) based on flags
    segments = await process_landingai_chunks_async(
        str(pdf_path),
        chunks,
        table_linearizer=table_linearizer,
        figure_processor=figure_processor,
        segmentation_level=segmentation_level,
        index_text=index_text,
        index_tables=index_tables,
        index_figures=index_figures
    )

    documents = []
    ref_id = pdf_path.stem

    # Use extracted metadata for title if available, otherwise fall back to filename
    ref_title = pdf_path.stem.replace("_", " ").strip()
    if metadata and metadata.get("metadata"):
        extracted_meta = metadata["metadata"]
        if extracted_meta.get("title"):
            ref_title = extracted_meta["title"]
        elif extracted_meta.get("chapter_title"):
            ref_title = extracted_meta["chapter_title"]

    # --- Phase 1: Collect all text units with their metadata ---
    text_unit_records = []  # list of (text, segment, segment_page, paragraph_on_page, sent_idx)
    current_page = None
    paragraph_on_page = 0

    for seg_idx, segment in enumerate(segments):
        # Track paragraph number per page (reset when page changes)
        segment_page = segment.page + 1  # Convert from 0-indexed to 1-indexed
        if segment_page != current_page:
            current_page = segment_page
            paragraph_on_page = 1
        else:
            paragraph_on_page += 1

        # Publication strategy: use segments as citation-based chunks
        # Table segments are already linearized - don't sentencize them
        if segment.segment_type in ("table", "table_row", "table_cell") or segmentation_level == "block":
            text_units = [(segment.text, 0)]
        else:
            if sentenciser:
                sentences = sentenciser.sentencise(segment.text)
            else:
                import stanza
                nlp = stanza.Pipeline("en", processors="tokenize")
                doc = nlp(segment.text)
                sentences = [s.text for s in doc.sentences]
            text_units = [(sent, idx) for idx, sent in enumerate(sentences)]

        for unit_idx, (text_unit, sent_idx) in enumerate(text_units):
            if len(text_unit.strip()) < config.segmentation.min_len:
                continue
            text_unit_records.append((text_unit, segment, segment_page, paragraph_on_page, sent_idx))

    if not text_unit_records:
        logger.info(f"  → 0 segments after filtering")
        return documents

    all_texts = [r[0] for r in text_unit_records]
    logger.info(f"  Collected {len(all_texts)} text units, running batch NLP...")

    # --- Phase 2: Batch gRPC calls ---
    # Embedding/NER: lightweight models, can handle larger batches
    GRPC_BATCH_SIZE = 512
    # GLiNER: heavy model, needs smaller batches to avoid OOM on server
    GLINER_BATCH_SIZE = 64

    # Embeddings
    logger.info(f"  Encoding {len(all_texts)} texts (batch size {GRPC_BATCH_SIZE})...")
    all_embeddings_parts = []
    for i in range(0, len(all_texts), GRPC_BATCH_SIZE):
        chunk = all_texts[i:i + GRPC_BATCH_SIZE]
        all_embeddings_parts.append(embedder.encode(chunk))
        if len(all_texts) > GRPC_BATCH_SIZE:
            logger.info(f"    Embedding batch {i // GRPC_BATCH_SIZE + 1}/{(len(all_texts) + GRPC_BATCH_SIZE - 1) // GRPC_BATCH_SIZE} done")
    all_embeddings = np.concatenate(all_embeddings_parts, axis=0) if len(all_embeddings_parts) > 1 else all_embeddings_parts[0]
    logger.info(f"  Embeddings done")

    # Numeric extraction (GLiNER — smaller batches to avoid OOM)
    if numeric_extractor and hasattr(numeric_extractor, 'extract_batch'):
        total_gliner_batches = (len(all_texts) + GLINER_BATCH_SIZE - 1) // GLINER_BATCH_SIZE
        logger.info(f"  Extracting numeric entities ({len(all_texts)} texts, batch size {GLINER_BATCH_SIZE}, {total_gliner_batches} batches)...")
        all_numeric_tokens = []
        for i in range(0, len(all_texts), GLINER_BATCH_SIZE):
            chunk = all_texts[i:i + GLINER_BATCH_SIZE]
            all_numeric_tokens.extend(numeric_extractor.extract_batch(chunk))
            logger.info(f"    Numeric batch {i // GLINER_BATCH_SIZE + 1}/{total_gliner_batches} done")
        logger.info(f"  Numeric extraction done")
    elif numeric_extractor:
        all_numeric_tokens = [numeric_extractor.extract(t) for t in all_texts]
    else:
        all_numeric_tokens = [extract_numbers(t) for t in all_texts]

    # NER extraction
    if ner_extractor and config.ner.enabled and hasattr(ner_extractor, 'extract_batch'):
        logger.info(f"  Extracting NER entities (batch size {GRPC_BATCH_SIZE})...")
        all_ner_results = []
        for i in range(0, len(all_texts), GRPC_BATCH_SIZE):
            chunk = all_texts[i:i + GRPC_BATCH_SIZE]
            all_ner_results.extend(ner_extractor.extract_batch(chunk))
            if len(all_texts) > GRPC_BATCH_SIZE:
                logger.info(f"    NER batch {i // GRPC_BATCH_SIZE + 1}/{(len(all_texts) + GRPC_BATCH_SIZE - 1) // GRPC_BATCH_SIZE} done")
        logger.info(f"  NER extraction done")
    elif ner_extractor and config.ner.enabled:
        all_ner_results = [ner_extractor.extract(t) for t in all_texts]
    else:
        all_ner_results = [[] for _ in all_texts]

    # --- Phase 3: Build documents ---
    # Prepare metadata once
    extracted_meta = metadata.get("metadata") if metadata else None
    if extracted_meta:
        year = extracted_meta.get("year")
        if not year:
            year = extract_year_from_filename(pdf_path.name)
            if year:
                logger.debug(f"Year extracted from filename: {year}")
        doc_metadata = {
            "document_type": metadata.get("document_type_info", {}).get("document_type"),
            "title": extracted_meta.get("title") or extracted_meta.get("chapter_title"),
            "authors": extracted_meta.get("authors"),
            "year": year,
            "journal_name": extracted_meta.get("journal_name"),
            "volume": extracted_meta.get("volume"),
            "issue": extracted_meta.get("issue"),
            "doi": extracted_meta.get("doi"),
            "publisher": extracted_meta.get("publisher"),
        }
    else:
        year = extract_year_from_filename(pdf_path.name)
        if year:
            logger.debug(f"Year extracted from filename (no metadata): {year}")
            doc_metadata = {"year": year}
        else:
            doc_metadata = None

    for i, (text_unit, segment, segment_page, para_on_page, sent_idx) in enumerate(text_unit_records):
        embedding = all_embeddings[i]
        numeric_tokens = all_numeric_tokens[i]
        ner_results = all_ner_results[i]

        entities = [e["text"] for e in ner_results]
        entity_types = [e["type"] for e in ner_results]

        doc_id = f"{ref_id}::page{segment_page}::para{para_on_page}::sent{sent_idx+1}"
        source = segment.metadata.get("source") if segment.metadata else None

        doc = {
            "ref_id": ref_id,
            "ref_title": ref_title,
            "sent_id": doc_id,
            "text": text_unit,
            "section": segment.section,
            "page": segment_page,
            "paragraph_number": para_on_page,
            "sentence_number": sent_idx + 1,
            "bbox": {
                "x0": segment.bbox[0],
                "y0": segment.bbox[1],
                "x1": segment.bbox[2],
                "y1": segment.bbox[3]
            },
            "numeric_tokens": numeric_tokens,
            "entities": entities,
            "entity_types": entity_types,
            "vector": embedding.tolist(),
            "labels": ["reference", segment.segment_type],
            "segmentation_strategy": "publication",
            "segmentation_level": segmentation_level,
            "segment_type": segment.segment_type,
            "source": source,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }

        if doc_metadata:
            doc["doc_metadata"] = doc_metadata

        documents.append(doc)

    sent_counter = len(text_unit_records)
    if segmentation_level == "block":
        logger.info(f"  → {sent_counter} block segments extracted from {len(segments)} segments")
    else:
        logger.info(f"  → {sent_counter} sentence segments extracted from {len(segments)} segments")
    return documents


async def index_reference_directory(
    refs_dir: Path,
    org_id: int,
    brand_id: int,
    es_client: Optional[ESClient] = None,
    config: Optional[Config] = None,
    index_name: Optional[str] = None,
    batch_size: int = 500,
    segmentation_level: SegmentationLevel = "sentence",
    cache_dir: Optional[str] = None,
    embedder=None,
    ner_extractor=None,
    sentenciser=None,
    table_linearizer=None,
    figure_processor=None,
    numeric_extractor=None,
    index_text: bool = True,
    index_tables: bool = True,
    index_figures: bool = True
) -> Dict[str, Any]:
    """Index all PDFs in a reference directory using publication strategy.

    Args:
        refs_dir: Directory containing PDF files.
        org_id: Organization ID.
        brand_id: Brand ID.
        es_client: Elasticsearch client.
        config: Configuration object.
        index_name: Custom index name.
        batch_size: Batch size for bulk indexing.
        segmentation_level: Level of segmentation ("sentence" or "block").
        cache_dir: Directory to save parsed document cache. If None, saves next to PDFs.
        embedder: Embedding model (local or gRPC).
        ner_extractor: NER extractor (local or gRPC).
        sentenciser: Sentence splitter (gRPC only, None for local stanza).
        table_linearizer: ClaudeLinearizer for table-to-text conversion.
        figure_processor: FigureProcessor for figure description cleaning.
        numeric_extractor: Numeric entity extractor (Stanza-based, local or gRPC).
        index_text: Whether to index text chunks.
        index_tables: Whether to index table chunks.
        index_figures: Whether to index figure chunks.
    """
    if config is None:
        config = get_config()

    if es_client is None:
        es_client = ESClient(config.es)

    # Use provided index name or default from config
    target_index = index_name or config.es.index_name

    # Create index if needed
    create_index(es_client, target_index, config.embed)

    # Create alias
    if config.es.alias != target_index:
        es_client.create_alias(target_index, config.es.alias)

    # Load models only if not provided
    if embedder is None:
        embedder = load_embedder(config.embed)
    if ner_extractor is None and config.ner.enabled:
        ner_extractor = NERExtractor(config.ner)

    # Find all PDFs
    pdf_files = list(refs_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {refs_dir}")

    # Use refs_dir as cache_dir if not specified
    effective_cache_dir = cache_dir or str(refs_dir)

    total_document_count = 0
    total_stats = {"success": 0, "errors": 0}
    num_pdfs = len(pdf_files)

    for pdf_idx, pdf_path in enumerate(pdf_files, 1):
        try:
            logger.info(f"[PDF {pdf_idx}/{num_pdfs}] Starting: {pdf_path.name}")
            docs = await process_reference_document(
                pdf_path,
                config,
                embedder,
                ner_extractor,
                sentenciser,
                table_linearizer,
                figure_processor,
                segmentation_level,
                cache_dir=effective_cache_dir,
                numeric_extractor=numeric_extractor,
                index_text=index_text,
                index_tables=index_tables,
                index_figures=index_figures
            )

            # Add org/brand IDs
            for doc in docs:
                doc["org_id"] = org_id
                doc["brand_id"] = brand_id

            # Index to ES immediately after each PDF
            if docs:
                logger.info(f"[PDF {pdf_idx}/{num_pdfs}] Indexing {len(docs)} documents to {target_index}...")
                stats = bulk_index(es_client, target_index, docs, batch_size)
                total_stats["success"] += stats.get("success", 0)
                total_stats["errors"] += stats.get("errors", 0)

            total_document_count += len(docs)
            logger.info(f"[PDF {pdf_idx}/{num_pdfs}] Done: {pdf_path.name} → {len(docs)} documents indexed")

        except Exception as e:
            logger.error(f"[PDF {pdf_idx}/{num_pdfs}] Error processing {pdf_path}: {e}", exc_info=True)
            continue

    return {
        "pdf_count": len(pdf_files),
        "document_count": total_document_count,
        "index_stats": total_stats
    }


async def index_references(
    pdf_paths: List[Path],
    org_id: int,
    brand_id: int,
    es_client: Optional[ESClient] = None,
    config: Optional[Config] = None,
    index_name: Optional[str] = None,
    batch_size: int = 500,
    segmentation_level: SegmentationLevel = "sentence",
    cache_dir: Optional[str] = None,
    embedder=None,
    ner_extractor=None,
    sentenciser=None,
    table_linearizer=None,
    figure_processor=None,
    numeric_extractor=None,
    index_text: bool = True,
    index_tables: bool = True,
    index_figures: bool = True
) -> Dict[str, Any]:
    """Index specific reference PDFs using publication strategy.

    Args:
        pdf_paths: List of PDF file paths to index.
        org_id: Organization ID.
        brand_id: Brand ID.
        es_client: Elasticsearch client.
        config: Configuration object.
        index_name: Custom index name.
        batch_size: Batch size for bulk indexing.
        segmentation_level: Level of segmentation ("sentence" or "block").
        cache_dir: Directory to save parsed document cache (JSON files).
        embedder: Embedding model (local or gRPC).
        ner_extractor: NER extractor (local or gRPC).
        sentenciser: Sentence splitter (gRPC only, None for local stanza).
        table_linearizer: ClaudeLinearizer for table-to-text conversion.
        figure_processor: FigureProcessor for figure description cleaning.
        numeric_extractor: Numeric entity extractor (Stanza-based, local or gRPC).
        index_text: Whether to index text chunks.
        index_tables: Whether to index table chunks.
        index_figures: Whether to index figure chunks.
    """
    if config is None:
        config = get_config()

    if es_client is None:
        es_client = ESClient(config.es)

    # Use provided index name or default from config
    target_index = index_name or config.es.index_name

    # Create index if needed
    create_index(es_client, target_index, config.embed)

    # Load models only if not provided
    if embedder is None:
        embedder = load_embedder(config.embed)
    if ner_extractor is None and config.ner.enabled:
        ner_extractor = NERExtractor(config.ner)

    total_document_count = 0
    total_stats = {"success": 0, "errors": 0}
    num_pdfs = len(pdf_paths)

    for pdf_idx, pdf_path in enumerate(pdf_paths, 1):
        try:
            logger.info(f"[PDF {pdf_idx}/{num_pdfs}] Starting: {pdf_path.name}")
            docs = await process_reference_document(
                pdf_path,
                config,
                embedder,
                ner_extractor,
                sentenciser,
                table_linearizer,
                figure_processor,
                segmentation_level,
                cache_dir=cache_dir,
                numeric_extractor=numeric_extractor,
                index_text=index_text,
                index_tables=index_tables,
                index_figures=index_figures
            )

            # Add org/brand IDs
            for doc in docs:
                doc["org_id"] = org_id
                doc["brand_id"] = brand_id

            # Index to ES immediately after each PDF
            if docs:
                logger.info(f"[PDF {pdf_idx}/{num_pdfs}] Indexing {len(docs)} documents to {target_index}...")
                stats = bulk_index(es_client, target_index, docs, batch_size)
                total_stats["success"] += stats.get("success", 0)
                total_stats["errors"] += stats.get("errors", 0)

            total_document_count += len(docs)
            logger.info(f"[PDF {pdf_idx}/{num_pdfs}] Done: {pdf_path.name} → {len(docs)} documents indexed")

        except Exception as e:
            logger.error(f"[PDF {pdf_idx}/{num_pdfs}] Error processing {pdf_path}: {e}", exc_info=True)
            continue

    return {
        "pdf_count": len(pdf_paths),
        "document_count": total_document_count,
        "index_stats": total_stats
    }