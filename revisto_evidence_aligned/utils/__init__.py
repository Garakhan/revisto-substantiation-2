"""Utility modules"""

from .logging import get_logger
from .types import BBox, Segment, PageGeometry
from .evidence_formatting import (
    make_evidence_key,
    extract_evidence_fields,
    format_evidence,
    format_all_evidence,
    format_sentence_evidence,
    parse_evidence_blocks,
    extract_file_from_block,
    extract_text_content,
)
from .evidence_filter import (
    ClaudeEvidenceFilter,
    get_source_priority,
    SOURCE_PRIORITY,
    create_evidence_filter_from_config,
)

__all__ = [
    "get_logger",
    "BBox",
    "Segment",
    "PageGeometry",
    # Evidence formatting
    "make_evidence_key",
    "extract_evidence_fields",
    "format_evidence",
    "format_all_evidence",
    "format_sentence_evidence",
    "parse_evidence_blocks",
    "extract_file_from_block",
    "extract_text_content",
    # Evidence filtering
    "ClaudeEvidenceFilter",
    "get_source_priority",
    "SOURCE_PRIORITY",
    "create_evidence_filter_from_config",
]