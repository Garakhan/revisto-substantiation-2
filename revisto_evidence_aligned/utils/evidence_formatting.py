"""Evidence formatting utilities.

This module provides centralized functions for formatting evidence blocks
for display in CSV, XLSX, and other outputs.
"""

import re
from typing import Dict, Any, List, Tuple, Optional, Callable

from .text_utils import sanitize_for_xml


def make_evidence_key(evidence: Dict[str, Any]) -> Tuple:
    """Create a unique key for an evidence item.

    Used for deduplication across sentences and claims.

    Args:
        evidence: Evidence dictionary with ref_id, page, paragraph_number, sentence_number

    Returns:
        Tuple of (ref_id, page, paragraph_number, sentence_number)
    """
    return (
        evidence.get("ref_id"),
        evidence.get("page"),
        evidence.get("paragraph_number"),
        evidence.get("sentence_number")
    )


def extract_evidence_fields(
    evidence: Dict[str, Any],
    page_offsets: Dict = None,
    grades_lookup: Callable[[str], str] = None
) -> Dict[str, str]:
    """Extract and process fields from evidence dict.

    Args:
        evidence: Evidence dictionary from search results
        page_offsets: Optional dict of (ref_id, page) -> min_paragraph for relative numbering
        grades_lookup: Optional function to look up evidence grade by filename

    Returns:
        Dictionary with sanitized string fields for formatting
    """
    # Get metadata (may be nested in doc_metadata)
    doc_metadata = evidence.get("doc_metadata", {}) or {}

    # File name (ref_id)
    file_name = evidence.get("ref_id", "")

    # Evidence grade from lookup - try with and without .pdf extension
    evidence_grade = ""
    if grades_lookup:
        evidence_grade = grades_lookup(file_name) or grades_lookup(file_name + ".pdf") or ""

    # Authors - can be a list or string
    authors = doc_metadata.get("authors") or evidence.get("authors", "")
    if isinstance(authors, list):
        authors = ", ".join(str(a) for a in authors)

    # Title - from doc_metadata or ref_title
    title = doc_metadata.get("title") or evidence.get("ref_title", "")

    # Year
    year = doc_metadata.get("year") or evidence.get("year", "")

    # Location info - page is already 1-indexed from indexing
    page = evidence.get("page", 1)

    abs_paragraph = evidence.get("paragraph_number", "")
    sentence = evidence.get("sentence_number", "")

    # Check if evidence comes from a table
    table_source = evidence.get("table_source", "")

    if table_source:
        # Use table source info instead of paragraph number
        # e.g., "Table 1; row 12; column 7"
        paragraph = table_source
    else:
        # Calculate page-relative paragraph number
        paragraph = abs_paragraph
        if page_offsets and abs_paragraph:
            ref_id = evidence.get("ref_id")
            key = (ref_id, page)  # Use 1-indexed page for lookup (matches indexing)
            if key in page_offsets:
                paragraph = abs_paragraph - page_offsets[key] + 1

    # The actual text
    text = evidence.get("text", "")

    return {
        "evidence_grade": sanitize_for_xml(str(evidence_grade)) if evidence_grade else "",
        "authors": sanitize_for_xml(str(authors)) if authors else "",
        "title": sanitize_for_xml(str(title)) if title else "",
        "year": sanitize_for_xml(str(year)) if year else "",
        "page": sanitize_for_xml(str(page)) if page else "",
        "paragraph": sanitize_for_xml(str(paragraph)) if paragraph else "",
        "sentence": sanitize_for_xml(str(sentence)) if sentence else "",
        "file": sanitize_for_xml(str(file_name)) if file_name else "",
        "text": sanitize_for_xml(str(text)) if text else ""
    }


def format_evidence(
    evidence: Dict[str, Any],
    page_offsets: Dict = None,
    grades_lookup: Callable[[str], str] = None,
    include_grade: bool = True
) -> str:
    """Format a single evidence item as plain text (for CSV).

    Args:
        evidence: Evidence dictionary
        page_offsets: Optional dict for relative paragraph numbering
        grades_lookup: Optional function to look up evidence grade
        include_grade: Whether to include Evidence Grade line

    Returns:
        Formatted string with metadata and text
    """
    fields = extract_evidence_fields(evidence, page_offsets, grades_lookup)

    lines = []
    if include_grade:
        lines.append(f"Evidence Grade: {fields['evidence_grade']}")
    lines.extend([
        f"Authors: {fields['authors']}",
        f"Title: {fields['title']}",
        f"Year: {fields['year']}",
        f"Page: {fields['page']}",
        f"Paragraph: {fields['paragraph']}",
        f"Sentence: {fields['sentence']}",
        f"File: {fields['file']}",
        "",  # Empty line before text
        fields['text']
    ])

    return "\n".join(lines)


def format_all_evidence(
    evidence_list: List[Dict[str, Any]],
    page_offsets: Dict = None,
    grades_lookup: Callable[[str], str] = None,
    include_grade: bool = True
) -> str:
    """Format all evidence items for a claim into a single string (CSV).

    Args:
        evidence_list: List of evidence dictionaries
        page_offsets: Optional dict for relative paragraph numbering
        grades_lookup: Optional function to look up evidence grade
        include_grade: Whether to include Evidence Grade line

    Returns:
        Formatted string with all evidence items separated by blank lines
    """
    if not evidence_list:
        return ""

    formatted_items = []
    for evidence in evidence_list:
        formatted_items.append(format_evidence(evidence, page_offsets, grades_lookup, include_grade))

    # Use blank lines between multiple substantiations
    return "\n\n\n".join(formatted_items)


def format_sentence_evidence(
    sentence_results: List[Dict[str, Any]],
    page_offsets: Dict = None,
    grades_lookup: Callable[[str], str] = None,
    include_grade: bool = True
) -> str:
    """Format evidence grouped by sentence.

    Args:
        sentence_results: List of sentence result dicts with 'sentence_text' and 'evidence'
        page_offsets: Optional dict for relative paragraph numbering
        grades_lookup: Optional function to look up evidence grade
        include_grade: Whether to include Evidence Grade line

    Returns:
        Formatted string with sentences and their evidence
    """
    if not sentence_results:
        return ""

    formatted_items = []
    for sent_result in sentence_results:
        sentence_text = sent_result.get("sentence_text", "")
        evidence_list = sent_result.get("evidence", [])

        if evidence_list:
            for evidence in evidence_list:
                formatted_items.append(format_evidence(evidence, page_offsets, grades_lookup, include_grade))

    return "\n\n\n".join(formatted_items)


def parse_evidence_blocks(substantiation: str) -> List[str]:
    """Parse substantiation text into individual evidence blocks.

    Args:
        substantiation: Raw substantiation text with multiple evidence blocks

    Returns:
        List of evidence block strings, each starting with "Authors:"
    """
    if not substantiation:
        return []

    parts = str(substantiation).split("Authors:")
    blocks = []
    for part in parts[1:]:  # Skip first empty part
        block = "Authors: " + part.strip()
        if block.strip():
            blocks.append(block)
    return blocks


def extract_file_from_block(block: str) -> str:
    """Extract the File field from an evidence block.

    Args:
        block: Evidence block string with metadata header

    Returns:
        File name or empty string if not found
    """
    match = re.search(r'File:\s*(.+?)(?:\n|$)', block)
    if match:
        return match.group(1).strip()
    return ""


def extract_text_content(block: str) -> str:
    """Extract just the text content from an evidence block (after metadata).

    Args:
        block: Evidence block string with metadata header

    Returns:
        Text content portion after the metadata
    """
    parts = block.split("\n\n", 1)
    if len(parts) > 1:
        return parts[1].strip()
    return block
