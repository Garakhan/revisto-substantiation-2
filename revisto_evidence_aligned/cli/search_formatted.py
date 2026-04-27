#!/usr/bin/env python3
"""CLI for searching claims and exporting results to CSV/XLSX"""

import argparse
import asyncio
import csv
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import anthropic
from openpyxl import Workbook, load_workbook
from openpyxl.cell.rich_text import TextBlock, CellRichText
from openpyxl.cell.text import InlineFont
from openpyxl.styles import Alignment, Font

from ..config import get_config
from ..core.searching import search_claim_by_sentences, search_claim_combined_by_sentences
from ..nlp import load_embedder, NERExtractor
from ..nlp.tokenization import StanzaNumericExtractor
from ..core.scoring import ScoreCalculator
from ..storage import ESClient
from ..utils.logging import get_logger
from ..utils.secrets import get_anthropic_api_key
from ..utils.text_utils import fix_text_encoding, sanitize_for_xml, INVALID_XML_CHARS
from ..utils.evidence_formatting import (
    make_evidence_key,
    extract_evidence_fields as _extract_evidence_fields,
    format_evidence as _format_evidence_base,
    format_all_evidence as _format_all_evidence_base,
    format_sentence_evidence as _format_sentence_evidence_base,
)
from ..utils.evidence_filter import get_source_priority, SOURCE_PRIORITY

logger = get_logger(__name__)


class DiagnosticsLogger:
    """Context manager for diagnostics logging with proper resource cleanup."""

    def __init__(self, log_path: Path = None):
        self.log_path = log_path or Path(__file__).parent.parent.parent / "search_filter_diagnostics.log"
        self._file = None

    def __enter__(self):
        self._file = open(self.log_path, 'w', encoding='utf-8')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file.close()
            self._file = None
        return False

    def log(self, message: str):
        """Write message to diagnostics log."""
        if self._file:
            self._file.write(message + "\n")
            self._file.flush()


# Module-level diagnostics logger (initialized when needed)
_diagnostics_logger: DiagnosticsLogger = None


def _get_diagnostics_logger() -> DiagnosticsLogger:
    """Get or create the diagnostics logger."""
    global _diagnostics_logger
    if _diagnostics_logger is None:
        _diagnostics_logger = DiagnosticsLogger()
        _diagnostics_logger._file = open(_diagnostics_logger.log_path, 'w', encoding='utf-8')
    return _diagnostics_logger


def log_diagnostics(message: str):
    """Write to diagnostics log file."""
    _get_diagnostics_logger().log(message)


def cleanup_diagnostics():
    """Clean up diagnostics logger resources."""
    global _diagnostics_logger
    if _diagnostics_logger and _diagnostics_logger._file:
        _diagnostics_logger._file.close()
        _diagnostics_logger._file = None
    _diagnostics_logger = None


class EvidenceGradesCache:
    """Lazy-loaded cache for evidence grades."""

    def __init__(self):
        self._grades = None

    def load(self) -> Dict[str, str]:
        """Load evidence grades from CSV (cached)."""
        if self._grades is not None:
            return self._grades

        self._grades = {}
        grades_file = Path(__file__).parent.parent.parent / "thermofisher_reference_grades.csv"
        if grades_file.exists():
            try:
                with open(grades_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        filename = row.get('filename', '').strip()
                        grade = row.get('evidence_grade', '').strip()
                        if filename and grade:
                            self._grades[filename] = grade
                logger.debug(f"Loaded {len(self._grades)} evidence grades")
            except Exception as e:
                logger.warning(f"Failed to load evidence grades: {e}")
        return self._grades

    def get(self, filename: str) -> str:
        """Get evidence grade for a filename."""
        grades = self.load()
        return grades.get(filename, "") or grades.get(filename + ".pdf", "")


# Singleton instance
_evidence_grades_cache = EvidenceGradesCache()


def _load_evidence_grades() -> Dict[str, str]:
    """Load evidence grades from thermofisher_reference_grades.csv"""
    return _evidence_grades_cache.load()


class ClaudeEvidenceFilter:
    """Filter evidence using Claude Opus 4 with 3-pass approach: rank, filter, deduplicate."""

    def __init__(self, model: str = None):
        self.api_key = get_anthropic_api_key()
        self.model = model or "claude-opus-4-20250514"
        self._client = None

        if not self.api_key:
            logger.warning("Anthropic API key not configured - evidence filtering will be disabled")

    @property
    def client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
        return self._client

    def _parse_json_response(self, text: str) -> dict:
        """Parse JSON from Claude's response, handling various formats."""
        cleaned = text.strip()

        # Remove markdown code blocks
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            parts = cleaned.split("```")
            if len(parts) >= 2:
                cleaned = parts[1].strip()

        # Try direct parse first
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object
        match = re.search(r'\{[^{}]*\}', cleaned)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return {}

    async def rank_evidence(self, claim: str, evidence_blocks: List[str]) -> List[int]:
        """Pass 1: Rank evidence blocks by relevance to the claim."""
        if not evidence_blocks:
            return []

        numbered_evidence = "\n\n".join(f"[{i}] {block}" for i, block in enumerate(evidence_blocks))
        n = len(evidence_blocks)

        tools = [{
            "name": "submit_ranking",
            "description": "Submit the ranked evidence indices",
            "input_schema": {
                "type": "object",
                "properties": {
                    "ranked_indices": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Evidence indices ordered from most to least relevant"
                    }
                },
                "required": ["ranked_indices"]
            }
        }]

        prompt = f"""Rank evidence items by relevance to this claim. Use the submit_ranking tool.

CLAIM: {claim}

EVIDENCE:
{numbered_evidence}

Rank all indices 0 to {n-1} from most to least relevant."""

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=300,
                tools=tools,
                tool_choice={"type": "tool", "name": "submit_ranking"},
                messages=[{"role": "user", "content": prompt}]
            )

            for block in response.content:
                if block.type == "tool_use" and block.name == "submit_ranking":
                    return block.input.get("ranked_indices", list(range(n)))

        except Exception as e:
            logger.error(f"  Ranking error: {e}")

        return list(range(len(evidence_blocks)))

    async def filter_evidence(self, claim: str, evidence_blocks: List[str], ranked_indices: List[int]) -> Tuple[str, dict]:
        """Pass 2: Filter evidence, keeping only substantiating items.

        Returns tuple of (filtered_text, categorization_dict).
        """
        if not evidence_blocks:
            return "", {}

        numbered_evidence = "\n\n".join(f"[{i}] {block}" for i, block in enumerate(evidence_blocks))

        tools = [{
            "name": "submit_categorization",
            "description": "Submit evidence categorization with reasoning",
            "input_schema": {
                "type": "object",
                "properties": {
                    "sub_claims": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The claim broken into individual sub-claims"
                    },
                    "reasoning": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "index": {"type": "integer"},
                                "category": {"type": "string", "enum": ["full", "partial", "none"]},
                                "explanation": {"type": "string"}
                            },
                            "required": ["index", "category", "explanation"]
                        },
                        "description": "Reasoning for each evidence item"
                    },
                    "full": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Indices that support ALL sub-claims"
                    },
                    "partial": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Indices that support AT LEAST ONE sub-claim"
                    },
                    "none": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Indices that do not support any sub-claim"
                    }
                },
                "required": ["sub_claims", "reasoning", "full", "partial", "none"]
            }
        }]

        prompt = f"""Categorize each evidence item based on whether it supports the claim.

STEP 1: Identify the CORE ASSERTIONS in the claim (usually 1-3). Don't over-split - combine related ideas.
Example: "Food allergies can be challenging to identify, and a structured approach helps ensure patients receive the right diagnosis and management."
Core assertions:
  - "Food allergies are challenging to identify"
  - "Structured approaches help with correct diagnosis/management"

Example: "42% of children and 51% of adults with food allergies have experienced a severe reaction"
Core assertions:
  - "42% of children with food allergies had severe reactions"
  - "51% of adults with food allergies had severe reactions"

STEP 2: For each evidence, check if it supports any core assertion.
Accept equivalent terms (e.g., "anaphylaxis" = "severe reaction", "difficult" = "challenging").

CLAIM: {claim}

EVIDENCE:
{numbered_evidence}

Categories:
- full: Supports ALL core assertions
- partial: Supports AT LEAST ONE core assertion
- none: Does not support any core assertion

Use submit_categorization with: sub_claims (the core assertions), reasoning (for each evidence), and categorization arrays.
"""

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                tools=tools,
                tool_choice={"type": "tool", "name": "submit_categorization"},
                messages=[{"role": "user", "content": prompt}]
            )

            for block in response.content:
                if block.type == "tool_use" and block.name == "submit_categorization":
                    sub_claims = block.input.get("sub_claims", [])
                    reasoning = block.input.get("reasoning", [])
                    full_indices = block.input.get("full", [])
                    partial_indices = block.input.get("partial", [])
                    none_indices = block.input.get("none", [])

                    categorization = {
                        "sub_claims": sub_claims,
                        "reasoning": reasoning,
                        "full": full_indices,
                        "partial": partial_indices,
                        "none": none_indices
                    }

                    if full_indices:
                        for idx in ranked_indices:
                            if idx in full_indices and idx < len(evidence_blocks):
                                return evidence_blocks[idx], categorization

                    if partial_indices:
                        filtered = []
                        for idx in ranked_indices:
                            if idx in partial_indices and idx < len(evidence_blocks):
                                filtered.append(evidence_blocks[idx])
                        return "\n\n".join(filtered), categorization

                    return "", categorization

        except Exception as e:
            logger.error(f"  Filter error: {e}")

        return "\n\n".join(evidence_blocks), {"error": "fallback"}

    async def categorize_with_subclaim_support(
        self,
        claim: str,
        evidence_blocks: List[str],
        prior_sub_claims: Optional[List[str]] = None,
    ) -> dict:
        """Tier-aware variant of filter_evidence.

        Returns a dict:
          {
            "sub_claims": [str, ...],
            "reasoning": [{"index": int, "category": "full|partial|none",
                            "supports_sub_claims": [int, ...], "explanation": str}, ...],
            "full":    [int, ...],
            "partial": [int, ...],
            "none":    [int, ...],
          }

        If prior_sub_claims is provided, Claude must reuse those sub-claims (so
        indices stay consistent across tier calls).
        """
        if not evidence_blocks:
            return {"sub_claims": prior_sub_claims or [], "reasoning": [],
                    "full": [], "partial": [], "none": []}

        numbered_evidence = "\n\n".join(f"[{i}] {block}" for i, block in enumerate(evidence_blocks))

        tools = [{
            "name": "submit_categorization",
            "description": "Submit evidence categorization with per-sub-claim support.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "sub_claims": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Core assertions of the claim. If prior sub-claims are given, return them unchanged in the same order."
                    },
                    "reasoning": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "index": {"type": "integer"},
                                "category": {"type": "string", "enum": ["full", "partial", "none"]},
                                "supports_sub_claims": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "description": "0-based indices of sub_claims this evidence supports. Empty for category=none. All indices for category=full."
                                },
                                "explanation": {"type": "string"}
                            },
                            "required": ["index", "category", "supports_sub_claims", "explanation"]
                        }
                    },
                    "full":    {"type": "array", "items": {"type": "integer"}},
                    "partial": {"type": "array", "items": {"type": "integer"}},
                    "none":    {"type": "array", "items": {"type": "integer"}}
                },
                "required": ["sub_claims", "reasoning", "full", "partial", "none"]
            }
        }]

        if prior_sub_claims:
            sub_claims_section = (
                "USE EXACTLY THESE SUB-CLAIMS (return them unchanged, in this order):\n"
                + "\n".join(f"  [{i}] {s}" for i, s in enumerate(prior_sub_claims))
            )
        else:
            sub_claims_section = (
                "STEP 1: Identify the CORE ASSERTIONS in the claim (usually 1-3). Don't over-split - combine related ideas.\n"
                "Example: \"Food allergies can be challenging to identify, and a structured approach helps ensure patients receive the right diagnosis and management.\"\n"
                "Core assertions:\n"
                "  - \"Food allergies are challenging to identify\"\n"
                "  - \"Structured approaches help with correct diagnosis/management\"\n\n"
                "Example: \"42% of children and 51% of adults with food allergies have experienced a severe reaction\"\n"
                "Core assertions:\n"
                "  - \"42% of children with food allergies had severe reactions\"\n"
                "  - \"51% of adults with food allergies had severe reactions\""
            )

        prompt = f"""Categorize each evidence item by which sub-claims it supports.

{sub_claims_section}

STEP 2: For each evidence item, list which sub-claim INDICES (0-based) it supports.
Accept equivalent terms (e.g., "anaphylaxis" = "severe reaction", "difficult" = "challenging").
Then assign one category:
  - full:    supports_sub_claims covers ALL sub-claims
  - partial: supports_sub_claims covers AT LEAST ONE sub-claim
  - none:    supports_sub_claims is empty

CLAIM: {claim}

EVIDENCE:
{numbered_evidence}

Use submit_categorization. The 'full', 'partial', 'none' arrays must each list the evidence
indices in their respective category, consistent with the per-evidence categories in 'reasoning'.
"""

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=2500,
                tools=tools,
                tool_choice={"type": "tool", "name": "submit_categorization"},
                messages=[{"role": "user", "content": prompt}]
            )

            for block in response.content:
                if block.type == "tool_use" and block.name == "submit_categorization":
                    return {
                        "sub_claims": block.input.get("sub_claims", prior_sub_claims or []),
                        "reasoning": block.input.get("reasoning", []),
                        "full":    block.input.get("full", []),
                        "partial": block.input.get("partial", []),
                        "none":    block.input.get("none", []),
                    }

        except Exception as e:
            logger.error(f"  categorize_with_subclaim_support error: {e}")

        return {"sub_claims": prior_sub_claims or [], "reasoning": [],
                "full": [], "partial": [], "none": []}

    async def deduplicate_blocks(self, claim: str, evidence_blocks: List[str]) -> List[str]:
        """Pass 3: Remove redundant evidence blocks."""
        if len(evidence_blocks) <= 1:
            return evidence_blocks

        numbered_texts = []
        for i, block in enumerate(evidence_blocks):
            parts = block.split("\n\n", 1)
            text = parts[1].strip() if len(parts) > 1 else block
            numbered_texts.append(f"[{i}] {text}")

        evidence_str = "\n\n".join(numbered_texts)
        n = len(evidence_blocks)

        tools = [{
            "name": "submit_deduplication",
            "description": "Submit indices to keep after removing duplicates",
            "input_schema": {
                "type": "object",
                "properties": {
                    "keep": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Indices of evidence items to keep"
                    }
                },
                "required": ["keep"]
            }
        }]

        prompt = f"""Remove redundant evidence. Use the submit_deduplication tool.

CLAIM: {claim}

EVIDENCE:
{evidence_str}

Keep items that add NEW facts. Remove duplicates or restated information."""

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=200,
                tools=tools,
                tool_choice={"type": "tool", "name": "submit_deduplication"},
                messages=[{"role": "user", "content": prompt}]
            )

            for block in response.content:
                if block.type == "tool_use" and block.name == "submit_deduplication":
                    keep_indices = block.input.get("keep", list(range(n)))
                    return [evidence_blocks[i] for i in keep_indices if i < len(evidence_blocks)]

        except Exception as e:
            logger.error(f"    Dedup error: {e}")

        return evidence_blocks

    async def filter_claim_evidence(self, claim_text: str, evidence_list: List[Dict[str, Any]], page_offsets: dict = None) -> List[Dict[str, Any]]:
        """Apply 3-pass filtering to evidence for a single claim."""
        # Log to diagnostics file
        log_diagnostics("\n" + "=" * 80)
        log_diagnostics(f"CLAIM: {claim_text}")
        log_diagnostics("=" * 80)

        if not self.api_key:
            logger.warning("Skipping evidence filtering - Anthropic API key not configured")
            return evidence_list

        if not evidence_list:
            log_diagnostics("NO EVIDENCE FOUND")
            return []

        # Format evidence as text blocks (without scores to avoid biasing Claude)
        evidence_blocks = []
        for ev in evidence_list:
            fields = extract_evidence_fields(ev, page_offsets)
            lines = [
                f"Evidence Grade: {fields['evidence_grade']}",
                f"Authors: {fields['authors']}",
                f"Title: {fields['title']}",
                f"Year: {fields['year']}",
                f"Page: {fields['page']}",
                f"Paragraph: {fields['paragraph']}",
                f"Sentence: {fields['sentence']}",
                f"File: {fields['file']}",
                "",
                fields['text']
            ]
            evidence_blocks.append("\n".join(lines))

        log_diagnostics(f"\nSEARCH RESULTS ({len(evidence_blocks)} evidence blocks):")
        for i, block in enumerate(evidence_blocks):
            log_diagnostics(f"\n[Evidence {i}]")
            log_diagnostics(block)

        logger.info(f"  Found {len(evidence_blocks)} evidence blocks")

        # Pass 1: Rank
        logger.info("  Pass 1: Ranking evidence...")
        ranked_indices = await self.rank_evidence(claim_text, evidence_blocks)
        log_diagnostics(f"\nPASS 1 - RANKING: {ranked_indices}")
        await asyncio.sleep(1)

        # Pass 2: Filter
        logger.info("  Pass 2: Filtering evidence...")
        filtered_text, categorization = await self.filter_evidence(claim_text, evidence_blocks, ranked_indices)

        # Log detailed reasoning
        log_diagnostics(f"\nPASS 2 - CATEGORIZATION:")
        log_diagnostics(f"  Sub-claims identified: {categorization.get('sub_claims', [])}")
        log_diagnostics(f"\n  Reasoning for each evidence:")
        for r in categorization.get('reasoning', []):
            if isinstance(r, dict):
                log_diagnostics(f"    [{r.get('index')}] {r.get('category', 'unknown').upper()}: {r.get('explanation', '')}")
            else:
                log_diagnostics(f"    {r}")
        log_diagnostics(f"\n  Summary: full={categorization.get('full', [])}, partial={categorization.get('partial', [])}, none={categorization.get('none', [])}")
        await asyncio.sleep(1)

        if not filtered_text:
            log_diagnostics("RESULT: No evidence kept (all categorized as 'none')")
            return []

        # Parse filtered text back to blocks
        filtered_blocks = []
        parts = filtered_text.split("Evidence Grade:")
        for part in parts[1:]:
            block = "Evidence Grade:" + part.strip()
            if block.strip():
                filtered_blocks.append(block)

        log_diagnostics(f"\nPASS 2 - KEPT {len(filtered_blocks)} BLOCKS")

        # Pass 3: Deduplicate
        if len(filtered_blocks) > 1:
            logger.info("  Pass 3: Deduplicating evidence...")
            unique_blocks = await self.deduplicate_blocks(claim_text, filtered_blocks)
            log_diagnostics(f"\nPASS 3 - DEDUPLICATED: {len(filtered_blocks)} -> {len(unique_blocks)}")
            filtered_blocks = unique_blocks
            await asyncio.sleep(1)

        # Map back to original evidence dicts
        filtered_evidence = []
        for block in filtered_blocks:
            # Find matching evidence by text content
            block_text = block.split("\n\n", 1)[1].strip() if "\n\n" in block else block
            for ev in evidence_list:
                ev_text = ev.get("text", "").strip()
                if ev_text in block_text or block_text in ev_text:
                    if ev not in filtered_evidence:
                        filtered_evidence.append(ev)
                        break

        log_diagnostics(f"\nFINAL RESULT: {len(filtered_evidence)} evidence items kept")

        return filtered_evidence

    async def filter_all_claims(
        self,
        claims_with_evidence: List[Tuple[str, str, List[Dict[str, Any]], dict]]
    ) -> List[Tuple[str, str, List[Dict[str, Any]], dict]]:
        """Filter evidence for multiple claims sequentially."""
        if not self.api_key:
            logger.warning("Skipping evidence filtering - Anthropic API key not configured")
            return claims_with_evidence

        results = []
        total = len(claims_with_evidence)

        for i, (claim_id, claim_text, evidence, page_offsets) in enumerate(claims_with_evidence, 1):
            logger.info(f"\nFiltering claim {i}/{total}: {claim_text[:50]}...")
            filtered = await self.filter_claim_evidence(claim_text, evidence, page_offsets)
            results.append((claim_id, claim_text, filtered, page_offsets))

        total_before = sum(len(ev) for _, _, ev, _ in claims_with_evidence)
        total_after = sum(len(ev) for _, _, ev, _ in results)
        logger.info(f"\nFiltering complete: {total_before} -> {total_after} total evidence items")

        return results

    async def filter_sentence_results(
        self,
        sentence_results: List[Dict[str, Any]],
        page_offsets: dict = None
    ) -> List[Dict[str, Any]]:
        """Stage 1: Filter evidence for each sentence independently."""
        if not self.api_key:
            logger.warning("Skipping evidence filtering - Anthropic API key not configured")
            return sentence_results

        filtered_sentences = []
        for sent_result in sentence_results:
            sentence_text = sent_result.get("sentence_text", "")
            evidence_list = sent_result.get("evidence", [])

            log_diagnostics(f"\n\n{'#' * 80}")
            log_diagnostics(f"SENTENCE: {sentence_text}")
            log_diagnostics(f"EVIDENCE RECEIVED: {len(evidence_list)} items")
            for i, ev in enumerate(evidence_list):
                log_diagnostics(f"  [{i}] {ev.get('ref_id', 'unknown')}: {ev.get('text', '')[:80]}...")

            if evidence_list:
                logger.info(f"  Filtering sentence: {sentence_text[:50]}...")
                filtered_evidence = await self.filter_claim_evidence(
                    sentence_text, evidence_list, page_offsets
                )
            else:
                filtered_evidence = []

            filtered_sentences.append({
                "sentence_index": sent_result.get("sentence_index", 0),
                "sentence_text": sentence_text,
                "evidence": filtered_evidence,
                "evidence_count": len(filtered_evidence)
            })

        return filtered_sentences

    def deduplicate_across_sentences(
        self,
        sentence_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Stage 2: Remove duplicate evidence across all sentences.

        Keeps evidence in the first sentence where it appears, removes from later sentences.
        """
        seen_refs = set()
        deduplicated_sentences = []

        for sent_result in sentence_results:
            unique_evidence = []
            for ev in sent_result.get("evidence", []):
                # Create a unique key for this evidence
                ref_key = make_evidence_key(ev)
                if ref_key not in seen_refs:
                    seen_refs.add(ref_key)
                    unique_evidence.append(ev)

            deduplicated_sentences.append({
                "sentence_index": sent_result.get("sentence_index", 0),
                "sentence_text": sent_result.get("sentence_text", ""),
                "evidence": unique_evidence,
                "evidence_count": len(unique_evidence)
            })

        return deduplicated_sentences

    async def filter_all_sentence_claims(
        self,
        claims_with_sentences: List[Tuple[str, str, List[Dict[str, Any]], dict]]
    ) -> List[Tuple[str, str, List[Dict[str, Any]], dict]]:
        """Filter evidence for sentence-level search results.

        Each sentence's evidence is filtered independently using the 3-pass approach
        (rank, categorize, deduplicate). No cross-sentence deduplication is performed
        since each sentence needs its own complete substantiation.
        """
        if not self.api_key:
            logger.warning("Skipping evidence filtering - Anthropic API key not configured")
            return claims_with_sentences

        results = []
        total = len(claims_with_sentences)

        for i, (claim_id, claim_text, sentence_results, page_offsets) in enumerate(claims_with_sentences, 1):
            logger.info(f"\nFiltering claim {i}/{total}: {claim_text[:50]}...")

            # Count evidence before filtering
            before_count = sum(len(s.get("evidence", [])) for s in sentence_results)

            # Filter each sentence's evidence independently
            logger.info("  Filtering per sentence...")
            filtered_sentences = await self.filter_sentence_results(sentence_results, page_offsets)

            after_filter = sum(len(s.get("evidence", [])) for s in filtered_sentences)
            logger.info(f"  Filtering complete: {before_count} -> {after_filter}")

            # No cross-sentence deduplication - each sentence needs its own substantiation
            results.append((claim_id, claim_text, filtered_sentences, page_offsets))

        total_before = sum(
            sum(len(s.get("evidence", [])) for s in sents)
            for _, _, sents, _ in claims_with_sentences
        )
        total_after = sum(
            sum(len(s.get("evidence", [])) for s in sents)
            for _, _, sents, _ in results
        )
        logger.info(f"\nFiltering complete: {total_before} -> {total_after} total evidence items")

        return results


def get_page_paragraph_offsets(es_client, index_name: str, evidence_list: list) -> dict:
    """Get the first paragraph number on each page for each ref_id.

    Returns dict mapping (ref_id, page) -> min_paragraph_number
    """
    # Collect unique (ref_id, page) pairs
    page_keys = set()
    for ev in evidence_list:
        ref_id = ev.get("ref_id")
        page = ev.get("page")
        if ref_id is not None and page is not None:
            page_keys.add((ref_id, page))

    if not page_keys:
        return {}

    offsets = {}
    for ref_id, page in page_keys:
        # Query ES for minimum paragraph_number on this page
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"ref_id": ref_id}},
                        {"term": {"page": page}}
                    ]
                }
            },
            "aggs": {
                "min_para": {"min": {"field": "paragraph_number"}}
            },
            "size": 0
        }
        try:
            result = es_client.client.search(index=index_name, body=query)
            min_para = result.get("aggregations", {}).get("min_para", {}).get("value")
            if min_para is not None:
                offsets[(ref_id, page)] = int(min_para)
        except Exception:
            pass

    return offsets


def extract_evidence_fields(evidence: dict, page_offsets: dict = None) -> dict:
    """Extract and process fields from evidence dict.

    Wrapper around shared utility that adds evidence grades lookup.
    """
    return _extract_evidence_fields(
        evidence,
        page_offsets,
        grades_lookup=_evidence_grades_cache.get
    )


def format_evidence(evidence: dict, page_offsets: dict = None) -> str:
    """Format a single evidence item as plain text (for CSV).

    Uses shared formatting utility with evidence grades.
    """
    fields = extract_evidence_fields(evidence, page_offsets)

    # Score breakdown
    score_breakdown = evidence.get("score_breakdown", {})
    relevance_score = evidence.get("relevance_score", "")
    numeric_tokens = evidence.get("numeric_tokens", [])
    entities = evidence.get("entities", [])

    lines = [
        f"Evidence Grade: {fields['evidence_grade']}",
        f"Authors: {fields['authors']}",
        f"Title: {fields['title']}",
        f"Year: {fields['year']}",
        f"Page: {fields['page']}",
        f"Paragraph: {fields['paragraph']}",
        f"Sentence: {fields['sentence']}",
        f"File: {fields['file']}",
        f"Relevance Score: {relevance_score}",
        f"ES Score: {score_breakdown.get('es_score', '')}",
        f"Numeric Overlap: {score_breakdown.get('numeric', '')}",
        f"Entity Overlap: {score_breakdown.get('entity', '')}",
        f"Final Score: {score_breakdown.get('final_score', '')}",
        f"Numeric Tokens: {', '.join(numeric_tokens) if numeric_tokens else ''}",
        f"Entities: {', '.join(entities) if entities else ''}",
        "",  # Empty line before text
        fields['text']
    ]

    return "\n".join(lines)


def format_evidence_rich_text(evidence: dict, page_offsets: dict = None) -> CellRichText:
    """Format a single evidence item with bold labels for XLSX."""
    fields = extract_evidence_fields(evidence, page_offsets)

    score_breakdown = evidence.get("score_breakdown", {})
    relevance_score = str(evidence.get("relevance_score", ""))
    numeric_tokens = evidence.get("numeric_tokens", [])
    entities = evidence.get("entities", [])

    bold = InlineFont(b=True)

    parts = [
        TextBlock(bold, "Evidence Grade: "), fields['evidence_grade'], "\n",
        TextBlock(bold, "Authors: "), fields['authors'], "\n",
        TextBlock(bold, "Title: "), fields['title'], "\n",
        TextBlock(bold, "Year: "), fields['year'], "\n",
        TextBlock(bold, "Page: "), fields['page'], "\n",
        TextBlock(bold, "Paragraph: "), fields['paragraph'], "\n",
        TextBlock(bold, "Sentence: "), fields['sentence'], "\n",
        TextBlock(bold, "File: "), fields['file'], "\n",
        TextBlock(bold, "Relevance Score: "), relevance_score, "\n",
        TextBlock(bold, "ES Score: "), str(score_breakdown.get('es_score', '')), "\n",
        TextBlock(bold, "Numeric Overlap: "), str(score_breakdown.get('numeric', '')), "\n",
        TextBlock(bold, "Entity Overlap: "), str(score_breakdown.get('entity', '')), "\n",
        TextBlock(bold, "Final Score: "), str(score_breakdown.get('final_score', '')), "\n",
        TextBlock(bold, "Numeric Tokens: "), ', '.join(numeric_tokens) if numeric_tokens else '', "\n",
        TextBlock(bold, "Entities: "), ', '.join(entities) if entities else '', "\n",
        "\n",
        fields['text']
    ]

    return CellRichText(*parts)


def format_all_evidence(evidence_list: list, page_offsets: dict = None) -> str:
    """Format all evidence items for a claim into a single string (CSV)."""
    if not evidence_list:
        return ""

    formatted_items = []
    for evidence in evidence_list:
        formatted_items.append(format_evidence(evidence, page_offsets))

    # Use blank lines between multiple substantiations
    return "\n\n\n".join(formatted_items)


def format_all_evidence_rich_text(evidence_list: list, page_offsets: dict = None) -> CellRichText:
    """Format all evidence items with bold labels for XLSX."""
    if not evidence_list:
        return CellRichText("")

    bold = InlineFont(b=True)
    all_parts = []

    for i, evidence in enumerate(evidence_list):
        if i > 0:
            # Add blank lines between evidence items
            all_parts.append("\n\n\n")

        fields = extract_evidence_fields(evidence, page_offsets)

        # Always show all fields with newlines, even if empty
        all_parts.extend([
            TextBlock(bold, "Evidence Grade: "), fields['evidence_grade'], "\n",
            TextBlock(bold, "Authors: "), fields['authors'], "\n",
            TextBlock(bold, "Title: "), fields['title'], "\n",
            TextBlock(bold, "Year: "), fields['year'], "\n",
            TextBlock(bold, "Page: "), fields['page'], "\n",
            TextBlock(bold, "Paragraph: "), fields['paragraph'], "\n",
            TextBlock(bold, "Sentence: "), fields['sentence'], "\n",
            TextBlock(bold, "File: "), fields['file'], "\n",
            "\n",
            fields['text']
        ])

    return CellRichText(*all_parts)


def format_sentence_evidence(sentence_results: list, page_offsets: dict = None) -> str:
    """Format evidence from all sentences (CSV)."""
    if not sentence_results:
        return ""

    formatted_items = []
    for sent_result in sentence_results:
        evidence_list = sent_result.get("evidence", [])
        for evidence in evidence_list:
            formatted_items.append(format_evidence(evidence, page_offsets))

    # Use blank lines between multiple substantiations
    return "\n\n\n".join(formatted_items)


def format_sentence_evidence_rich_text(sentence_results: list, page_offsets: dict = None) -> CellRichText:
    """Format evidence from all sentences with bold labels (XLSX)."""
    if not sentence_results:
        return CellRichText("")

    bold = InlineFont(b=True)
    all_parts = []
    evidence_count = 0

    for sent_result in sentence_results:
        evidence_list = sent_result.get("evidence", [])
        for evidence in evidence_list:
            if evidence_count > 0:
                # Add blank lines between evidence items
                all_parts.append("\n\n\n")

            fields = extract_evidence_fields(evidence, page_offsets)
            score_breakdown = evidence.get("score_breakdown", {})
            relevance_score = str(evidence.get("relevance_score", ""))
            numeric_tokens = evidence.get("numeric_tokens", [])
            entities = evidence.get("entities", [])
            all_parts.extend([
                TextBlock(bold, "Evidence Grade: "), fields['evidence_grade'], "\n",
                TextBlock(bold, "Authors: "), fields['authors'], "\n",
                TextBlock(bold, "Title: "), fields['title'], "\n",
                TextBlock(bold, "Year: "), fields['year'], "\n",
                TextBlock(bold, "Page: "), fields['page'], "\n",
                TextBlock(bold, "Paragraph: "), fields['paragraph'], "\n",
                TextBlock(bold, "Sentence: "), fields['sentence'], "\n",
                TextBlock(bold, "File: "), fields['file'], "\n",
                TextBlock(bold, "Relevance Score: "), relevance_score, "\n",
                TextBlock(bold, "ES Score: "), str(score_breakdown.get('es_score', '')), "\n",
                TextBlock(bold, "Numeric Overlap: "), str(score_breakdown.get('numeric', '')), "\n",
                TextBlock(bold, "Entity Overlap: "), str(score_breakdown.get('entity', '')), "\n",
                TextBlock(bold, "Final Score: "), str(score_breakdown.get('final_score', '')), "\n",
                TextBlock(bold, "Numeric Tokens: "), ', '.join(numeric_tokens) if numeric_tokens else '', "\n",
                TextBlock(bold, "Entities: "), ', '.join(entities) if entities else '', "\n",
                "\n",
                fields['text']
            ])
            evidence_count += 1

    return CellRichText(*all_parts)


def _normalize_key(key) -> str:
    """Normalize a column header for case/separator-insensitive lookup."""
    if key is None:
        return ""
    return str(key).strip().lower().replace("_", " ")


def _pick(row: dict, *candidates: str) -> str:
    """Return the first non-empty value in row whose key matches any candidate (case-insensitive)."""
    lookup = {_normalize_key(k): v for k, v in row.items()}
    for name in candidates:
        value = lookup.get(_normalize_key(name))
        if value is not None and str(value).strip() != "":
            return str(value)
    return ""


# Column aliases (all matched case-insensitively; underscores and spaces treated the same)
_CLAIM_ID_ALIASES = ("claim identifier", "claim id", "id")
_CLAIM_TEXT_ALIASES = ("claim text", "text", "claim")
_EXPECTED_SUB_ALIASES = ("expected substantiation",)


def _row_to_claim(row: dict) -> dict:
    """Build a claim dict from a row (CSV or XLSX). Returns None if claim text is empty."""
    claim_id = fix_text_encoding(_pick(row, *_CLAIM_ID_ALIASES))
    claim_text = fix_text_encoding(_pick(row, *_CLAIM_TEXT_ALIASES))
    expected_sub = fix_text_encoding(_pick(row, *_EXPECTED_SUB_ALIASES))
    if not claim_text.strip():
        return None
    return {
        "claim_id": claim_id,
        "text": claim_text,
        "expected_substantiation": expected_sub,
    }


def _load_claims_xlsx(claims_file: Path) -> list:
    """Load claims from an XLSX file (first sheet, first row = headers)."""
    wb = load_workbook(claims_file, read_only=True, data_only=True)
    try:
        ws = wb.active
        rows = ws.iter_rows(values_only=True)
        header = next(rows, None)
        if not header:
            return []
        header = [("" if h is None else str(h)) for h in header]

        claims = []
        for row in rows:
            if row is None:
                continue
            row_dict = {}
            for i, col in enumerate(header):
                if not col:
                    continue
                value = row[i] if i < len(row) else None
                row_dict[col] = "" if value is None else str(value)
            claim = _row_to_claim(row_dict)
            if claim:
                claims.append(claim)
        return claims
    finally:
        wb.close()


def _load_claims_csv(claims_file: Path) -> list:
    """Load claims from a CSV file, trying several encodings."""
    encodings_to_try = [
        ('utf-8-sig', 'strict'),   # UTF-8 with BOM
        ('utf-8', 'strict'),       # UTF-8 without BOM
        ('cp1252', 'strict'),      # Windows-1252 (common for Excel exports)
        ('iso-8859-1', 'strict'),  # Latin-1
        ('mac_roman', 'strict'),   # macOS legacy encoding
        ('utf-8', 'replace'),      # Last resort: UTF-8 with replacement
    ]

    claims = []
    for encoding, errors in encodings_to_try:
        try:
            claims = []
            with open(claims_file, 'r', encoding=encoding, errors=errors) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    claim = _row_to_claim(row)
                    if claim:
                        claims.append(claim)
            # Check if we got replacement characters (indicates wrong encoding)
            sample = ' '.join(c.get('text', '')[:100] for c in claims[:3])
            if '�' in sample:
                logger.debug(f"Encoding {encoding} produced replacement characters, trying next...")
                claims = []
                continue
            logger.info(f"Successfully read CSV with {encoding} encoding")
            return claims
        except UnicodeDecodeError:
            claims = []
            continue

    return claims


def _load_claims(claims_file: Path) -> list:
    """Load claims from CSV or XLSX. Raises ValueError if nothing could be read."""
    ext = claims_file.suffix.lower()
    if ext in ('.xlsx', '.xlsm'):
        claims = _load_claims_xlsx(claims_file)
        if not claims:
            raise ValueError(
                f"No claims loaded from {claims_file}. "
                f"Expected a header row with one of: {_CLAIM_TEXT_ALIASES}"
            )
        logger.info(f"Loaded {len(claims)} claims from XLSX {claims_file}")
        return claims

    claims = _load_claims_csv(claims_file)
    if not claims:
        raise ValueError(f"Could not decode {claims_file} with any supported encoding")
    return claims


def search_and_export_csv(
    claims_file: Path,
    org_id: int,
    brand_id: int,
    output_file: Path,
    es_client: ESClient = None,
    config=None,
    index_name: str = None,
    use_llm_filter: bool = False,
    use_nlp_grpc: bool = False
):
    """Search claims and export results to CSV."""

    if config is None:
        config = get_config()

    if es_client is None:
        es_client = ESClient(config.es)

    claims = _load_claims(claims_file)
    logger.info(f"Loaded {len(claims)} claims from {claims_file}")

    # Load models (gRPC or local)
    sentenciser = None
    if use_nlp_grpc:
        from ..grpc_client.client import GrpcEmbeddingModel, GrpcNERExtractor, GrpcNumericExtractor, GrpcSentenciser
        logger.info("Using NLP gRPC server for embedding/NER/numeric extraction")
        embedder = GrpcEmbeddingModel(config=config.embed)
        ner_extractor = GrpcNERExtractor(config=config.ner) if config.ner.enabled else None
        numeric_extractor = GrpcNumericExtractor()
        sentenciser = GrpcSentenciser()
    else:
        embedder = load_embedder(config.embed)
        ner_extractor = NERExtractor(config.ner) if config.ner.enabled else None
        numeric_extractor = StanzaNumericExtractor()
    score_calculator = ScoreCalculator(config.score)
    target_index = index_name or config.es.index_name

    # Build lookup for expected substantiation
    expected_substantiation_map = {c["claim_id"]: c["expected_substantiation"] for c in claims}

    # Step 1: Search all claims using combined block+sentence search
    logger.info("Step 1: Searching all claims (combined block + sentence)...")
    all_results = []  # List of (claim_id, claim_text, sentence_results, page_offsets)

    for i, claim in enumerate(claims, 1):
        logger.info(f"Searching claim {i}/{len(claims)}")

        claim_id = claim.get("claim_id", "")
        claim_text = claim.get("text", claim.get("claim_text", ""))

        # Search using combined approach (block + sentence, deduplicated)
        sentence_results = search_claim_combined_by_sentences(
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

        # Collect all evidence from all sentences for page offsets
        all_evidence = []
        for sent_result in sentence_results:
            all_evidence.extend(sent_result.get("evidence", []))

        # Limit to top 20 UNIQUE results (combined blocks + sentences) before Claude filtering
        TOP_K = 20
        # First deduplicate - same evidence may appear in multiple sentences
        seen_keys = set()
        unique_evidence = []
        for e in all_evidence:
            key = make_evidence_key(e)
            if key not in seen_keys:
                seen_keys.add(key)
                unique_evidence.append(e)

        original_unique_count = len(unique_evidence)
        if len(unique_evidence) > TOP_K:
            # Sort by relevance_score descending and take top 20 unique
            unique_evidence_sorted = sorted(unique_evidence, key=lambda x: x.get("relevance_score", 0), reverse=True)
            top_evidence = unique_evidence_sorted[:TOP_K]
            top_evidence_keys = {
                make_evidence_key(e)
                for e in top_evidence
            }
        else:
            top_evidence = unique_evidence
            top_evidence_keys = {
                make_evidence_key(e)
                for e in top_evidence
            }

        # Rebuild sentence_results - assign each evidence to only ONE sentence (first occurrence)
        used_keys = set()
        filtered_sentence_results = []
        for sent_result in sentence_results:
            filtered_evidence = []
            for e in sent_result.get("evidence", []):
                key = make_evidence_key(e)
                if key in top_evidence_keys and key not in used_keys:
                    filtered_evidence.append(e)
                    used_keys.add(key)
            if filtered_evidence:
                filtered_sent = sent_result.copy()
                filtered_sent["evidence"] = filtered_evidence
                filtered_sentence_results.append(filtered_sent)

        sentence_results = filtered_sentence_results
        all_evidence = top_evidence
        if original_unique_count > TOP_K:
            log_diagnostics(f"Limited to top {TOP_K} unique results (from {original_unique_count} unique)")

        # Get page offsets now, during search phase
        page_offsets = get_page_paragraph_offsets(es_client, target_index, all_evidence)

        all_results.append((claim_id, claim_text, sentence_results, page_offsets))

    # Step 2: Filter with Claude Opus 4 if enabled (two-stage: per-sentence filter + global dedup)
    if use_llm_filter:
        logger.info("Step 2: Filtering evidence with Claude Opus 4 (two-stage)...")
        evidence_filter = ClaudeEvidenceFilter()
        filtered_results = asyncio.run(evidence_filter.filter_all_sentence_claims(all_results))
    else:
        # No filtering, keep as-is
        filtered_results = all_results

    # Step 3: Write results to CSV
    logger.info("Step 3: Writing results to CSV...")
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['claim_identifier', 'claim_text', 'expected_substantiation', 'substantiation'])

        for claim_id, claim_text, sentence_results, page_offsets in filtered_results:
            # Format evidence grouped by sentence
            substantiation = format_sentence_evidence(sentence_results, page_offsets)
            expected_sub = expected_substantiation_map.get(claim_id, '')

            writer.writerow([claim_id, claim_text, expected_sub, substantiation])

    logger.info(f"Results saved to {output_file}")


def search_and_export_xlsx(
    claims_file: Path,
    org_id: int,
    brand_id: int,
    output_file: Path,
    es_client: ESClient = None,
    config=None,
    index_name: str = None,
    use_llm_filter: bool = False,
    use_nlp_grpc: bool = False
):
    """Search claims and export results to XLSX with bold formatting."""

    if config is None:
        config = get_config()

    if es_client is None:
        es_client = ESClient(config.es)

    claims = _load_claims(claims_file)
    logger.info(f"Loaded {len(claims)} claims from {claims_file}")

    # Load models (gRPC or local)
    sentenciser = None
    if use_nlp_grpc:
        from ..grpc_client.client import GrpcEmbeddingModel, GrpcNERExtractor, GrpcNumericExtractor, GrpcSentenciser
        logger.info("Using NLP gRPC server for embedding/NER/numeric extraction")
        embedder = GrpcEmbeddingModel(config=config.embed)
        ner_extractor = GrpcNERExtractor(config=config.ner) if config.ner.enabled else None
        numeric_extractor = GrpcNumericExtractor()
        sentenciser = GrpcSentenciser()
    else:
        embedder = load_embedder(config.embed)
        ner_extractor = NERExtractor(config.ner) if config.ner.enabled else None
        numeric_extractor = StanzaNumericExtractor()
    score_calculator = ScoreCalculator(config.score)
    target_index = index_name or config.es.index_name

    # Build lookup for expected substantiation
    expected_substantiation_map = {c["claim_id"]: c["expected_substantiation"] for c in claims}

    # Step 1: Search all claims using combined block+sentence search
    logger.info("Step 1: Searching all claims (combined block + sentence)...")
    all_results = []  # List of (claim_id, claim_text, sentence_results, page_offsets)

    for i, claim in enumerate(claims, 1):
        logger.info(f"Searching claim {i}/{len(claims)}")

        claim_id = claim.get("claim_id", "")
        claim_text = claim.get("text", claim.get("claim_text", ""))

        # Search using combined approach (block + sentence, deduplicated)
        sentence_results = search_claim_combined_by_sentences(
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

        # Collect all evidence from all sentences for page offsets
        all_evidence = []
        for sent_result in sentence_results:
            all_evidence.extend(sent_result.get("evidence", []))

        # Limit to top 20 UNIQUE results (combined blocks + sentences) before Claude filtering
        TOP_K = 20
        # First deduplicate - same evidence may appear in multiple sentences
        seen_keys = set()
        unique_evidence = []
        for e in all_evidence:
            key = make_evidence_key(e)
            if key not in seen_keys:
                seen_keys.add(key)
                unique_evidence.append(e)

        original_unique_count = len(unique_evidence)
        if len(unique_evidence) > TOP_K:
            # Sort by relevance_score descending and take top 20 unique
            unique_evidence_sorted = sorted(unique_evidence, key=lambda x: x.get("relevance_score", 0), reverse=True)
            top_evidence = unique_evidence_sorted[:TOP_K]
            top_evidence_keys = {
                make_evidence_key(e)
                for e in top_evidence
            }
        else:
            top_evidence = unique_evidence
            top_evidence_keys = {
                make_evidence_key(e)
                for e in top_evidence
            }

        # Rebuild sentence_results - assign each evidence to only ONE sentence (first occurrence)
        used_keys = set()
        filtered_sentence_results = []
        for sent_result in sentence_results:
            filtered_evidence = []
            for e in sent_result.get("evidence", []):
                key = make_evidence_key(e)
                if key in top_evidence_keys and key not in used_keys:
                    filtered_evidence.append(e)
                    used_keys.add(key)
            if filtered_evidence:
                filtered_sent = sent_result.copy()
                filtered_sent["evidence"] = filtered_evidence
                filtered_sentence_results.append(filtered_sent)

        sentence_results = filtered_sentence_results
        all_evidence = top_evidence
        if original_unique_count > TOP_K:
            log_diagnostics(f"Limited to top {TOP_K} unique results (from {original_unique_count} unique)")

        # Get page offsets now, during search phase
        page_offsets = get_page_paragraph_offsets(es_client, target_index, all_evidence)

        all_results.append((claim_id, claim_text, sentence_results, page_offsets))

    # Step 2: Filter with Claude Opus 4 if enabled (two-stage: per-sentence filter + global dedup)
    if use_llm_filter:
        logger.info("Step 2: Filtering evidence with Claude Opus 4 (two-stage)...")
        evidence_filter = ClaudeEvidenceFilter()
        filtered_results = asyncio.run(evidence_filter.filter_all_sentence_claims(all_results))
    else:
        # No filtering, keep as-is
        filtered_results = all_results

    # Step 3: Write results to XLSX
    logger.info("Step 3: Writing results to XLSX...")
    wb = Workbook()
    ws = wb.active
    ws.title = "Claims"

    # Write header row
    ws.append(['claim_identifier', 'claim_text', 'expected_substantiation', 'substantiation'])

    # Make header bold
    for cell in ws[1]:
        cell.font = Font(bold=True)

    for claim_id, claim_text, sentence_results, page_offsets in filtered_results:
        # Format evidence grouped by sentence with rich text (bold labels)
        substantiation = format_sentence_evidence_rich_text(sentence_results, page_offsets)
        expected_sub = expected_substantiation_map.get(claim_id, '')

        # Add row (sanitize text for XML compatibility)
        row_num = ws.max_row + 1
        ws.cell(row=row_num, column=1, value=sanitize_for_xml(claim_id))
        ws.cell(row=row_num, column=2, value=sanitize_for_xml(claim_text))
        ws.cell(row=row_num, column=3, value=sanitize_for_xml(expected_sub))
        ws.cell(row=row_num, column=4, value=substantiation)

    # Adjust column widths
    ws.column_dimensions['A'].width = 20
    ws.column_dimensions['B'].width = 50
    ws.column_dimensions['C'].width = 80

    # Enable text wrap for all cells
    for row in ws.iter_rows(min_row=2, max_col=3, max_row=ws.max_row):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical='top')

    wb.save(output_file)
    logger.info(f"Results saved to {output_file}")


def main():
    """Main entry point for CSV search CLI"""
    parser = argparse.ArgumentParser(
        description="Search for evidence matching claims and export to CSV or XLSX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output formats:
  .csv  - Plain text CSV
  .xlsx - Excel with bold labels (Authors:, Title:, etc.)

Examples:
  # Export to CSV
  python -m revisto_evidence_aligned.cli.search_formatted \\
    --claims-file claims.csv --org-id 1 --brand-id 1 -o results.csv

  # Export to XLSX (with bold formatting)
  python -m revisto_evidence_aligned.cli.search_formatted \\
    --claims-file claims.csv --org-id 1 --brand-id 1 -o results.xlsx
        """
    )

    # Required arguments
    parser.add_argument(
        "--claims-file",
        type=Path,
        required=True,
        help=(
            "Claims file (.csv or .xlsx). "
            "Column matching is case-insensitive; accepted headers: "
            "'claim identifier' / 'claim id' / 'id'; "
            "'claim text' / 'text' / 'claim'; "
            "'expected substantiation' (optional)."
        )
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
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output file for results (.csv or .xlsx)"
    )

    # Optional arguments
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
        "--llm-filter",
        action="store_true",
        help="Use Claude Opus 4 to filter evidence (3-pass: rank, filter, deduplicate)"
    )
    parser.add_argument(
        "--nlp-grpc",
        action="store_true",
        help="Use NLP gRPC server for embedding/NER/numeric extraction instead of loading models locally"
    )
    parser.add_argument(
        "--index",
        type=str,
        help="Elasticsearch index name (overrides config)"
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

    # Create ES client
    try:
        es_client = ESClient(config.es)
        _ = es_client.client
    except Exception as e:
        logger.error(f"Failed to connect to Elasticsearch: {e}")
        return 1

    # Run search - detect format from file extension
    try:
        output_ext = args.output.suffix.lower()

        if output_ext == '.xlsx':
            search_and_export_xlsx(
                claims_file=args.claims_file,
                org_id=args.org_id,
                brand_id=args.brand_id,
                output_file=args.output,
                es_client=es_client,
                config=config,
                index_name=args.index,
                use_llm_filter=args.llm_filter,
                use_nlp_grpc=args.nlp_grpc
            )
        else:
            # Default to CSV
            search_and_export_csv(
                claims_file=args.claims_file,
                org_id=args.org_id,
                brand_id=args.brand_id,
                output_file=args.output,
                es_client=es_client,
                config=config,
                index_name=args.index,
                use_llm_filter=args.llm_filter,
                use_nlp_grpc=args.nlp_grpc
            )
        return 0

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
