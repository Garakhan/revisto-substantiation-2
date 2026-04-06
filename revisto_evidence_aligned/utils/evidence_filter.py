"""Shared evidence filtering utilities using Claude Opus 4.

This module provides the core evidence filtering logic used by both
the search CLI and standalone filtering scripts.
"""

import json
import re
from typing import List, Dict, Any, Tuple

import anthropic

from .logging import get_logger
from .secrets import get_anthropic_api_key
from .evidence_formatting import (
    extract_file_from_block,
    parse_evidence_blocks,
    extract_text_content,
    make_evidence_key,
)

logger = get_logger(__name__)

# Default model for evidence filtering
DEFAULT_CLAUDE_MODEL = "claude-opus-4-20250514"

# Default max tokens for API calls
DEFAULT_MAX_TOKENS_RANK = 500
DEFAULT_MAX_TOKENS_FILTER = 500
DEFAULT_MAX_TOKENS_DEDUP = 400

# Default text truncation length for deduplication
DEFAULT_TEXT_TRUNCATION = 500

# Source priority for evidence selection (lower number = higher priority)
# Based on official reference hierarchy for Propeller/Abrizult claims
SOURCE_PRIORITY = {
    # Priority 1: PI (Must use if available)
    "DRAFT PI Cytisinicline": 1,

    # Priority 2: Peer-reviewed landmark trials for the drug
    "Rigotti JAMA 2023_ORCA-2": 2,
    "Rigotti JAMA Intern Med 2025_ORCA-3": 2,
    "Rigotti JAMA 2023_ORCA-2 (suppl 3)": 2,
    "ORCA 2 Supplemental Online data": 2,
    "Rigotti SRNT Abstract Final": 2,

    # Priority 3: CSRs of the peer-reviewed landmark trials
    "Cytisinicline CSR_ACH-CYT-03_ORCA 2-1-124": 3,
    "Cytisinicline CSR_ACH-CYT-03_ORCA 2-2698-2919": 3,
    "Cytisinicline CSR_ACH-CYT-04_ORCA 3-1-125": 3,
    "Cytisinicline CSR_ACH-CYT-04_ORCA 3-2221-2487": 3,

    # Priority 4: Other published studies
    "Benowitz Am J Med 2008": 4,
    "Benowitz N Engl J Med 2010": 4,
    "Chaiton BMJ Open 2016": 4,
    "Coe J Med Chem 2005": 4,
    "Da Ré Int Arch Otorhinolaryngol 2017": 4,
    "DRAFT MANUSCRIPT Comparative effectiveness of Cytisinicline and varenicline for smoking cessation": 4,
    "Hughes Nicotine Tob Res 2013": 4,
    "Mendez Nicotine Tob Res 2022": 4,
    "Prochaska Thorax 2025": 4,
    "Rafful Addict Behav 2013": 4,
    "Rollema Psychopharmacology 2018": 4,
    "Tutka Addiction 2019": 4,
    "VanFrank MMWR 2024": 4,
    "Rigotti JAMA 2022": 4,

    # Priority 5: Pharma congress presentation/posters
    "Lummis SRNT E 2020 5HT3 presentation": 5,
    "Cytisinicline and Its Minimal Binding to 5HT3 1a": 5,

    # Priority 6: Health guidance reports
    "US Surgeon General Report 2014": 6,
    "US Surgeon General Report 2014 - Chapter 5": 6,
    "US Surgeon General Report 2014-107-138": 6,
    "US Surgeon General Report 2020": 6,
    "US Surgeon General Report 2020 - Chapter 1": 6,
    "US Surgeon General Report 2020 - Chapter 2": 6,
    "US Surgeon General Report 2020 - Chapter 3": 6,
    "US Surgeon General Report 2020 - Chapter 4": 6,
    "US Surgeon General Report 2020 - Chapter 5": 6,
    "US Surgeon General Report 2020 - Chapter 6": 6,
    "US Surgeon General Report 2020 - Chapter 7": 6,
    "US Surgeon General Report 2020 - Chapter 8": 6,

    # Priority 7: Health guidance websites
    "ACS. Health benefits of quitting smoking over time": 7,
    "AHA The benefits of quitting smoking now": 7,
    "CDC Benefits of quitting smoking": 7,
    "FDA CDER Drug Approval Package_ Chantix (Varenicline) NDA #021928": 7,
    "Mayo Clinic_Nicotine dependence": 7,
    "Smoke Free_Benefits of Quitting": 7,
    "Smokefree.gov-Managing Nicotine Withdrawal": 7,

    # Priority 8: Internal/other (lowest priority)
    "DOF weight gain": 8,
}

# Default priority for sources not in the list
DEFAULT_SOURCE_PRIORITY = 99

# Pre-computed normalized priority lookup for faster matching
# Built at module load time to avoid repeated string normalization
_SOURCE_PRIORITY_NORMALIZED = {}
for _key, _priority in SOURCE_PRIORITY.items():
    _normalized = _key.lower().replace('_', ' ').replace('-', ' ')
    _SOURCE_PRIORITY_NORMALIZED[_normalized] = (_priority, _key)


def get_source_priority(file_name: str) -> int:
    """Get the priority level for a source file.

    Uses pre-computed normalized lookup for O(1) matching instead of
    iterating through all entries.

    Args:
        file_name: Name of the source file

    Returns:
        Priority level (1-8, lower is higher priority; 99 for unknown)
    """
    if not file_name:
        return DEFAULT_SOURCE_PRIORITY

    # Try exact match first (fastest)
    if file_name in SOURCE_PRIORITY:
        return SOURCE_PRIORITY[file_name]

    # Try normalized match using pre-computed lookup
    file_normalized = file_name.lower().replace('_', ' ').replace('-', ' ')
    if file_normalized in _SOURCE_PRIORITY_NORMALIZED:
        return _SOURCE_PRIORITY_NORMALIZED[file_normalized][0]

    # Try substring match as fallback (for partial matches)
    for normalized_key, (priority, _) in _SOURCE_PRIORITY_NORMALIZED.items():
        if normalized_key in file_normalized or file_normalized in normalized_key:
            return priority

    return DEFAULT_SOURCE_PRIORITY


# Note: parse_evidence_blocks, extract_text_content, extract_file_from_block
# are now imported from evidence_formatting module


class ClaudeEvidenceFilter:
    """Filter evidence using source priority ranking and Claude Opus 4 for filtering.

    Pipeline:
    1. Sort by source priority (deterministic, based on official reference hierarchy)
    2. Filter evidence (Claude API - keep substantiating items)
    3. Deduplicate (Claude API - remove redundant blocks)

    This is the canonical implementation of evidence filtering. Other modules
    should import and use this class rather than duplicating the logic.
    """

    def __init__(
        self,
        model: str = None,
        max_tokens_rank: int = DEFAULT_MAX_TOKENS_RANK,
        max_tokens_filter: int = DEFAULT_MAX_TOKENS_FILTER,
        max_tokens_dedup: int = DEFAULT_MAX_TOKENS_DEDUP,
        text_truncation: int = DEFAULT_TEXT_TRUNCATION
    ):
        """Initialize the evidence filter.

        Args:
            model: Claude model to use (defaults to claude-opus-4-20250514)
            max_tokens_rank: Max tokens for ranking API call
            max_tokens_filter: Max tokens for filtering API call
            max_tokens_dedup: Max tokens for deduplication API call
            text_truncation: Max characters to include per evidence in dedup
        """
        self.api_key = get_anthropic_api_key()
        self.model = model or DEFAULT_CLAUDE_MODEL
        self.max_tokens_rank = max_tokens_rank
        self.max_tokens_filter = max_tokens_filter
        self.max_tokens_dedup = max_tokens_dedup
        self.text_truncation = text_truncation
        self._client = None

        if not self.api_key:
            logger.warning("Anthropic API key not configured - evidence filtering will be disabled")

    @property
    def client(self) -> anthropic.AsyncAnthropic:
        """Lazy-load the async Anthropic client."""
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
        return self._client

    async def rank_evidence(self, claim: str, evidence_blocks: List[str]) -> List[int]:
        """Pass 1: Rank evidence blocks by relevance to the claim."""
        if not evidence_blocks:
            return []

        numbered_evidence = "\n\n".join(f"[{i}] {block}" for i, block in enumerate(evidence_blocks))

        prompt = f"""You are ranking evidence items by how relevant they are to substantiating a claim.

CLAIM:
{claim}

EVIDENCE ITEMS (each contains metadata like Authors/Title/Year followed by the actual text content):
{numbered_evidence}

Rank ALL evidence items from most relevant to least relevant based on how well they support the claim.
Consider the TEXT CONTENT (after the metadata) when evaluating relevance.

Return a JSON object with the indices in order from most to least relevant:
{{"ranked_indices": [2, 0, 5, 1, 3, 4]}}

Include ALL indices in your ranking, even if some are not relevant at all."""

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens_rank,
                messages=[{"role": "user", "content": prompt}]
            )

            result_text = response.content[0].text.strip()
            logger.debug(f"Ranking response: {result_text}")

            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result.get("ranked_indices", list(range(len(evidence_blocks))))

        except Exception as e:
            logger.error(f"Ranking error: {e}")

        return list(range(len(evidence_blocks)))

    def sort_by_source_priority(self, evidence_blocks: List[str]) -> List[int]:
        """Sort evidence blocks by source priority (no API call).

        Uses the SOURCE_PRIORITY dictionary to rank evidence by the official
        reference hierarchy. Higher priority sources (lower numbers) come first.

        Args:
            evidence_blocks: List of evidence block strings

        Returns:
            List of indices sorted by source priority (highest priority first)
        """
        if not evidence_blocks:
            return []

        # Extract file name and priority for each block
        priorities = []
        for i, block in enumerate(evidence_blocks):
            file_name = extract_file_from_block(block)
            priority = get_source_priority(file_name)
            priorities.append((i, priority, file_name))
            logger.debug(f"Block [{i}] file='{file_name}' priority={priority}")

        # Sort by priority (lower number = higher priority)
        priorities.sort(key=lambda x: x[1])

        # Log the sorted order
        sorted_indices = [p[0] for p in priorities]
        logger.info(f"Source priority order: {[(p[2], p[1]) for p in priorities[:5]]}...")

        return sorted_indices

    async def filter_evidence(
        self,
        claim: str,
        evidence_blocks: List[str],
        ranked_indices: List[int]
    ) -> str:
        """Pass 2: Filter evidence, keeping only substantiating items."""
        if not evidence_blocks:
            return ""

        numbered_evidence = "\n\n".join(f"[{i}] {block}" for i, block in enumerate(evidence_blocks))

        prompt = f"""You are evaluating whether evidence items substantiate a claim.

CLAIM:
{claim}

EVIDENCE ITEMS (each contains metadata like Authors/Title/Year followed by the actual text content):
{numbered_evidence}

For each evidence item, evaluate the TEXT CONTENT (after the metadata) and categorize it as:
- "full": Substantiates the claim with clear, specific evidence
- "partial": Partially supports the claim but not completely
- "none": Does not substantiate the claim

Evidence should be considered "full" if it:
- Supports the SAME CONCLUSION as the claim (e.g., treatment is effective vs placebo)
- Provides data from the same or related studies, even if exact numbers differ slightly
- Shows the same direction and similar magnitude of effect (e.g., 4x vs 8x odds ratio both show strong effect)
- Contains specific facts, data, or statements that validate the claim's core message
- NOTE: Pooled data vs individual trial data may have different numbers but support the same claim

Evidence should be "partial" if it:
- Supports some aspect of the claim but not all
- Provides related but indirect evidence
- Contains general information that aligns with the claim
- Shows similar trends but from different populations or timepoints

Evidence should be "none" if it:
- Is unrelated to the claim
- Contradicts the claim (shows opposite effect)
- Is too vague or generic to be useful
- Only tangentially mentions related topics without supporting the specific claim

Return a JSON object categorizing each evidence index:
{{"full": [0, 2], "partial": [5], "none": [1, 3, 4]}}

Focus on whether the evidence supports the SAME SCIENTIFIC CONCLUSION, not whether exact numbers match perfectly."""

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens_filter,
                messages=[{"role": "user", "content": prompt}]
            )

            result_text = response.content[0].text.strip()
            logger.debug(f"Filter response: {result_text}")

            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                full_indices = result.get("full", [])
                partial_indices = result.get("partial", [])

                # If any evidence fully substantiates, keep only the highest-ranked one
                if full_indices:
                    for idx in ranked_indices:
                        if idx in full_indices and idx < len(evidence_blocks):
                            logger.info(f"Full substantiation found, keeping best: index {idx}")
                            return evidence_blocks[idx]

                # Otherwise, keep all partial substantiations (in ranked order)
                if partial_indices:
                    filtered = []
                    for idx in ranked_indices:
                        if idx in partial_indices and idx < len(evidence_blocks):
                            filtered.append(evidence_blocks[idx])
                    logger.info(f"Partial substantiations: {len(partial_indices)} blocks")
                    return "\n\n".join(filtered)

                logger.info("No substantiating evidence found")
                return ""

            logger.warning("Could not parse filter response, keeping all")
            return "\n\n".join(evidence_blocks)

        except Exception as e:
            logger.error(f"Filter error: {e}")
            return "\n\n".join(evidence_blocks)

    async def deduplicate_blocks(self, claim: str, evidence_blocks: List[str]) -> List[str]:
        """Pass 3: Remove redundant evidence blocks."""
        if len(evidence_blocks) <= 1:
            return evidence_blocks

        # Build numbered evidence with truncated text content for comparison
        numbered_texts = []
        for i, block in enumerate(evidence_blocks):
            text = extract_text_content(block)
            numbered_texts.append(f"[{i}] {text[:self.text_truncation]}")

        evidence_str = "\n\n".join(numbered_texts)

        prompt = f"""You must remove redundant evidence. Be STRICT.

CLAIM: {claim}

EVIDENCE:
{evidence_str}

RULES:
1. Evidence [0] is the baseline - it stays
2. For each subsequent evidence, ask: "Does this add NEW FACTS not already covered?"
3. REMOVE if it: restates the same information, provides a different perspective of the same fact, or uses different words to say the same thing
4. KEEP only if it adds genuinely NEW information (new data points, new studies, new mechanisms)

For each evidence after [0], decide: KEEP (adds new facts) or REMOVE (redundant)

Return ONLY this JSON structure:
{{"keep": [0, 2], "remove": [1, 3, 4], "reasoning": {{"1": "same mechanism as 0", "3": "restates clinical data from 2", "4": "no new information"}}}}"""

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens_dedup,
                messages=[{"role": "user", "content": prompt}]
            )

            result_text = response.content[0].text.strip()
            logger.debug(f"Dedup response: {result_text}")

            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                keep_indices = result.get("keep", list(range(len(evidence_blocks))))
                reasoning = result.get("reasoning", {})

                # Log what's being removed and why
                for idx, reason in reasoning.items():
                    logger.debug(f"Removing [{idx}]: {reason}")

                return [evidence_blocks[i] for i in keep_indices if i < len(evidence_blocks)]

        except Exception as e:
            logger.error(f"Dedup error: {e}")

        return evidence_blocks

    async def filter_substantiation(self, claim: str, substantiation: str) -> str:
        """Apply filtering pipeline to a substantiation string.

        Pipeline:
        1. Sort by source priority (deterministic, no API call)
        2. Filter evidence (Claude API)
        3. Deduplicate if needed (Claude API)
        """
        if not substantiation or not str(substantiation).strip():
            return ""

        evidence_blocks = parse_evidence_blocks(str(substantiation))
        if not evidence_blocks:
            return ""

        logger.info(f"Found {len(evidence_blocks)} evidence blocks")

        # Step 1: Sort by source priority (no API call)
        logger.info("Step 1: Sorting by source priority...")
        priority_indices = self.sort_by_source_priority(evidence_blocks)

        # Step 2: Filter evidence (Claude API)
        logger.info("Step 2: Filtering evidence...")
        filtered = await self.filter_evidence(claim, evidence_blocks, priority_indices)

        # Step 3: Deduplicate if multiple blocks remain
        filtered_blocks = parse_evidence_blocks(filtered)

        if len(filtered_blocks) > 1:
            logger.info("Step 3: Deduplicating evidence...")
            unique_blocks = await self.deduplicate_blocks(claim, filtered_blocks)
            logger.info(f"Deduplicated: {len(filtered_blocks)} -> {len(unique_blocks)} blocks")
            return "\n\n".join(unique_blocks)

        return filtered


def count_evidence_blocks(text: str) -> int:
    """Count the number of evidence blocks in a substantiation string."""
    if not text:
        return 0
    return len([p for p in str(text).split("Authors:")[1:] if p.strip()])


def create_evidence_filter_from_config(config=None) -> ClaudeEvidenceFilter:
    """Create a ClaudeEvidenceFilter instance from config.

    Args:
        config: Config object with claude settings, or None to use defaults

    Returns:
        Configured ClaudeEvidenceFilter instance
    """
    if config is None:
        return ClaudeEvidenceFilter()

    claude_config = getattr(config, 'claude', None)
    if claude_config is None:
        return ClaudeEvidenceFilter()

    return ClaudeEvidenceFilter(
        model=claude_config.model,
        max_tokens_rank=claude_config.max_tokens_rank,
        max_tokens_filter=claude_config.max_tokens_filter,
        max_tokens_dedup=claude_config.max_tokens_dedup,
        text_truncation=claude_config.text_truncation
    )
