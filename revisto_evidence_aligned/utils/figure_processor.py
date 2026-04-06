"""
Figure description processing using Claude.
Cleans figure descriptions and transforms them into declarative claims for embedding.
"""

import anthropic
import json
import re
from typing import List, Optional

from .secrets import get_anthropic_api_key
from .logging import get_logger

logger = get_logger(__name__)

# Maximum claims to process per batch
MAX_BATCH_SIZE = 10


class FigureProcessor:
    """Process figure descriptions into clean declarative claims using Claude."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-opus-4-5-20251101"
    ):
        """
        Initialize the figure processor.

        Args:
            api_key: Anthropic API key (defaults to secrets manager or env var)
            model: Claude model to use (Sonnet for speed/cost balance)
        """
        self.api_key = api_key or get_anthropic_api_key()
        if not self.api_key:
            raise ValueError("Anthropic API key not configured")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model
        logger.info(f"Initialized FigureProcessor with model: {self.model}")

    def process(self, description: str) -> List[str]:
        """
        Process a figure description into clean declarative claims.

        Removes figure references (e.g., "Figure 3 shows...") and transforms
        the description into factual declarative statements.

        Args:
            description: Raw figure description from LandingAI

        Returns:
            List of clean declarative claim strings
        """
        if not description or not description.strip():
            logger.info("Empty figure description — skipping")
            return []

        logger.info(f"Processing figure description via Claude API ({len(description)} chars)")

        prompt = f"""Transform this figure description into clean declarative factual claims.

Rules:
1. Remove ALL figure references like "Figure 1 shows", "The chart depicts", "This graph illustrates", etc.
2. Convert each fact into a standalone declarative statement
3. Preserve all numerical values, percentages, p-values, and sample sizes exactly
4. Each claim should be a complete sentence that stands alone
5. Do not add interpretations or comparisons not present in the original
6. If the description contains multiple facts, output multiple claims

Output ONLY a JSON array of strings, one claim per string.

Figure description:
{description}

JSON output:"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )

            if not response.content:
                logger.warning(f"Empty response for figure description: {description[:100]}...")
                return [self._fallback_clean(description)]

            response_text = response.content[0].text.strip()

            # Handle potential markdown code blocks
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            claims = json.loads(response_text)

            if not isinstance(claims, list):
                logger.warning(f"Response is not a list: {response_text[:100]}")
                return [self._fallback_clean(description)]

            # Filter empty claims
            claims = [c.strip() for c in claims if c and c.strip()]

            if not claims:
                logger.info("Claude returned no claims — using fallback")
                return [self._fallback_clean(description)]

            logger.info(f"Figure processing done: {len(claims)} claims extracted")
            return claims

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse response as JSON: {e}")
            return [self._fallback_clean(description)]
        except Exception as e:
            logger.error(f"Error processing figure description: {e}")
            return [self._fallback_clean(description)]

    def process_batch(self, descriptions: List[dict]) -> List[List[str]]:
        """
        Process multiple figure descriptions in a single API call.

        Args:
            descriptions: List of dicts with 'id' and 'description' keys

        Returns:
            List of claim lists, one per input description
        """
        if not descriptions:
            return []

        logger.info(f"Batch processing {len(descriptions)} figure descriptions via Claude API")

        # Build batch prompt
        items_text = "\n\n".join([
            f"[{item['id']}]\n{item['description']}"
            for item in descriptions
        ])

        prompt = f"""Transform each figure description into clean declarative factual claims.

Rules:
1. Remove ALL figure references like "Figure 1 shows", "The chart depicts", "This graph illustrates", etc.
2. Convert each fact into a standalone declarative statement
3. Preserve all numerical values, percentages, p-values, and sample sizes exactly
4. Each claim should be a complete sentence that stands alone
5. Do not add interpretations or comparisons not present in the original

Output ONLY a JSON object where keys are the IDs and values are arrays of claim strings.

Figure descriptions:
{items_text}

JSON output:"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )

            if not response.content:
                logger.warning("Empty response from batch processing")
                return [[self._fallback_clean(d['description'])] for d in descriptions]

            response_text = response.content[0].text.strip()

            # Handle potential markdown code blocks
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            results = json.loads(response_text)

            # Map results back by ID
            output = []
            for item in descriptions:
                item_claims = results.get(item['id'], [])
                if not item_claims:
                    item_claims = [self._fallback_clean(item['description'])]
                output.append(item_claims)

            total_claims = sum(len(c) for c in output)
            logger.info(f"Figure batch done: {len(descriptions)} descriptions → {total_claims} claims")
            return output

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse batch response as JSON: {e}")
            # Fall back to individual processing
            return [self.process(d['description']) for d in descriptions]
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return [[self._fallback_clean(d['description'])] for d in descriptions]

    def _fallback_clean(self, description: str) -> str:
        """
        Simple regex-based fallback for cleaning figure descriptions.
        Used when LLM processing fails.

        Args:
            description: Raw figure description

        Returns:
            Cleaned description string
        """
        if not description:
            return ""

        text = description.strip()

        # Remove common figure reference patterns
        patterns = [
            r'^Figure\s+\d+[a-zA-Z]?\s*[:.]?\s*',
            r'^Fig\.\s*\d+[a-zA-Z]?\s*[:.]?\s*',
            r'^The\s+figure\s+(shows?|depicts?|illustrates?|displays?|presents?)\s+',
            r'^This\s+(chart|graph|figure|plot|diagram)\s+(shows?|depicts?|illustrates?|displays?|presents?)\s+',
            r'^(Chart|Graph|Plot|Diagram)\s+\d+\s*[:.]?\s*',
        ]

        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Capitalize first letter if needed
        if text and text[0].islower():
            text = text[0].upper() + text[1:]

        return text.strip()
