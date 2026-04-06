import json
import os
import re
from typing import Dict, List, Optional, Any
import anthropic

from .logging import get_logger
from .secrets import get_anthropic_api_key

logger = get_logger(__name__)


class MetadataExtractor:
    """Extract bibliographic metadata from page 0 of LandingAI chunks using Claude Opus."""

    DOCUMENT_TYPES = {
        "journal_article": {
            "description": "Scientific/academic journal article with DOI, volume, issue",
            "required": ["authors", "title", "journal_name", "year"],
            "optional": [
                "journal_name_full", "volume", "issue", "supplement",
                "page_range", "doi", "column_number", "line_number",
                "table_number", "figure_number"
            ]
        },
        "book": {
            "description": "Complete book publication",
            "required": ["authors", "title", "publisher", "year"],
            "optional": ["edition", "publication_city", "state", "isbn"]
        },
        "book_chapter": {
            "description": "Chapter within an edited book",
            "required": ["chapter_authors", "chapter_title", "editors", "book_title", "publisher", "year"],
            "optional": ["edition", "publication_city", "state", "page_range"]
        },
        "website": {
            "description": "Website or webpage",
            "required": ["title", "website_name", "url", "accessed_date"],
            "optional": ["authors", "year"]
        },
        "online_video": {
            "description": "Online video content (YouTube, Vimeo, etc.)",
            "required": ["title", "url", "accessed_date"],
            "optional": ["author_host", "distributor", "year"]
        },
        "drug_database": {
            "description": "Drug/pharmaceutical database entry",
            "required": ["database_title", "entry_name", "url", "accessed_date"],
            "optional": ["publisher"]
        }
    }

    FIELD_DESCRIPTIONS = {
        "authors": "List of author names in order (e.g., ['Smith J', 'Jones M'])",
        "title": "Title of the article/document",
        "journal_name": "Abbreviated journal name (e.g., 'N Engl J Med')",
        "journal_name_full": "Full journal name",
        "year": "Publication year (4 digits)",
        "volume": "Volume number",
        "issue": "Issue number",
        "supplement": "Supplement information if any",
        "page_range": "Page range (e.g., '123-145' or 'e123')",
        "doi": "DOI number (e.g., '10.1000/xyz123')",
        "column_number": "Column number for multi-column layouts",
        "line_number": "Line number if referenced",
        "table_number": "Table number if content is from a table",
        "figure_number": "Figure number if content is from a figure",
        "edition": "Edition number or name (e.g., '2nd', 'Revised')",
        "publication_city": "City of publication",
        "state": "State/country of publication",
        "publisher": "Publisher name",
        "isbn": "ISBN number",
        "chapter_authors": "Authors of the specific chapter",
        "chapter_title": "Title of the chapter",
        "editors": "Book editor(s) (e.g., ['Smith J', 'Jones M'])",
        "book_title": "Title of the book containing the chapter",
        "website_name": "Name of the website (e.g., 'FDA.gov')",
        "url": "Full URL",
        "accessed_date": "Date the resource was accessed (YYYY-MM-DD)",
        "author_host": "Author or host of the video",
        "distributor": "Video platform/distributor (e.g., 'YouTube')",
        "database_title": "Name of the database (e.g., 'Micromedex')",
        "entry_name": "Name of the drug/entry"
    }

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-opus-4-20250514"):
        self.api_key = api_key or get_anthropic_api_key()
        self.model = model
        self._client = None
        if not self.api_key:
            logger.warning("Anthropic API key not configured - metadata extraction will be disabled")

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def get_text_from_markdown(self, markdown: str, max_words: int = 300) -> str:
        """Extract first N words from markdown text for metadata extraction."""
        return " ".join(markdown.split()[:max_words])

    def get_page0_text(self, chunks: List[Dict]) -> str:
        """Return combined text from chunks where page == 0. (Legacy method)"""
        texts = []
        for chunk in chunks:
            page = chunk.get("grounding", {}).get("page", 0)
            if page == 0:
                texts.append(chunk.get("markdown") or chunk.get("text", ""))
        return "\n\n".join(filter(None, texts))

    def _build_detection_prompt(self, text: str) -> str:
        type_descriptions = "\n".join([
            f"- {doc_type}: {info['description']}"
            for doc_type, info in self.DOCUMENT_TYPES.items()
        ])

        return f"""You are a bibliographic metadata assistant.

Determine the type of this document from the following text (only page 0 shown).

Document types:
{type_descriptions}

Text:
---
{text[:6000]}
---

Return JSON only:
{{
  "document_type": "<type>",
  "confidence": <0.0-1.0>,
  "reasoning": "<brief explanation>"
}}

Valid types: {list(self.DOCUMENT_TYPES.keys())}
"""

    def _build_extraction_prompt(self, text: str, doc_type: str) -> str:
        info = self.DOCUMENT_TYPES[doc_type]
        required = info["required"]
        optional = info["optional"]
        fields = required + optional

        field_desc = "\n".join([
            f"- {f}: {self.FIELD_DESCRIPTIONS.get(f, 'No description available')}" for f in fields
        ])

        return f"""Extract bibliographic metadata from this {doc_type.replace('_', ' ')} (page 0 content only).

Required fields: {required}
Optional fields: {optional}

Field descriptions:
{field_desc}

Text:
---
{text[:8000]}
---

Respond with ONLY valid JSON. Use null for missing fields. Use arrays for authors.
"""

    async def detect_document_type(self, text: str) -> Dict[str, Any]:
        prompt = self._build_detection_prompt(text)
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            result_text = response.content[0].text.strip()
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            result = json.loads(json_match.group() if json_match else result_text)

            if result.get("document_type") not in self.DOCUMENT_TYPES:
                result["document_type"] = "journal_article"
            return result

        except Exception as e:
            logger.error(f"Detection error: {e}")
            return {"document_type": "journal_article", "confidence": 0.0, "reasoning": str(e)}

    async def extract_metadata(self, text: str, doc_type: str) -> Dict[str, Any]:
        if doc_type not in self.DOCUMENT_TYPES:
            doc_type = "journal_article"
        prompt = self._build_extraction_prompt(text, doc_type)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )
            result_text = response.content[0].text.strip()
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            metadata = json.loads(json_match.group() if json_match else result_text)
            metadata["_document_type"] = doc_type
            return metadata

        except Exception as e:
            logger.error(f"Metadata extraction error: {e}")
            return {"_document_type": doc_type, "_error": str(e)}

    async def extract(self, source: Any, max_words: int = 300) -> Dict[str, Any]:
        """Extract metadata from document.

        Args:
            source: Either markdown text (str) or chunks list (List[Dict]) for legacy support.
            max_words: Maximum words to use for extraction (default 300).

        Returns:
            Dict with document_type_info and metadata.
        """
        # Check if API key is configured
        if not self.api_key:
            logger.warning("Skipping metadata extraction - Anthropic API key not configured")
            return {
                "document_type_info": {"document_type": "unknown", "confidence": 0.0},
                "metadata": {}
            }

        # Handle both markdown text and legacy chunks input
        if isinstance(source, str):
            text = self.get_text_from_markdown(source, max_words)
        elif isinstance(source, list):
            # Legacy: extract from chunks
            text = self.get_page0_text(source)
        else:
            logger.warning("Invalid source type for metadata extraction")
            return {
                "document_type_info": {"document_type": "unknown", "confidence": 0.0},
                "metadata": {}
            }

        if not text.strip():
            logger.warning("No content found for metadata extraction")
            return {
                "document_type_info": {"document_type": "unknown", "confidence": 0.0},
                "metadata": {}
            }

        logger.info("Detecting document type...")
        type_info = await self.detect_document_type(text)
        doc_type = type_info.get("document_type", "journal_article")

        logger.info(f"Extracting metadata for type: {doc_type}")
        metadata = await self.extract_metadata(text, doc_type)

        return {
            "document_type_info": type_info,
            "metadata": metadata
        }
