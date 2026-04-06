"""Text tokenization and processing utilities.

Primary classes/functions for production use:
- StanzaNumericExtractor: NER-based numeric entity extraction

Legacy/testing utilities (not used in main search pipeline):
- tokenize(): Basic word tokenization
- extract_numbers(): Regex-based number extraction (fallback)
- extract_stopwords(), remove_stopwords(): Stopword handling
"""

import re
from typing import List, Set, Optional, Dict, Any


def tokenize(text: str, lowercase: bool = True) -> List[str]:
    """Simple word tokenization.

    Note: This is a basic utility function used primarily for testing.
    The main search pipeline uses Elasticsearch's tokenization.
    """
    if lowercase:
        text = text.lower()

    # Simple regex-based tokenization
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def extract_numbers_regex(text: str) -> List[str]:
    """Extract numeric tokens from text using regex patterns (legacy method)."""
    # Pattern to match various number formats
    patterns = [
        r'\b\d+\.?\d*%\b',  # Percentages
        r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b',  # Numbers with commas
        r'\b\d+(?:\.\d+)?\b',  # Simple decimals
        r'\bp\s*[<>=]\s*\d+(?:\.\d+)?\b',  # p-values
        r'\b\d+(?:\.\d+)?\s*(?:mg|g|kg|ml|l|mm|cm|m|km|°C|°F)\b',  # Units
    ]

    numbers = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        numbers.extend(matches)

    # Remove duplicates while preserving order
    seen = set()
    unique_numbers = []
    for num in numbers:
        normalized = num.lower().replace(' ', '')
        if normalized not in seen:
            seen.add(normalized)
            unique_numbers.append(num)

    return unique_numbers


# Keep backward compatibility alias
def extract_numbers(text: str) -> List[str]:
    """Extract numeric tokens from text (legacy regex-based method).

    For Stanza-based extraction, use StanzaNumericExtractor or GrpcNumericExtractor.
    """
    return extract_numbers_regex(text)


class StanzaNumericExtractor:
    """
    Local Stanza-based numeric entity extractor.

    Extracts numeric entities (CARDINAL, PERCENT, QUANTITY, MONEY, ORDINAL, DATE, TIME)
    using Stanza NER pipeline.
    """

    # Stanza entity types that represent numeric values
    NUMERIC_ENTITY_TYPES = {"CARDINAL", "PERCENT", "QUANTITY", "MONEY", "ORDINAL", "DATE", "TIME"}

    def __init__(self, lang: str = "en"):
        """
        Initialize the Stanza numeric extractor.

        Args:
            lang: Language code for Stanza pipeline.
        """
        self._pipeline = None
        self._lang = lang

    @property
    def pipeline(self):
        """Lazy load the Stanza NER pipeline."""
        if self._pipeline is None:
            import stanza
            self._pipeline = stanza.Pipeline(
                self._lang,
                processors="tokenize,ner"
            )
        return self._pipeline

    def extract(self, text: str) -> List[str]:
        """
        Extract numeric tokens from text.

        Args:
            text: Text to extract numbers from.

        Returns:
            List of numeric token strings (e.g., ["45%", "500mg", "2024"]).
        """
        if not text or not text.strip():
            return []

        doc = self.pipeline(text)

        tokens = []
        seen = set()

        for ent in doc.ents:
            if ent.type in self.NUMERIC_ENTITY_TYPES:
                # Deduplicate
                normalized = ent.text.lower().replace(' ', '')
                if normalized not in seen:
                    seen.add(normalized)
                    tokens.append(ent.text)

        return tokens

    def extract_with_types(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract numeric entities with full type information.

        Args:
            text: Text to extract numbers from.

        Returns:
            List of entity dicts with text, type, start, end.
        """
        if not text or not text.strip():
            return []

        doc = self.pipeline(text)

        entities = []
        for ent in doc.ents:
            if ent.type in self.NUMERIC_ENTITY_TYPES:
                entities.append({
                    "text": ent.text,
                    "type": ent.type,
                    "start": ent.start_char,
                    "end": ent.end_char
                })

        return entities

    def extract_batch(self, texts: List[str]) -> List[List[str]]:
        """Extract numeric tokens from multiple texts."""
        return [self.extract(text) for text in texts]


def extract_stopwords() -> Set[str]:
    """Get a basic set of English stopwords.

    Note: Currently unused in the main pipeline. Kept for potential
    future use and testing purposes.
    """
    return {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'will', 'with', 'the', 'this', 'these', 'those',
        'i', 'you', 'we', 'they', 'them', 'their', 'what', 'which', 'who',
        'when', 'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more',
        'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than',
        'too', 'very', 'can', 'could', 'may', 'might', 'must', 'shall', 'should',
        'would', 'am', 'is', 'are', 'was', 'were', 'been', 'being', 'have', 'has',
        'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
        'might', 'must', 'shall', 'can', 'need', 'ought', 'dare', 'used',
    }


def remove_stopwords(tokens: List[str], stopwords: Optional[Set[str]] = None) -> List[str]:
    """Remove stopwords from token list"""
    if stopwords is None:
        stopwords = extract_stopwords()

    return [token for token in tokens if token.lower() not in stopwords]