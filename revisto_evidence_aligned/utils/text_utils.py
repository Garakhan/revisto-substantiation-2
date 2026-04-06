"""Shared text processing utilities for encoding fixes and sanitization."""

import re
from typing import Optional

# Regex to match invalid XML characters (control chars except tab, newline, carriage return)
INVALID_XML_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')

# Common encoding fixes for mojibake (UTF-8 decoded as Windows-1252/Latin-1)
# This handles text that was UTF-8 encoded but incorrectly read as Windows-1252
ENCODING_FIXES = {
    # Quotes
    'â€™': "'",           # Right single quote (U+2019)
    'â€˜': "'",           # Left single quote (U+2018)
    'â€œ': '"',           # Left double quote (U+201C)
    'â€\u009d': '"',      # Right double quote (U+201D)
    'â€': '"',            # Truncated right double quote
    # Dashes
    'â€"': '—',           # Em dash (U+2014)
    'â€"': '–',           # En dash (U+2013)
    '\xe2\x80\x94': '—',  # Em dash as raw bytes
    '\xe2\x80\x93': '–',  # En dash as raw bytes
    # Math symbols
    'â‰¤': '≤',           # Less than or equal
    'â‰¥': '≥',           # Greater than or equal
    # Accented characters
    'Ã©': 'é',
    'Ã¨': 'è',
    'Ã¼': 'ü',
    'Ã¶': 'ö',
    'Ã¤': 'ä',
    'Ã±': 'ñ',
    'Ãº': 'ú',
    'Ã³': 'ó',
    'Ã­': 'í',
    'Ã¡': 'á',
    # Artifacts and special chars
    'Â': '',              # Artifact from double-encoding
    '�': '',              # Replacement character (indicates encoding failure)
    '\u200b': '',         # Zero-width space
    '\u00a0': ' ',        # Non-breaking space
    '\ufeff': '',         # BOM
}


def fix_text_encoding(text: str) -> str:
    """Fix common encoding issues in text (mojibake from UTF-8/Windows-1252 mixup).

    Args:
        text: Text that may have encoding issues

    Returns:
        Text with encoding issues fixed
    """
    if not text:
        return ""

    text = str(text)

    # Try to fix mojibake by re-encoding
    try:
        # If text was UTF-8 misread as Latin-1, this will fix it
        text = text.encode('latin-1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass  # Already properly decoded or unfixable

    # Apply character-level fixes
    for bad, good in ENCODING_FIXES.items():
        text = text.replace(bad, good)

    # Remove any remaining invalid XML characters
    text = INVALID_XML_CHARS.sub('', text)

    return text.strip()


def sanitize_for_xml(text: str) -> str:
    """Remove characters that are invalid in XML/XLSX.

    Args:
        text: Text to sanitize

    Returns:
        Text with invalid XML characters removed
    """
    if not text:
        return ""
    return INVALID_XML_CHARS.sub('', text)


def normalize_whitespace(text: str) -> str:
    """Normalize line endings and whitespace in text.

    Args:
        text: Text to normalize

    Returns:
        Text with normalized whitespace
    """
    if not text:
        return ""

    text = str(text)
    # Normalize line endings
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\r', '\n', text)

    return text.strip()
