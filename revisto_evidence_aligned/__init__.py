"""
Revisto Evidence Aligned - A modular evidence claim extraction system
"""

__version__ = "0.2.0"

from .config import get_config
from .core.indexing import index_references
from .core.searching import search_claims

__all__ = ["get_config", "index_references", "search_claims"]