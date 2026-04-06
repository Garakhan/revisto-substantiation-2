"""Command-line interfaces"""

from .index import main as index_main
from .search import main as search_main

__all__ = ["index_main", "search_main"]