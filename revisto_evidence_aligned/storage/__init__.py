"""Storage backend modules"""

from .elasticsearch import (
    ESClient,
    create_index,
    bulk_index,
    search_documents,
)
from .mappings import (
    get_index_mapping,
    get_index_settings,
)

__all__ = [
    "ESClient",
    "create_index",
    "bulk_index",
    "search_documents",
    "get_index_mapping",
    "get_index_settings",
]