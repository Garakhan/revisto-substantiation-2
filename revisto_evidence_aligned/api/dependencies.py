"""Shared dependencies and resources for FastAPI routes"""

from typing import Dict, Any

# Global resources dictionary
resources: Dict[str, Any] = {}


def get_resources() -> Dict[str, Any]:
    """Get shared resources"""
    return resources