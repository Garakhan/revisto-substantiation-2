"""FastAPI application for evidence alignment"""

from .app import create_app
from .models import *

__all__ = ["create_app"]