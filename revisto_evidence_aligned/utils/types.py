"""Common type definitions"""

from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass

BBox = Tuple[float, float, float, float]  # (x0, y0, x1, y1)

@dataclass
class PageGeometry:
    """Page geometry information"""
    width: float
    height: float
    page_num: int

@dataclass
class Segment:
    """Document segment with text and metadata"""
    text: str
    page: int
    bbox: tuple
    section: str = "body"
    segment_type: str = "text"  # "text" or "table"
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ClaimSegment(Segment):
    """Segment representing a claim"""
    claim_id: str = ""
    score: float = 0.0

@dataclass
class ReferenceSegment(Segment):
    """Segment from a reference document"""
    ref_id: str = ""
    ref_title: str = ""
    sent_id: str = ""