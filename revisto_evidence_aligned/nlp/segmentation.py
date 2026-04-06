"""Document segmentation utilities"""

from typing import List, Optional
from ..utils.types import Segment
from ..config import SegmentationConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


def segment_text(
    text: str,
    config: Optional[SegmentationConfig] = None
) -> List[Segment]:
    """Segment plain text into paragraphs"""
    if config is None:
        config = SegmentationConfig()

    segments = []

    # Split by double newlines for paragraphs
    paragraphs = text.split('\n\n')

    for i, para in enumerate(paragraphs):
        para = para.strip()
        if len(para) >= config.min_len:
            segment = Segment(
                text=para,
                page=1,
                bbox=(0, 0, 0, 0),
                section="paragraph",
                metadata={"paragraph_index": i}
            )
            segments.append(segment)

    return segments


def merge_adjacent_segments(
    segments: List[Segment],
    max_gap: float = 10.0
) -> List[Segment]:
    """Merge segments that are adjacent on the same page"""
    if not segments:
        return []

    # Sort by page and vertical position
    sorted_segments = sorted(
        segments,
        key=lambda s: (s.page, s.bbox[1])  # page, top coordinate
    )

    merged = []
    current = None

    for segment in sorted_segments:
        if current is None:
            current = segment
        elif (
            current.page == segment.page and
            segment.bbox[1] - current.bbox[3] <= max_gap  # Vertical gap
        ):
            # Merge segments
            current = Segment(
                text=current.text + " " + segment.text,
                page=current.page,
                bbox=(
                    min(current.bbox[0], segment.bbox[0]),  # leftmost
                    current.bbox[1],  # top of first
                    max(current.bbox[2], segment.bbox[2]),  # rightmost
                    segment.bbox[3]  # bottom of second
                ),
                section=current.section,
                metadata={**current.metadata, **segment.metadata}
            )
        else:
            merged.append(current)
            current = segment

    if current:
        merged.append(current)

    return merged
