"""NLP processing modules"""

from .embeddings import EmbeddingModel, load_embedder
from .ner import NERExtractor, extract_entities
from .tokenization import tokenize, extract_numbers
from .segmentation import segment_text, merge_adjacent_segments

__all__ = [
    "EmbeddingModel",
    "load_embedder",
    "NERExtractor",
    "extract_entities",
    "tokenize",
    "extract_numbers",
    "segment_text",
    "merge_adjacent_segments",
]
