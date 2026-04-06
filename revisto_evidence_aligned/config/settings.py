"""Configuration settings for the system"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import os
from functools import lru_cache

@lru_cache(maxsize=1)
def _get_es_secrets():
    """Get Elasticsearch secrets from AWS Secrets Manager (cached)"""
    try:
        from ..utils.secrets import get_elasticsearch_config
        return get_elasticsearch_config()
    except Exception:
        # Fallback to empty dict if secrets manager fails
        return {}

@dataclass
class ESConfig:
    """Elasticsearch configuration"""
    url: str = field(default_factory=lambda: os.getenv("ES_URL") or _get_es_secrets().get("es_url") or "https://ccddfe7f3a0949d59ffdec913c9702ab.us-east-1.aws.found.io:443")
    user: str = field(default_factory=lambda: os.getenv("ES_USER", "elastic"))
    password: str = field(default_factory=lambda: os.getenv("ES_PASS", "password"))
    api_key: Optional[str] = field(default_factory=lambda: _get_es_secrets().get("api_key") or os.getenv("API_KEY"))
    index_name: str = field(default_factory=lambda: os.getenv("ES_INDEX", "tf_index_med"))
    alias: str = field(default_factory=lambda: os.getenv("ES_ALIAS", "tf_index_med"))
    shards: int = field(default_factory=lambda: int(os.getenv("ES_SHARDS", "2")))
    replicas: int = field(default_factory=lambda: int(os.getenv("ES_REPLICAS", "1")))
    verify_certs: bool = field(default_factory=lambda: os.getenv("ES_VERIFY_CERTS", "true").lower() == "true")

    @property
    def use_api_key(self) -> bool:
        return self.api_key is not None

@dataclass
class EmbedConfig:
    """Embedding model configuration"""
    model_name: str = field(
        default_factory=lambda: os.getenv(
            "EMBED_MODEL", 
            "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
        )
    )
    normalize: bool = True
    device: str = field(default_factory=lambda: os.getenv("EMBED_DEVICE", "cpu"))
    batch_size: int = 32

@dataclass
class SegmentationConfig:
    """Document segmentation configuration"""
    min_len: int = field(default_factory=lambda: int(os.getenv("SEG_MIN_LEN", "5")))
    drop_empty_bbox: bool = False
    merge_adjacent: bool = True

@dataclass
class NERConfig:
    """Named Entity Recognition configuration"""
    model_name: str = field(
        default_factory=lambda: os.getenv("NER_MODEL", "en_ner_bc5cdr_md")
    )
    enabled: bool = field(default_factory=lambda: os.getenv("NER_ENABLED", "true").lower() == "true")
    batch_size: int = 16

@dataclass
class ScoreConfig:
    """Scoring weights and thresholds for two-stage search pipeline.

    Stage 1 (Retrieval): ES uses semantic similarity
    Stage 2 (Re-ranking): Currently uses ES score directly (numeric boost applied in query)

    Note: The scoring system uses ES score directly which already includes
    numeric token boosting from the Elasticsearch query. The overlap values
    (numeric, entity) are calculated for reference but not used in scoring.
    """
    # Note: These weights are reserved for future weighted scoring but currently unused.
    # The scoring module uses ES score directly which includes numeric boosting.
    # Uncomment and implement in core/scoring.py if weighted re-ranking is needed.
    # w_semantic: float = field(default_factory=lambda: float(os.getenv("W_SEMANTIC", "0.3")))
    # w_numeric: float = field(default_factory=lambda: float(os.getenv("W_NUMERIC", "0.7")))
    # w_entity: float = field(default_factory=lambda: float(os.getenv("W_ENTITY", "0.2")))

    # BM25 normalization
    bm25_norm: float = field(default_factory=lambda: float(os.getenv("BM25_NORM", "1.0")))

    # Thresholds
    threshold: float = field(default_factory=lambda: float(os.getenv("SCORE_THRESHOLD", "0.40")))
    topk: int = field(default_factory=lambda: int(os.getenv("TOPK", "20")))

    # Minimum word count for evidence (filters out very short fragments)
    min_evidence_words: int = field(default_factory=lambda: int(os.getenv("MIN_EVIDENCE_WORDS", "6")))

@dataclass
class ClaudeConfig:
    """Claude API configuration for evidence filtering"""
    model: str = field(default_factory=lambda: os.getenv("CLAUDE_MODEL", "claude-opus-4-20250514"))
    max_tokens_rank: int = field(default_factory=lambda: int(os.getenv("CLAUDE_MAX_TOKENS_RANK", "500")))
    max_tokens_filter: int = field(default_factory=lambda: int(os.getenv("CLAUDE_MAX_TOKENS_FILTER", "500")))
    max_tokens_dedup: int = field(default_factory=lambda: int(os.getenv("CLAUDE_MAX_TOKENS_DEDUP", "400")))
    text_truncation: int = field(default_factory=lambda: int(os.getenv("CLAUDE_TEXT_TRUNCATION", "500")))
    enabled: bool = field(default_factory=lambda: os.getenv("CLAUDE_FILTER_ENABLED", "true").lower() == "true")

@dataclass
class GrpcConfig:
    """gRPC NLP service configuration"""
    host: str = field(default_factory=lambda: os.getenv("NLP_GRPC_HOST", "nlp-grpc"))
    port: int = field(default_factory=lambda: int(os.getenv("NLP_GRPC_PORT", "50051")))

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"


@dataclass
class FigureConfig:
    """Figure processing configuration"""
    enabled: bool = field(default_factory=lambda: os.getenv("FIGURE_ENABLED", "true").lower() == "true")
    model: str = field(default_factory=lambda: os.getenv("FIGURE_MODEL", "claude-opus-4-5-20251101"))


@dataclass
class IndexingConfig:
    """Content type indexing configuration"""
    index_text: bool = field(default_factory=lambda: os.getenv("INDEX_TEXT", "true").lower() == "true")
    index_tables: bool = field(default_factory=lambda: os.getenv("INDEX_TABLES", "true").lower() == "true")
    index_figures: bool = field(default_factory=lambda: os.getenv("INDEX_FIGURES", "true").lower() == "true")

@dataclass
class Config:
    """Main configuration container"""
    es: ESConfig = field(default_factory=ESConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    ner: NERConfig = field(default_factory=NERConfig)
    score: ScoreConfig = field(default_factory=ScoreConfig)
    claude: ClaudeConfig = field(default_factory=ClaudeConfig)
    grpc: GrpcConfig = field(default_factory=GrpcConfig)
    figure: FigureConfig = field(default_factory=FigureConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    nlp_mode: str = field(default_factory=lambda: os.getenv("NLP_MODE", "local"))

def get_config() -> Config:
    """Get the main configuration instance"""
    return Config()