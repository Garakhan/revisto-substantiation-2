"""Elasticsearch index mappings and settings"""

from typing import Dict, Any

from ..config import ESConfig, EmbedConfig


def get_index_settings(config: ESConfig) -> Dict[str, Any]:
    """Get Elasticsearch index settings"""
    return {
        "number_of_shards": config.shards,
        "number_of_replicas": config.replicas,
        "analysis": {
            "analyzer": {
                "text_analyzer": {
                    "type": "standard",
                    "stopwords": "_english_"
                }
            }
        }
    }


def get_index_mapping(embed_config: EmbedConfig) -> Dict[str, Any]:
    """Get Elasticsearch index mapping for reference documents"""
    # Get embedding dimension
    embed_dim = 768  # Default for BERT-based models
    if "bert" in embed_config.model_name.lower():
        embed_dim = 768
    elif "minilm" in embed_config.model_name.lower():
        embed_dim = 384
    
    return {
        "properties": {
            # Document type: "segment" or "ref_metadata"
            "doc_type": {"type": "keyword"},

            # Document identifiers
            "ref_id": {"type": "keyword"},
            "ref_title": {"type": "text", "analyzer": "text_analyzer"},
            "sent_id": {"type": "keyword"},
            "org_id": {"type": "integer"},
            "brand_id": {"type": "integer"},
            
            # Document content
            "text": {
                "type": "text",
                "analyzer": "text_analyzer",
                "fields": {
                    "keyword": {"type": "keyword", "ignore_above": 256}
                }
            },
            "section": {"type": "keyword"},
            "page": {"type": "integer"},
            
            # Bounding box
            "bbox": {
                "properties": {
                    "x0": {"type": "float"},
                    "y0": {"type": "float"},
                    "x1": {"type": "float"},
                    "y1": {"type": "float"}
                }
            },
            
            # Paragraph and sentence numbering
            "paragraph_number": {"type": "integer"},
            "sentence_number": {"type": "integer"},

            # Source info for tables/figures (e.g., "Table 1; row 12" or "Figure 2; claim 1")
            # When present, this is used instead of paragraph_number in output
            "source": {"type": "keyword"},

            # Extracted features
            "numeric_tokens": {"type": "keyword"},
            "entities": {"type": "keyword"},
            "entity_types": {"type": "keyword"},
            
            # Embedding vector
            "vector": {
                "type": "dense_vector",
                "dims": embed_dim,
                "index": True,
                "similarity": "cosine"
            },
            
            # Metadata
            "labels": {"type": "keyword"},
            "timestamp": {"type": "date"},
        }
    }


def get_claim_mapping() -> Dict[str, Any]:
    """Get mapping for claim documents"""
    return {
        "properties": {
            "claim_id": {"type": "keyword"},
            "claim_text": {
                "type": "text",
                "analyzer": "text_analyzer"
            },
            "tactic_id": {"type": "keyword"},
            "page": {"type": "integer"},
            "bbox": {
                "properties": {
                    "x0": {"type": "float"},
                    "y0": {"type": "float"},
                    "x1": {"type": "float"},
                    "y1": {"type": "float"}
                }
            }
        }
    }