"""gRPC client for NLP service.

This module provides gRPC client wrappers that match the interface of the local
EmbeddingModel and NERExtractor classes, allowing seamless switching between
local and remote model execution.
"""

import os
from typing import List, Dict, Any, Optional, Union

import grpc
import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Default gRPC server address
DEFAULT_NLP_GRPC_HOST = os.getenv("NLP_GRPC_HOST", "localhost")
DEFAULT_NLP_GRPC_PORT = os.getenv("NLP_GRPC_PORT", "50051")
DEFAULT_NLP_GRPC_ADDRESS = f"{DEFAULT_NLP_GRPC_HOST}:{DEFAULT_NLP_GRPC_PORT}"


class NLPGrpcClient:
    """gRPC client for the NLP service."""

    def __init__(self, address: str = None, secure: bool = False, credentials: grpc.ChannelCredentials = None):
        """
        Initialize the gRPC client.

        Args:
            address: Server address (host:port). Defaults to NLP_GRPC_HOST:NLP_GRPC_PORT env vars.
            secure: Whether to use SSL/TLS.
            credentials: gRPC credentials for secure connection.
        """
        self.address = address or DEFAULT_NLP_GRPC_ADDRESS
        self.secure = secure
        self.credentials = credentials
        self._channel = None
        self._stub = None

    @property
    def channel(self) -> grpc.Channel:
        """Get or create the gRPC channel."""
        if self._channel is None:
            if self.secure:
                if self.credentials is None:
                    self.credentials = grpc.ssl_channel_credentials()
                self._channel = grpc.insecure_channel(self.address)
            else:
                self._channel = grpc.insecure_channel(self.address)
            logger.info(f"Connected to NLP gRPC server at {self.address}")
        return self._channel

    @property
    def stub(self):
        """Get or create the gRPC stub."""
        if self._stub is None:
            # Import generated stubs
            from .generated.nlp import nlp_pb2_grpc
            self._stub = nlp_pb2_grpc.NLPServiceStub(self.channel)
        return self._stub

    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: List of texts to encode.
            normalize: Whether to normalize embeddings.

        Returns:
            numpy array of embeddings with shape (len(texts), dimension)
        """
        from .generated.nlp import nlp_pb2

        request = nlp_pb2.EmbedRequest(texts=texts, normalize=normalize)
        response = self.stub.Encode(request)

        # Convert to numpy array
        embeddings = []
        for emb in response.embeddings:
            embeddings.append(list(emb.values))

        return np.array(embeddings, dtype=np.float32)

    def extract_entities(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """
        Extract named entities from texts.

        Args:
            texts: List of texts to process.

        Returns:
            List of entity lists, one per input text.
        """
        from .generated.nlp import nlp_pb2

        request = nlp_pb2.NERRequest(texts=texts)
        response = self.stub.ExtractEntities(request)

        results = []
        for text_entities in response.results:
            entities = []
            for ent in text_entities.entities:
                entities.append({
                    "text": ent.text,
                    "type": ent.type,
                    "score": ent.score,
                    "start": ent.start,
                    "end": ent.end
                })
            results.append(entities)

        return results

    def sentencise(self, texts: List[str]) -> List[List[str]]:
        """
        Split texts into sentences using stanza.

        Args:
            texts: List of texts to split.

        Returns:
            List of sentence lists, one per input text.
        """
        from .generated.nlp import nlp_pb2

        request = nlp_pb2.SentenciseRequest(texts=texts)
        response = self.stub.Sentencise(request)

        results = []
        for text_sentences in response.results:
            results.append(list(text_sentences.sentences))

        return results

    def extract_numeric_entities(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Extract numeric entities from texts using Stanza NER.

        Args:
            texts: List of texts to process.

        Returns:
            List of dicts with 'entities' (full entity info) and 'tokens' (just text values).
        """
        from .generated.nlp import nlp_pb2

        request = nlp_pb2.NumericRequest(texts=texts)
        response = self.stub.ExtractNumericEntities(request)

        results = []
        for text_result in response.results:
            entities = []
            for ent in text_result.entities:
                entities.append({
                    "text": ent.text,
                    "type": ent.type,
                    "start": ent.start,
                    "end": ent.end
                })
            results.append({
                "entities": entities,
                "tokens": list(text_result.tokens)
            })

        return results

    def health_check(self) -> Dict[str, Any]:
        """Check server health."""
        from .generated.nlp import nlp_pb2

        request = nlp_pb2.HealthRequest()
        response = self.stub.Health(request)

        return {
            "healthy": response.healthy,
            "embedding_model": response.embedding_model,
            "ner_model": response.ner_model,
            "models_loaded": response.models_loaded
        }

    def close(self):
        """Close the gRPC channel."""
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None


class GrpcEmbeddingModel:
    """
    gRPC-based embedding model that matches the interface of EmbeddingModel.

    This allows seamless switching between local and remote model execution.
    """

    def __init__(self, config=None, grpc_address: str = None):
        """
        Initialize the gRPC embedding model.

        Args:
            config: EmbedConfig (for compatibility, some settings may be ignored).
            grpc_address: gRPC server address. Defaults to env vars.
        """
        self.config = config
        self._client = NLPGrpcClient(address=grpc_address)
        self._dimension = None

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        show_progress: bool = False
    ) -> np.ndarray:
        """Encode texts to embeddings."""
        if isinstance(texts, str):
            texts = [texts]

        normalize = self.config.normalize if self.config else True
        return self._client.encode(texts, normalize=normalize)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text."""
        return self.encode(text)[0]

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is None:
            # Get dimension from health check or by encoding a test string
            try:
                health = self._client.health_check()
                # Encode a test string to get dimension
                test_emb = self._client.encode(["test"])
                self._dimension = test_emb.shape[1]
            except Exception as e:
                logger.warning(f"Could not get embedding dimension: {e}")
                self._dimension = 768  # Default for BERT-based models
        return self._dimension

    def close(self):
        """Close the gRPC connection."""
        self._client.close()


class GrpcNERExtractor:
    """
    gRPC-based NER extractor that matches the interface of NERExtractor.

    This allows seamless switching between local and remote model execution.
    """

    def __init__(self, config=None, grpc_address: str = None):
        """
        Initialize the gRPC NER extractor.

        Args:
            config: NERConfig (for compatibility, some settings may be ignored).
            grpc_address: gRPC server address. Defaults to env vars.
        """
        self.config = config
        self._client = NLPGrpcClient(address=grpc_address)

    def extract(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text."""
        if not text or not text.strip():
            return []

        results = self._client.extract_entities([text])
        return results[0] if results else []

    def extract_batch(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """Extract entities from multiple texts."""
        return self._client.extract_entities(texts)

    def get_entity_types(self) -> List[str]:
        """Get available entity types."""
        # Common biomedical entity types
        return [
            "DRUG", "DISEASE", "GENE", "PROTEIN", "CHEMICAL",
            "ORGANISM", "CELL_LINE", "CELL_TYPE", "DNA", "RNA"
        ]

    def close(self):
        """Close the gRPC connection."""
        self._client.close()


class GrpcSentenciser:
    """
    gRPC-based sentence splitter using stanza on the server.

    This allows sentence tokenization without loading stanza locally.
    """

    def __init__(self, grpc_address: str = None):
        """
        Initialize the gRPC sentenciser.

        Args:
            grpc_address: gRPC server address. Defaults to env vars.
        """
        self._client = NLPGrpcClient(address=grpc_address)

    def sentencise(self, text: str) -> List[str]:
        """Split text into sentences."""
        if not text or not text.strip():
            return []

        results = self._client.sentencise([text])
        return results[0] if results else []

    def sentencise_batch(self, texts: List[str]) -> List[List[str]]:
        """Split multiple texts into sentences."""
        return self._client.sentencise(texts)

    def close(self):
        """Close the gRPC connection."""
        self._client.close()


class GrpcNumericExtractor:
    """
    gRPC-based numeric entity extractor using Stanza NER on the server.

    Extracts numeric entities (CARDINAL, PERCENT, QUANTITY, MONEY, ORDINAL, DATE, TIME)
    without loading Stanza locally.
    """

    def __init__(self, grpc_address: str = None):
        """
        Initialize the gRPC numeric extractor.

        Args:
            grpc_address: gRPC server address. Defaults to env vars.
        """
        self._client = NLPGrpcClient(address=grpc_address)

    def extract(self, text: str) -> List[str]:
        """
        Extract numeric tokens from text.

        Args:
            text: Text to extract numbers from.

        Returns:
            List of numeric token strings (e.g., ["45%", "500mg", "2024"]).
        """
        if not text or not text.strip():
            return []

        results = self._client.extract_numeric_entities([text])
        return results[0]["tokens"] if results else []

    def extract_with_types(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract numeric entities with full type information.

        Args:
            text: Text to extract numbers from.

        Returns:
            List of entity dicts with text, type, start, end.
        """
        if not text or not text.strip():
            return []

        results = self._client.extract_numeric_entities([text])
        return results[0]["entities"] if results else []

    def extract_batch(self, texts: List[str]) -> List[List[str]]:
        """Extract numeric tokens from multiple texts."""
        results = self._client.extract_numeric_entities(texts)
        return [r["tokens"] for r in results]

    def close(self):
        """Close the gRPC connection."""
        self._client.close()


# Factory functions for easy switching between local and gRPC models

def load_embedder_grpc(config=None, grpc_address: str = None) -> GrpcEmbeddingModel:
    """Load gRPC-based embedding model."""
    return GrpcEmbeddingModel(config=config, grpc_address=grpc_address)


def load_ner_extractor_grpc(config=None, grpc_address: str = None) -> GrpcNERExtractor:
    """Load gRPC-based NER extractor."""
    return GrpcNERExtractor(config=config, grpc_address=grpc_address)


def load_numeric_extractor_grpc(grpc_address: str = None) -> GrpcNumericExtractor:
    """Load gRPC-based numeric entity extractor."""
    return GrpcNumericExtractor(grpc_address=grpc_address)