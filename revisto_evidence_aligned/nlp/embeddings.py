"""Embedding generation module"""

from typing import List, Union, Optional, TYPE_CHECKING
import numpy as np

from ..config import EmbedConfig
from ..utils.logging import get_logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)


class EmbeddingModel:
    """Wrapper for sentence transformer models"""

    def __init__(self, config: EmbedConfig):
        self.config = config
        self._model = None

    @property
    def model(self) -> "SentenceTransformer":
        """Lazy load the model"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.config.model_name}")
            self._model = SentenceTransformer(
                self.config.model_name,
                device=self.config.device
            )
        return self._model
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        show_progress: bool = False
    ) -> np.ndarray:
        """Encode texts to embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        
        batch_size = batch_size or self.config.batch_size
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.config.normalize
        )
        
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text"""
        return self.encode(text)[0]
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        return self.model.get_sentence_embedding_dimension()


def load_embedder(config: Optional[EmbedConfig] = None) -> EmbeddingModel:
    """Load embedding model with configuration"""
    if config is None:
        config = EmbedConfig()
    return EmbeddingModel(config)