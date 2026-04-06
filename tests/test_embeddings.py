"""Tests for embedding generation"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from revisto_evidence_aligned.nlp.embeddings import EmbeddingModel, load_embedder
from revisto_evidence_aligned.config import EmbedConfig


class TestEmbeddingModel:
    """Test EmbeddingModel functionality"""
    
    @pytest.fixture
    def embed_config(self):
        """Create a test embedding config"""
        return EmbedConfig(
            model_name="test-model",
            device="cpu",
            batch_size=16,
            normalize=True
        )
    
    @pytest.fixture
    def mock_sentence_transformer(self):
        """Create a mock SentenceTransformer"""
        mock = MagicMock()
        mock.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        return mock
    
    def test_init(self, embed_config):
        """Test EmbeddingModel initialization"""
        model = EmbeddingModel(embed_config)
        assert model.config == embed_config
        assert model._model is None  # Lazy loading
    
    @patch('revisto_evidence_aligned.nlp.embeddings.SentenceTransformer')
    def test_lazy_loading(self, mock_st_class, embed_config):
        """Test model lazy loading on first access"""
        mock_instance = MagicMock()
        mock_st_class.return_value = mock_instance
        
        model = EmbeddingModel(embed_config)
        assert model._model is None
        
        # Access model property
        result = model.model
        
        assert result == mock_instance
        assert model._model == mock_instance
        mock_st_class.assert_called_once_with(
            embed_config.model_name,
            device=embed_config.device
        )
    
    @patch('revisto_evidence_aligned.nlp.embeddings.SentenceTransformer')
    def test_encode_single_text(self, mock_st_class, embed_config, mock_sentence_transformer):
        """Test encoding a single text"""
        mock_st_class.return_value = mock_sentence_transformer
        
        model = EmbeddingModel(embed_config)
        text = "This is a test sentence"
        
        result = model.encode(text)
        
        mock_sentence_transformer.encode.assert_called_once_with(
            [text],
            batch_size=16,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        assert isinstance(result, np.ndarray)
    
    @patch('revisto_evidence_aligned.nlp.embeddings.SentenceTransformer')
    def test_encode_multiple_texts(self, mock_st_class, embed_config, mock_sentence_transformer):
        """Test encoding multiple texts"""
        mock_st_class.return_value = mock_sentence_transformer
        
        model = EmbeddingModel(embed_config)
        texts = ["First sentence", "Second sentence", "Third sentence"]
        
        result = model.encode(texts)
        
        mock_sentence_transformer.encode.assert_called_once_with(
            texts,
            batch_size=16,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        assert isinstance(result, np.ndarray)
    
    @patch('revisto_evidence_aligned.nlp.embeddings.SentenceTransformer')
    def test_encode_custom_batch_size(self, mock_st_class, embed_config, mock_sentence_transformer):
        """Test encoding with custom batch size"""
        mock_st_class.return_value = mock_sentence_transformer
        
        model = EmbeddingModel(embed_config)
        texts = ["Test sentence"]
        
        result = model.encode(texts, batch_size=32)
        
        mock_sentence_transformer.encode.assert_called_once_with(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True
        )
    
    @patch('revisto_evidence_aligned.nlp.embeddings.SentenceTransformer')
    def test_encode_with_progress(self, mock_st_class, embed_config, mock_sentence_transformer):
        """Test encoding with progress bar"""
        mock_st_class.return_value = mock_sentence_transformer
        
        model = EmbeddingModel(embed_config)
        texts = ["Test sentence"]
        
        result = model.encode(texts, show_progress=True)
        
        mock_sentence_transformer.encode.assert_called_once_with(
            texts,
            batch_size=16,
            show_progress_bar=True,
            normalize_embeddings=True
        )
    
    @patch('revisto_evidence_aligned.nlp.embeddings.SentenceTransformer')
    def test_model_reuse(self, mock_st_class, embed_config):
        """Test that model is only loaded once"""
        mock_instance = MagicMock()
        mock_st_class.return_value = mock_instance
        
        model = EmbeddingModel(embed_config)
        
        # Access model multiple times
        result1 = model.model
        result2 = model.model
        result3 = model.model
        
        # Should only be created once
        mock_st_class.assert_called_once()
        assert result1 == result2 == result3 == mock_instance


class TestLoadEmbedder:
    """Test load_embedder function"""
    
    def test_load_embedder_returns_embedding_model(self):
        """Test that load_embedder returns an EmbeddingModel instance"""
        config = EmbedConfig(model_name="test-model")
        result = load_embedder(config)
        
        assert isinstance(result, EmbeddingModel)
        assert result.config == config
    
    def test_load_embedder_with_custom_config(self):
        """Test load_embedder with custom configuration"""
        config = EmbedConfig(
            model_name="custom-model",
            device="cuda",
            batch_size=32,
            normalize=False
        )
        result = load_embedder(config)
        
        assert result.config.model_name == "custom-model"
        assert result.config.device == "cuda"
        assert result.config.batch_size == 32
        assert result.config.normalize == False


@pytest.mark.integration
class TestEmbeddingModelIntegration:
    """Integration tests for EmbeddingModel (requires actual model)"""
    
    @pytest.mark.skipif(True, reason="Requires downloading actual model")
    def test_real_encoding(self):
        """Test with actual sentence transformer model"""
        config = EmbedConfig(
            model_name="all-MiniLM-L6-v2",  # Small model for testing
            device="cpu"
        )
        model = EmbeddingModel(config)
        
        texts = ["Hello world", "This is a test"]
        embeddings = model.encode(texts)
        
        assert embeddings.shape[0] == 2  # Two texts
        assert embeddings.shape[1] > 0   # Has dimensions
        assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)  # Normalized