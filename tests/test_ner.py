"""Tests for Named Entity Recognition"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from revisto_evidence_aligned.nlp.ner import NERExtractor, extract_entities
from revisto_evidence_aligned.config import NERConfig


class TestNERExtractor:
    """Test NERExtractor functionality"""
    
    @pytest.fixture
    def ner_config(self):
        """Create test NER configuration"""
        return NERConfig(
            enabled=True,
            model_name="test-ner-model",
            batch_size=8
        )
    
    @pytest.fixture
    def mock_pipeline_results(self):
        """Mock NER pipeline results"""
        return [
            {
                "word": "John Smith",
                "entity_group": "PER",
                "score": 0.95,
                "start": 0,
                "end": 10
            },
            {
                "word": "Apple Inc",
                "entity_group": "ORG",
                "score": 0.92,
                "start": 20,
                "end": 29
            },
            {
                "word": "New York",
                "entity_group": "LOC",
                "score": 0.88,
                "start": 35,
                "end": 43
            }
        ]
    
    def test_init(self, ner_config):
        """Test NERExtractor initialization"""
        extractor = NERExtractor(ner_config)
        assert extractor.config == ner_config
        assert extractor._pipeline is None  # Lazy loading
    
    @patch('revisto_evidence_aligned.nlp.ner.pipeline')
    @patch('revisto_evidence_aligned.nlp.ner.torch.cuda.is_available')
    def test_lazy_loading_cuda(self, mock_cuda, mock_pipeline, ner_config):
        """Test pipeline lazy loading with CUDA available"""
        mock_cuda.return_value = True
        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance
        
        extractor = NERExtractor(ner_config)
        
        # Access pipeline property
        result = extractor.pipeline
        
        assert result == mock_pipeline_instance
        mock_pipeline.assert_called_once_with(
            "ner",
            model=ner_config.model_name,
            aggregation_strategy="simple",
            device=0
        )
    
    @patch('revisto_evidence_aligned.nlp.ner.pipeline')
    @patch('revisto_evidence_aligned.nlp.ner.torch.cuda.is_available')
    def test_lazy_loading_cpu(self, mock_cuda, mock_pipeline, ner_config):
        """Test pipeline lazy loading with CPU only"""
        mock_cuda.return_value = False
        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance
        
        extractor = NERExtractor(ner_config)
        
        # Access pipeline property
        result = extractor.pipeline
        
        mock_pipeline.assert_called_once_with(
            "ner",
            model=ner_config.model_name,
            aggregation_strategy="simple",
            device=-1
        )
    
    @patch('revisto_evidence_aligned.nlp.ner.pipeline')
    def test_extract_entities(self, mock_pipeline, ner_config, mock_pipeline_results):
        """Test entity extraction from text"""
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = mock_pipeline_results
        mock_pipeline.return_value = mock_pipeline_instance
        
        extractor = NERExtractor(ner_config)
        text = "John Smith works at Apple Inc in New York."
        
        entities = extractor.extract(text)
        
        mock_pipeline_instance.assert_called_once_with(text)
        assert len(entities) == 3
        
        # Check first entity
        assert entities[0]["text"] == "John Smith"
        assert entities[0]["type"] == "PER"
        assert entities[0]["score"] == 0.95
        assert entities[0]["start"] == 0
        assert entities[0]["end"] == 10
        
        # Check second entity
        assert entities[1]["text"] == "Apple Inc"
        assert entities[1]["type"] == "ORG"
        
        # Check third entity
        assert entities[2]["text"] == "New York"
        assert entities[2]["type"] == "LOC"
    
    def test_extract_empty_text(self, ner_config):
        """Test extraction with empty text"""
        extractor = NERExtractor(ner_config)
        
        assert extractor.extract("") == []
        assert extractor.extract("   ") == []
        assert extractor.extract(None) == []
    
    @patch('revisto_evidence_aligned.nlp.ner.pipeline')
    def test_extract_error_handling(self, mock_pipeline, ner_config):
        """Test error handling during extraction"""
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.side_effect = Exception("Pipeline error")
        mock_pipeline.return_value = mock_pipeline_instance
        
        extractor = NERExtractor(ner_config)
        
        # Should return empty list on error
        entities = extractor.extract("Test text")
        assert entities == []
    
    @patch('revisto_evidence_aligned.nlp.ner.pipeline')
    def test_extract_batch_single(self, mock_pipeline, ner_config, mock_pipeline_results):
        """Test batch extraction with single text"""
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = mock_pipeline_results
        mock_pipeline.return_value = mock_pipeline_instance
        
        extractor = NERExtractor(ner_config)
        texts = ["John Smith works at Apple Inc."]
        
        results = extractor.extract_batch(texts)
        
        assert len(results) == 1
        assert len(results[0]) == 3  # Three entities
        assert results[0][0]["text"] == "John Smith"
    
    @patch('revisto_evidence_aligned.nlp.ner.pipeline')
    def test_extract_batch_multiple(self, mock_pipeline, ner_config):
        """Test batch extraction with multiple texts"""
        # Mock different results for each text
        batch_results = [
            [{"word": "Entity1", "entity_group": "PER", "score": 0.9, "start": 0, "end": 7}],
            [{"word": "Entity2", "entity_group": "ORG", "score": 0.8, "start": 0, "end": 7}],
            [{"word": "Entity3", "entity_group": "LOC", "score": 0.7, "start": 0, "end": 7}]
        ]
        
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = batch_results
        mock_pipeline.return_value = mock_pipeline_instance
        
        extractor = NERExtractor(ner_config)
        texts = ["Text 1", "Text 2", "Text 3"]
        
        results = extractor.extract_batch(texts)
        
        assert len(results) == 3
        assert results[0][0]["text"] == "Entity1"
        assert results[1][0]["text"] == "Entity2"
        assert results[2][0]["text"] == "Entity3"
    
    @patch('revisto_evidence_aligned.nlp.ner.pipeline')
    def test_extract_batch_large(self, mock_pipeline, ner_config):
        """Test batch extraction with texts larger than batch size"""
        ner_config.batch_size = 2
        extractor = NERExtractor(ner_config)
        
        # Mock pipeline to return appropriate results
        call_count = 0
        def mock_results(texts):
            nonlocal call_count
            call_count += 1
            if isinstance(texts, list):
                return [[{"word": f"Entity{i}", "entity_group": "PER", "score": 0.9, "start": 0, "end": 6}] 
                        for i, _ in enumerate(texts)]
            else:
                return [{"word": f"Entity", "entity_group": "PER", "score": 0.9, "start": 0, "end": 6}]
        
        mock_pipeline_instance = Mock(side_effect=mock_results)
        mock_pipeline.return_value = mock_pipeline_instance
        
        texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]
        results = extractor.extract_batch(texts)
        
        assert len(results) == 5
        # Should be called 3 times (batch_size=2: [2, 2, 1])
        assert mock_pipeline_instance.call_count == 3


class TestExtractEntities:
    """Test the extract_entities helper function"""
    
    @patch('revisto_evidence_aligned.nlp.ner.NERExtractor')
    def test_extract_entities_basic(self, mock_extractor_class):
        """Test basic entity extraction helper"""
        mock_extractor = Mock()
        mock_extractor.extract.return_value = [
            {"text": "Test Entity", "type": "ORG", "score": 0.9, "start": 0, "end": 11}
        ]
        mock_extractor_class.return_value = mock_extractor
        
        text = "Test Entity is here"
        config = NERConfig()
        
        entities = extract_entities(text, config)
        
        mock_extractor_class.assert_called_once_with(config)
        mock_extractor.extract.assert_called_once_with(text)
        assert len(entities) == 1
        assert entities[0]["text"] == "Test Entity"
    
    @patch('revisto_evidence_aligned.nlp.ner.NERExtractor')
    def test_extract_entities_disabled(self, mock_extractor_class):
        """Test extraction when NER is disabled"""
        config = NERConfig(enabled=False)
        
        entities = extract_entities("Some text", config)
        
        # Should not create extractor when disabled
        mock_extractor_class.assert_not_called()
        assert entities == []
    
    def test_extract_entities_no_config(self):
        """Test extraction without config"""
        entities = extract_entities("Some text", None)
        assert entities == []