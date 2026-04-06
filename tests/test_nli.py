"""Tests for Natural Language Inference"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List

from revisto_evidence_aligned.nlp.nli import NLIModel, compute_nli_scores
from revisto_evidence_aligned.config import NLIConfig


class TestNLIModel:
    """Test NLIModel functionality"""
    
    @pytest.fixture
    def nli_config(self):
        """Create test NLI configuration"""
        return NLIConfig(
            enabled=True,
            model_name="test-nli-model",
            device="cpu",
            batch_size=4
        )
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model"""
        model = Mock()
        model.config.id2label = {0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}
        return model
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer"""
        tokenizer = Mock()
        # Mock tokenizer visual_claims
        mock_output = MagicMock()
        mock_output.to.return_value = mock_output
        tokenizer.return_value = mock_output
        return tokenizer
    
    def test_init(self, nli_config):
        """Test NLIModel initialization"""
        model = NLIModel(nli_config)
        assert model.config == nli_config
        assert model._model is None
        assert model._tokenizer is None
        assert model._device is None
        assert model.label_map == {
            "ENTAILMENT": "entailment",
            "NEUTRAL": "neutral",
            "CONTRADICTION": "contradiction"
        }
    
    @patch('revisto_evidence_aligned.nlp.nli.torch.cuda.is_available')
    def test_device_property_auto_cuda(self, mock_cuda, nli_config):
        """Test device property with auto selection and CUDA available"""
        mock_cuda.return_value = True
        nli_config.device = "auto"
        model = NLIModel(nli_config)
        
        device = model.device
        assert device.type == "cuda"
    
    @patch('revisto_evidence_aligned.nlp.nli.torch.cuda.is_available')
    def test_device_property_auto_cpu(self, mock_cuda, nli_config):
        """Test device property with auto selection and no CUDA"""
        mock_cuda.return_value = False
        nli_config.device = "auto"
        model = NLIModel(nli_config)
        
        device = model.device
        assert device.type == "cpu"
    
    def test_device_property_explicit(self, nli_config):
        """Test device property with explicit selection"""
        nli_config.device = "cpu"
        model = NLIModel(nli_config)
        
        device = model.device
        assert device.type == "cpu"
    
    @patch('revisto_evidence_aligned.nlp.nli.AutoModelForSequenceClassification')
    @patch('revisto_evidence_aligned.nlp.nli.AutoTokenizer')
    def test_load_model(self, mock_tokenizer_class, mock_model_class, nli_config, mock_model, mock_tokenizer):
        """Test model loading"""
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model
        
        nli_model = NLIModel(nli_config)
        nli_model.load_model()
        
        # Verify model and tokenizer loaded
        mock_tokenizer_class.from_pretrained.assert_called_once_with(nli_config.model_name)
        mock_model_class.from_pretrained.assert_called_once_with(nli_config.model_name)
        mock_model.to.assert_called_once()
        mock_model.eval.assert_called_once()
        
        assert nli_model._model == mock_model
        assert nli_model._tokenizer == mock_tokenizer
    
    @patch('revisto_evidence_aligned.nlp.nli.AutoModelForSequenceClassification')
    @patch('revisto_evidence_aligned.nlp.nli.AutoTokenizer')
    def test_compute_entailment(self, mock_tokenizer_class, mock_model_class, nli_config):
        """Test single entailment computation"""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.config.id2label = {0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}
        mock_model.to.return_value = mock_model
        
        # Mock model visual_claims
        mock_output = Mock()
        mock_logits = torch.tensor([[2.0, 0.5, -1.0]])  # Favors entailment
        mock_output.logits = mock_logits
        mock_model.return_value = mock_output
        
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Test
        nli_model = NLIModel(nli_config)
        premise = "The cat is on the mat."
        hypothesis = "There is a cat."
        
        scores = nli_model.compute_entailment(premise, hypothesis)
        
        # Verify tokenizer called correctly
        mock_tokenizer.assert_called_once_with(
            premise,
            hypothesis,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        # Verify scores
        assert "entailment" in scores
        assert "neutral" in scores
        assert "contradiction" in scores
        assert scores["entailment"] > scores["neutral"] > scores["contradiction"]
        assert abs(sum(scores.values()) - 1.0) < 0.001  # Should sum to ~1
    
    @patch('revisto_evidence_aligned.nlp.nli.AutoModelForSequenceClassification')
    @patch('revisto_evidence_aligned.nlp.nli.AutoTokenizer')
    def test_compute_batch_entailment(self, mock_tokenizer_class, mock_model_class, nli_config):
        """Test batch entailment computation"""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.config.id2label = {0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}
        mock_model.to.return_value = mock_model
        
        # Mock model visual_claims for batch of 3
        mock_output = Mock()
        mock_logits = torch.tensor([
            [2.0, 0.5, -1.0],   # Favors entailment
            [0.5, 2.0, -1.0],   # Favors neutral
            [-1.0, 0.5, 2.0]    # Favors contradiction
        ])
        mock_output.logits = mock_logits
        mock_model.return_value = mock_output
        
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Test
        nli_model = NLIModel(nli_config)
        premises = ["The cat is on the mat.", "A dog is running.", "The sky is green."]
        hypothesis = "There is an animal."
        
        results = nli_model.compute_batch_entailment(premises, hypothesis)
        
        # Verify results
        assert len(results) == 3
        assert results[0]["entailment"] > results[0]["neutral"]
        assert results[1]["neutral"] > results[1]["entailment"]
        assert results[2]["contradiction"] > results[2]["entailment"]
    
    @patch('revisto_evidence_aligned.nlp.nli.AutoModelForSequenceClassification')
    @patch('revisto_evidence_aligned.nlp.nli.AutoTokenizer')
    def test_compute_batch_entailment_large(self, mock_tokenizer_class, mock_model_class, nli_config):
        """Test batch entailment with size larger than batch_size"""
        nli_config.batch_size = 2
        
        # Setup mocks
        mock_tokenizer = Mock()
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.config.id2label = {0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}
        mock_model.to.return_value = mock_model
        
        # Mock different outputs for each batch
        call_count = 0
        def mock_forward(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_output = Mock()
            if call_count == 1:  # First batch (2 items)
                mock_output.logits = torch.tensor([[2.0, 0.5, -1.0], [0.5, 2.0, -1.0]])
            elif call_count == 2:  # Second batch (2 items)
                mock_output.logits = torch.tensor([[-1.0, 0.5, 2.0], [1.0, 1.0, 1.0]])
            else:  # Third batch (1 item)
                mock_output.logits = torch.tensor([[0.0, 0.0, 0.0]])
            return mock_output
        
        mock_model.side_effect = mock_forward
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Test
        nli_model = NLIModel(nli_config)
        premises = ["P1", "P2", "P3", "P4", "P5"]
        hypothesis = "H"
        
        results = nli_model.compute_batch_entailment(premises, hypothesis)
        
        assert len(results) == 5
        assert call_count == 3  # 3 batches
    
    def test_get_support_score(self, nli_config):
        """Test support score calculation"""
        model = NLIModel(nli_config)
        
        # High support (high entailment, low contradiction)
        scores1 = {"entailment": 0.8, "neutral": 0.15, "contradiction": 0.05}
        support1 = model.get_support_score(scores1)
        assert support1 == 0.75
        
        # Low support (low entailment, high contradiction)
        scores2 = {"entailment": 0.1, "neutral": 0.2, "contradiction": 0.7}
        support2 = model.get_support_score(scores2)
        assert support2 == -0.6
        
        # Neutral (similar entailment and contradiction)
        scores3 = {"entailment": 0.4, "neutral": 0.2, "contradiction": 0.4}
        support3 = model.get_support_score(scores3)
        assert support3 == 0.0
        
        # Missing keys
        scores4 = {"neutral": 0.5}
        support4 = model.get_support_score(scores4)
        assert support4 == 0.0


class TestComputeNLIScores:
    """Test compute_nli_scores helper function"""
    
    @patch('revisto_evidence_aligned.nlp.nli.NLIModel')
    def test_compute_nli_scores_with_model(self, mock_nli_class):
        """Test with provided NLI model"""
        mock_model = Mock()
        mock_model.compute_batch_entailment.return_value = [
            {"entailment": 0.8, "neutral": 0.15, "contradiction": 0.05}
        ]
        
        premises = ["The cat is sleeping."]
        hypothesis = "The cat is awake."
        
        scores = compute_nli_scores(premises, hypothesis, nli_model=mock_model)
        
        mock_model.compute_batch_entailment.assert_called_once_with(premises, hypothesis)
        assert len(scores) == 1
        assert scores[0]["entailment"] == 0.8
    
    @patch('revisto_evidence_aligned.nlp.nli.NLIModel')
    def test_compute_nli_scores_with_config(self, mock_nli_class):
        """Test with provided config"""
        mock_model = Mock()
        mock_model.compute_batch_entailment.return_value = [
            {"entailment": 0.6, "neutral": 0.3, "contradiction": 0.1}
        ]
        mock_nli_class.return_value = mock_model
        
        config = NLIConfig(model_name="custom-model")
        premises = ["Test premise"]
        hypothesis = "Test hypothesis"
        
        scores = compute_nli_scores(premises, hypothesis, config=config)
        
        mock_nli_class.assert_called_once_with(config)
        mock_model.compute_batch_entailment.assert_called_once_with(premises, hypothesis)
    
    @patch('revisto_evidence_aligned.nlp.nli.NLIModel')
    def test_compute_nli_scores_default(self, mock_nli_class):
        """Test with default config"""
        mock_model = Mock()
        mock_model.compute_batch_entailment.return_value = []
        mock_nli_class.return_value = mock_model
        
        scores = compute_nli_scores([], "hypothesis")
        
        # Should create model with default config
        mock_nli_class.assert_called_once()
        assert isinstance(mock_nli_class.call_args[0][0], NLIConfig)