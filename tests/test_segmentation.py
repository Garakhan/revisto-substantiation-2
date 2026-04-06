"""Tests for document segmentation"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List

from revisto_evidence_aligned.nlp.segmentation import segment_document
from revisto_evidence_aligned.utils.types import Segment, PageGeometry
from revisto_evidence_aligned.config import SegmentationConfig


class TestSegmentDocument:
    """Test document segmentation functionality"""
    
    @pytest.fixture
    def mock_converter(self):
        """Create mock DocumentConverter"""
        converter = Mock()
        return converter
    
    @pytest.fixture
    def mock_result(self):
        """Create mock conversion result"""
        result = Mock()
        
        # Mock document with export methods
        document = Mock()
        document.export_to_markdown.return_value = "# Title\n\nParagraph 1\n\n## Section 1\n\nParagraph 2"
        document.export_to_dict.return_value = {
            "texts": [
                {
                    "text": "Title",
                    "page": 1,
                    "bbox": {"x0": 0, "y0": 0, "x1": 100, "y1": 20}
                },
                {
                    "text": "Paragraph 1",
                    "page": 1,
                    "bbox": {"x0": 0, "y0": 30, "x1": 200, "y1": 50}
                },
                {
                    "text": "Section 1", 
                    "page": 1,
                    "bbox": {"x0": 0, "y0": 60, "x1": 150, "y1": 80}
                },
                {
                    "text": "Paragraph 2",
                    "page": 2,
                    "bbox": {"x0": 0, "y0": 10, "x1": 200, "y1": 30}
                }
            ],
            "pages": {
                "1": {"page_no": 1, "size": {"width": 612, "height": 792}},
                "2": {"page_no": 2, "size": {"width": 612, "height": 792}}
            }
        }
        
        result.document = document
        return result
    
    @pytest.fixture
    def mock_parser(self):
        """Create mock DoclingParser"""
        parser = Mock()
        parser.parse.return_value = [
            Segment(
                text="Title\nParagraph 1",
                section_title="Title",
                page=1,
                bbox={"x0": 0, "y0": 0, "x1": 200, "y1": 50}
            ),
            Segment(
                text="Section 1\nParagraph 2",
                section_title="Section 1",
                page=2,
                bbox={"x0": 0, "y0": 10, "x1": 200, "y1": 30}
            )
        ]
        return parser
    
    @patch('revisto_evidence_aligned.nlp.segmentation.DoclingParser')
    @patch('revisto_evidence_aligned.nlp.segmentation.DocumentConverter')
    def test_segment_document_basic(self, mock_converter_class, mock_parser_class, mock_converter, mock_result, mock_parser):
        """Test basic document segmentation"""
        mock_converter_class.return_value = mock_converter
        mock_converter.convert.return_value = mock_result
        mock_parser_class.return_value = mock_parser
        
        file_path = "test.pdf"
        segments = segment_document(file_path)
        
        # Verify calls
        mock_converter.convert.assert_called_once_with(file_path)
        mock_result.document.export_to_markdown.assert_called_once()
        mock_result.document.export_to_dict.assert_called_once()
        
        # Verify segments
        assert len(segments) == 2
        assert segments[0].text == "Title\nParagraph 1"
        assert segments[0].section_title == "Title"
        assert segments[0].page == 1
        
        assert segments[1].text == "Section 1\nParagraph 2" 
        assert segments[1].section_title == "Section 1"
        assert segments[1].page == 2
    
    @patch('revisto_evidence_aligned.nlp.segmentation.DoclingParser')
    @patch('revisto_evidence_aligned.nlp.segmentation.DocumentConverter')
    def test_segment_document_with_config(self, mock_converter_class, mock_parser_class):
        """Test segmentation with custom config"""
        config = SegmentationConfig(min_len=50, max_len=1000)
        
        mock_converter_class.return_value = Mock()
        mock_parser_class.return_value = Mock()
        
        # Mock minimal response
        mock_result = Mock()
        mock_doc = Mock()
        mock_doc.export_to_markdown.return_value = "Test"
        mock_doc.export_to_dict.return_value = {"texts": []}
        mock_result.document = mock_doc
        mock_converter_class.return_value.convert.return_value = mock_result
        mock_parser_class.return_value.parse.return_value = []
        
        segments = segment_document("test.pdf", config)
        
        # Config should be passed through
        assert isinstance(segments, list)
    
    @patch('revisto_evidence_aligned.nlp.segmentation.DoclingParser')
    @patch('revisto_evidence_aligned.nlp.segmentation.DocumentConverter')
    def test_segment_document_with_page_geometries(self, mock_converter_class, mock_parser_class, mock_converter, mock_result, mock_parser):
        """Test extraction of page geometries"""
        mock_converter_class.return_value = mock_converter
        mock_converter.convert.return_value = mock_result
        mock_parser_class.return_value = mock_parser
        
        segments = segment_document("test.pdf")
        
        # Parser should have page geometries added
        add_page_geometry_calls = [call for call in mock_parser.method_calls if call[0] == 'add_page_geometry']
        assert len(add_page_geometry_calls) == 2
    
    @patch('revisto_evidence_aligned.nlp.segmentation.DoclingParser')
    @patch('revisto_evidence_aligned.nlp.segmentation.DocumentConverter')
    def test_segment_document_error_handling(self, mock_converter_class, mock_parser_class):
        """Test error handling during segmentation"""
        mock_converter_class.return_value.convert.side_effect = Exception("Conversion failed")
        
        with pytest.raises(Exception) as exc_info:
            segment_document("test.pdf")
        
        assert "Conversion failed" in str(exc_info.value)
    
    @patch('revisto_evidence_aligned.nlp.segmentation.DoclingParser')
    @patch('revisto_evidence_aligned.nlp.segmentation.DocumentConverter')
    def test_segment_document_empty_result(self, mock_converter_class, mock_parser_class):
        """Test handling of empty document"""
        mock_converter = Mock()
        mock_converter_class.return_value = mock_converter
        
        # Mock empty document
        mock_result = Mock()
        mock_doc = Mock()
        mock_doc.export_to_markdown.return_value = ""
        mock_doc.export_to_dict.return_value = {"texts": [], "pages": {}}
        mock_result.document = mock_doc
        mock_converter.convert.return_value = mock_result
        
        # Parser returns empty list
        mock_parser_class.return_value.parse.return_value = []
        
        segments = segment_document("empty.pdf")
        
        assert segments == []
    
    @patch('revisto_evidence_aligned.nlp.segmentation.DoclingParser')
    @patch('revisto_evidence_aligned.nlp.segmentation.DocumentConverter')
    def test_segment_document_no_pages_info(self, mock_converter_class, mock_parser_class):
        """Test handling when no page info is available"""
        mock_converter = Mock()
        mock_converter_class.return_value = mock_converter
        
        # Mock document without pages info
        mock_result = Mock()
        mock_doc = Mock()
        mock_doc.export_to_markdown.return_value = "Some text"
        mock_doc.export_to_dict.return_value = {"texts": [{"text": "Some text"}]}  # No pages key
        mock_result.document = mock_doc
        mock_converter.convert.return_value = mock_result
        
        mock_parser = Mock()
        mock_parser.parse.return_value = [Segment(text="Some text", page=1)]
        mock_parser_class.return_value = mock_parser
        
        segments = segment_document("test.pdf")
        
        # Should still work without page geometries
        assert len(segments) == 1
        assert segments[0].text == "Some text"