"""Tests for tokenization utilities"""

import pytest
from revisto_evidence_aligned.nlp.tokenization import (
    split_sentences,
    tokenize,
    extract_numbers,
    extract_stopwords,
    remove_stopwords
)


class TestSplitSentences:
    """Test sentence splitting functionality"""
    
    def test_split_sentences_basic(self):
        text = "This is the first sentence. This is the second sentence."
        result = split_sentences(text)
        assert len(result) == 2
        assert result[0] == "This is the first sentence."
        assert result[1] == "This is the second sentence."
    
    def test_split_sentences_with_abbreviations(self):
        text = "Dr. Smith went to the U.S.A. yesterday. He returned today."
        result = split_sentences(text)
        assert len(result) == 2
        assert "Dr. Smith" in result[0]
        assert "He returned today." in result[1]
    
    def test_split_sentences_empty_text(self):
        assert split_sentences("") == []
        assert split_sentences("   ") == []
        assert split_sentences(None) == []
    
    def test_split_sentences_single_sentence(self):
        text = "This is a single sentence"
        result = split_sentences(text)
        assert len(result) == 1
        assert result[0] == text
    
    def test_split_sentences_with_newlines(self):
        text = "First sentence.\n\nSecond sentence.\nThird sentence."
        result = split_sentences(text)
        assert len(result) == 3
    
    def test_split_sentences_filters_single_chars(self):
        text = "Valid sentence. . Another sentence."
        result = split_sentences(text)
        assert len(result) == 2
        assert "." not in [r.strip() for r in result]


class TestTokenize:
    """Test word tokenization functionality"""
    
    def test_tokenize_basic(self):
        text = "This is a simple test"
        tokens = tokenize(text)
        assert tokens == ["this", "is", "a", "simple", "test"]
    
    def test_tokenize_without_lowercase(self):
        text = "This IS a Simple TEST"
        tokens = tokenize(text, lowercase=False)
        assert tokens == ["This", "IS", "a", "Simple", "TEST"]
    
    def test_tokenize_with_punctuation(self):
        text = "Hello, world! How are you?"
        tokens = tokenize(text)
        assert tokens == ["hello", "world", "how", "are", "you"]
    
    def test_tokenize_with_numbers(self):
        text = "I have 123 apples and 456 oranges"
        tokens = tokenize(text)
        assert "123" in tokens
        assert "456" in tokens
    
    def test_tokenize_empty_text(self):
        assert tokenize("") == []
        assert tokenize("   ") == []


class TestExtractNumbers:
    """Test number extraction functionality"""
    
    def test_extract_simple_numbers(self):
        text = "I have 123 apples and 456 oranges"
        numbers = extract_numbers(text)
        assert "123" in numbers
        assert "456" in numbers
    
    def test_extract_percentages(self):
        text = "The accuracy is 95.5% and precision is 87%"
        numbers = extract_numbers(text)
        assert "95.5%" in numbers
        assert "87%" in numbers
    
    def test_extract_numbers_with_commas(self):
        text = "The population is 1,234,567 people"
        numbers = extract_numbers(text)
        assert "1,234,567" in numbers
    
    def test_extract_decimals(self):
        text = "The value is 3.14159 and another is 2.5"
        numbers = extract_numbers(text)
        assert "3.14159" in numbers
        assert "2.5" in numbers
    
    def test_extract_p_values(self):
        text = "The result is significant with p < 0.05"
        numbers = extract_numbers(text)
        assert any("p" in num and "0.05" in num for num in numbers)
    
    def test_extract_numbers_with_units(self):
        text = "Take 500mg twice daily. The distance is 10km."
        numbers = extract_numbers(text)
        assert any("500" in num and "mg" in num for num in numbers)
        assert any("10" in num and "km" in num for num in numbers)
    
    def test_extract_numbers_no_duplicates(self):
        text = "The value 100 appears here and 100 appears again"
        numbers = extract_numbers(text)
        assert numbers.count("100") == 1
    
    def test_extract_numbers_empty_text(self):
        assert extract_numbers("") == []
        assert extract_numbers("No numbers here") == []


class TestStopwords:
    """Test stopword functionality"""
    
    def test_extract_stopwords(self):
        stopwords = extract_stopwords()
        assert isinstance(stopwords, set)
        assert len(stopwords) > 0
        assert "the" in stopwords
        assert "is" in stopwords
        assert "and" in stopwords
        assert "a" in stopwords
    
    def test_remove_stopwords_basic(self):
        tokens = ["this", "is", "a", "test", "sentence"]
        filtered = remove_stopwords(tokens)
        assert "test" in filtered
        assert "sentence" in filtered
        assert "is" not in filtered
        assert "a" not in filtered
    
    def test_remove_stopwords_custom(self):
        tokens = ["this", "is", "custom", "test"]
        custom_stopwords = {"custom", "test"}
        filtered = remove_stopwords(tokens, stopwords=custom_stopwords)
        assert filtered == ["this", "is"]
    
    def test_remove_stopwords_empty_list(self):
        assert remove_stopwords([]) == []
    
    def test_remove_stopwords_case_insensitive(self):
        tokens = ["This", "IS", "A", "Test"]
        filtered = remove_stopwords(tokens)
        assert "Test" in filtered
        assert "IS" not in filtered
        assert "A" not in filtered