import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.processing.text_processor import TextProcessor, TextSegment


class TestTextProcessor:
    
    @pytest.fixture
    def text_processor(self):
        return TextProcessor()
    
    @pytest.fixture
    def sample_text(self):
        return """
        This is a sample text for testing. It contains multiple sentences with different content.
        The text processor should be able to segment this text into meaningful chunks.
        Each segment should contain semantic information that can be matched with video content.
        """
    
    def test_text_processor_initialization(self, text_processor):
        """Test TextProcessor initialization."""
        assert text_processor.config is not None
        assert text_processor.text_encoder is not None
        assert hasattr(text_processor, 'logger')
    
    def test_segment_text(self, text_processor, sample_text):
        """Test text segmentation."""
        segments = text_processor.segment_text(sample_text)
        
        assert len(segments) > 0
        assert all(isinstance(segment, TextSegment) for segment in segments)
        
        # Check segment properties
        for segment in segments:
            assert segment.text
            assert segment.start_pos >= 0
            assert segment.end_pos > segment.start_pos
            assert segment.tokens is not None
            assert len(segment.tokens) > 0
    
    def test_segment_text_with_max_length(self, text_processor):
        """Test text segmentation with max length constraint."""
        long_text = "This is a very long sentence. " * 20  # Create long text
        
        segments = text_processor.segment_text(long_text, max_segment_length=50)
        
        assert len(segments) > 1
        
        # Check that segments respect max length (allowing some tolerance)
        for segment in segments:
            assert len(segment.text) <= 60  # Small tolerance for sentence boundaries
    
    def test_segment_empty_text(self, text_processor):
        """Test segmentation of empty text."""
        segments = text_processor.segment_text("")
        
        assert len(segments) == 0
    
    def test_segment_single_sentence(self, text_processor):
        """Test segmentation of single sentence."""
        text = "This is a single sentence."
        segments = text_processor.segment_text(text)
        
        assert len(segments) == 1
        assert segments[0].text.strip() == text
    
    def test_generate_embeddings(self, text_processor):
        """Test embedding generation for text segments."""
        text = "This is a test sentence for embedding generation."
        segments = text_processor.segment_text(text)
        
        segments_with_embeddings = text_processor.generate_embeddings(segments)
        
        assert len(segments_with_embeddings) == len(segments)
        
        for segment in segments_with_embeddings:
            assert segment.embedding is not None
            assert isinstance(segment.embedding, np.ndarray)
            assert len(segment.embedding.shape) == 1  # 1D embedding vector
            assert segment.embedding.shape[0] > 0
    
    def test_generate_embeddings_empty_list(self, text_processor):
        """Test embedding generation with empty segment list."""
        segments = text_processor.generate_embeddings([])
        
        assert len(segments) == 0
    
    def test_clean_text(self, text_processor):
        """Test text cleaning functionality."""
        dirty_text = "  This   has    extra   spaces!!!   And @#$% symbols  "
        cleaned = text_processor._clean_text(dirty_text)
        
        assert "  " not in cleaned  # No double spaces
        assert cleaned.strip() == cleaned  # No leading/trailing spaces
        assert "@#$%" not in cleaned  # Special characters removed
    
    def test_split_into_sentences(self, text_processor):
        """Test sentence splitting."""
        text = "First sentence. Second sentence! Third sentence? Fourth sentence."
        sentences = text_processor._split_into_sentences(text)
        
        assert len(sentences) == 4
        assert "First sentence" in sentences[0]
        assert "Second sentence" in sentences[1]
        assert "Third sentence" in sentences[2]
        assert "Fourth sentence" in sentences[3]
    
    def test_tokenize(self, text_processor):
        """Test text tokenization."""
        text = "Hello world! This is a test."
        tokens = text_processor._tokenize(text)
        
        expected_tokens = ["hello", "world", "this", "is", "a", "test"]
        assert tokens == expected_tokens
    
    def test_tokenize_empty_text(self, text_processor):
        """Test tokenization of empty text."""
        tokens = text_processor._tokenize("")
        
        assert len(tokens) == 0
    
    def test_extract_keywords(self, text_processor):
        """Test keyword extraction."""
        text = "The quick brown fox jumps over the lazy dog. The fox is very quick."
        keywords = text_processor.extract_keywords(text, top_k=5)
        
        assert len(keywords) <= 5
        assert "fox" in keywords  # Should appear due to frequency
        assert "quick" in keywords  # Should appear due to frequency
        assert "the" not in keywords  # Should be filtered as stop word
    
    def test_extract_keywords_empty_text(self, text_processor):
        """Test keyword extraction from empty text."""
        keywords = text_processor.extract_keywords("", top_k=5)
        
        assert len(keywords) == 0
    
    def test_compute_similarity(self, text_processor):
        """Test semantic similarity computation."""
        text1 = "The cat sits on the mat."
        text2 = "A feline rests on the carpet."
        text3 = "The weather is sunny today."
        
        similarity12 = text_processor.compute_similarity(text1, text2)
        similarity13 = text_processor.compute_similarity(text1, text3)
        
        assert 0.0 <= similarity12 <= 1.0
        assert 0.0 <= similarity13 <= 1.0
        
        # Similar texts should have higher similarity
        assert similarity12 > similarity13
    
    def test_compute_similarity_identical_texts(self, text_processor):
        """Test similarity of identical texts."""
        text = "This is a test sentence."
        similarity = text_processor.compute_similarity(text, text)
        
        assert similarity > 0.99  # Should be very close to 1.0
    
    def test_get_text_statistics(self, text_processor):
        """Test text statistics computation."""
        text = "This is a test. It has multiple sentences. Each sentence has words."
        stats = text_processor.get_text_statistics(text)
        
        assert "character_count" in stats
        assert "word_count" in stats
        assert "sentence_count" in stats
        assert "avg_sentence_length" in stats
        assert "unique_words" in stats
        assert "vocabulary_richness" in stats
        
        assert stats["character_count"] > 0
        assert stats["word_count"] > 0
        assert stats["sentence_count"] == 3
        assert stats["avg_sentence_length"] > 0
        assert stats["unique_words"] > 0
        assert 0.0 <= stats["vocabulary_richness"] <= 1.0
    
    def test_get_text_statistics_empty_text(self, text_processor):
        """Test text statistics for empty text."""
        stats = text_processor.get_text_statistics("")
        
        assert stats["character_count"] == 0
        assert stats["word_count"] == 0
        assert stats["sentence_count"] == 0
        assert stats["avg_sentence_length"] == 0
        assert stats["unique_words"] == 0
        assert stats["vocabulary_richness"] == 0
    
    def test_text_segment_properties(self):
        """Test TextSegment dataclass properties."""
        text = "This is a test segment."
        tokens = ["this", "is", "a", "test", "segment"]
        embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        segment = TextSegment(
            text=text,
            start_pos=0,
            end_pos=len(text),
            embedding=embedding,
            tokens=tokens
        )
        
        assert segment.text == text
        assert segment.start_pos == 0
        assert segment.end_pos == len(text)
        assert np.array_equal(segment.embedding, embedding)
        assert segment.tokens == tokens
    
    @patch('src.processing.text_processor.SentenceTransformer')
    def test_text_processor_model_loading_error(self, mock_sentence_transformer):
        """Test error handling during model loading."""
        mock_sentence_transformer.side_effect = Exception("Model loading error")
        
        with pytest.raises(Exception):
            TextProcessor()
    
    def test_segment_text_preserves_order(self, text_processor):
        """Test that text segmentation preserves order."""
        text = "First sentence. Second sentence. Third sentence."
        segments = text_processor.segment_text(text)
        
        # Check that segments are in order
        for i in range(len(segments) - 1):
            assert segments[i].end_pos <= segments[i + 1].start_pos
    
    def test_segment_text_with_special_characters(self, text_processor):
        """Test text segmentation with special characters."""
        text = "Hello! How are you? I'm fine, thanks. What about you?"
        segments = text_processor.segment_text(text)
        
        assert len(segments) > 0
        
        # Check that all segments have content
        for segment in segments:
            assert len(segment.text.strip()) > 0
    
    def test_long_text_segmentation(self, text_processor):
        """Test segmentation of very long text."""
        # Create a long text that will definitely need segmentation
        long_text = "This is a sentence. " * 50  # 50 sentences
        
        segments = text_processor.segment_text(long_text, max_segment_length=100)
        
        assert len(segments) > 1
        
        # Check that segments don't exceed max length significantly
        for segment in segments:
            assert len(segment.text) <= 120  # Small tolerance for sentence boundaries
    
    def test_text_with_numbers_and_punctuation(self, text_processor):
        """Test text processing with numbers and punctuation."""
        text = "The year 2024 is here! We have $100 and 50% progress."
        segments = text_processor.segment_text(text)
        
        assert len(segments) > 0
        
        # Check that numbers are preserved in tokens
        tokens = text_processor._tokenize(text)
        assert "2024" in tokens
        assert "100" in tokens
        assert "50" in tokens