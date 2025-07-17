import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

from src.matching.semantic_matcher import SemanticMatcher, VideoTextMatch
from src.processing.video_processor import VideoSegment
from src.processing.text_processor import TextSegment


class TestSemanticMatcher:
    
    @pytest.fixture
    def semantic_matcher(self):
        return SemanticMatcher()
    
    @pytest.fixture
    def sample_video_segments(self):
        """Create sample video segments for testing."""
        segments = []
        for i in range(3):
            frames = np.random.randint(0, 255, (5, 64, 64, 3), dtype=np.uint8)
            segment = VideoSegment(
                start_time=i * 2.0,
                end_time=(i + 1) * 2.0,
                frames=frames,
                frame_rate=30.0,
                duration=2.0
            )
            segments.append(segment)
        return segments
    
    @pytest.fixture
    def sample_text_segments(self):
        """Create sample text segments for testing."""
        texts = [
            "A person walking down the street",
            "A cat sitting on a chair",
            "A beautiful sunset over the ocean"
        ]
        
        segments = []
        for i, text in enumerate(texts):
            embedding = np.random.rand(384).astype(np.float32)  # Common embedding size
            segment = TextSegment(
                text=text,
                start_pos=i * 50,
                end_pos=(i + 1) * 50,
                embedding=embedding,
                tokens=text.lower().split()
            )
            segments.append(segment)
        return segments
    
    def test_semantic_matcher_initialization(self, semantic_matcher):
        """Test SemanticMatcher initialization."""
        assert semantic_matcher.config is not None
        assert semantic_matcher.clip_encoder is not None
        assert semantic_matcher.min_confidence >= 0.0
        assert semantic_matcher.high_confidence > semantic_matcher.min_confidence
    
    @patch('src.matching.semantic_matcher.SemanticMatcher._get_video_embeddings')
    @patch('src.matching.semantic_matcher.SemanticMatcher._get_text_embeddings')
    def test_match_video_to_text(self, mock_text_embeddings, mock_video_embeddings, 
                                semantic_matcher, sample_video_segments, sample_text_segments):
        """Test video-to-text matching."""
        # Mock embeddings
        mock_video_embeddings.return_value = torch.randn(3, 512)
        mock_text_embeddings.return_value = torch.randn(3, 512)
        
        matches = semantic_matcher.match_video_to_text(
            sample_video_segments, 
            sample_text_segments, 
            matching_strategy="optimal"
        )
        
        assert isinstance(matches, list)
        assert all(isinstance(match, VideoTextMatch) for match in matches)
        
        # Check match properties
        for match in matches:
            assert match.video_segment_idx >= 0
            assert match.text_segment_idx >= 0
            assert match.video_start >= 0
            assert match.video_end > match.video_start
            assert match.text_start >= 0
            assert match.text_end > match.text_start
            assert 0.0 <= match.confidence <= 1.0
    
    def test_match_empty_inputs(self, semantic_matcher):
        """Test matching with empty inputs."""
        matches = semantic_matcher.match_video_to_text([], [])
        assert len(matches) == 0
        
        matches = semantic_matcher.match_video_to_text([], [MagicMock()])
        assert len(matches) == 0
        
        matches = semantic_matcher.match_video_to_text([MagicMock()], [])
        assert len(matches) == 0
    
    def test_compute_similarity_matrix(self, semantic_matcher):
        """Test similarity matrix computation."""
        video_embeddings = torch.randn(2, 512)
        text_embeddings = torch.randn(3, 512)
        
        similarity_matrix = semantic_matcher._compute_similarity_matrix(
            video_embeddings, text_embeddings
        )
        
        assert similarity_matrix.shape == (2, 3)
        assert isinstance(similarity_matrix, np.ndarray)
        assert -1.0 <= similarity_matrix.min() <= 1.0
        assert -1.0 <= similarity_matrix.max() <= 1.0
    
    def test_optimal_matching_strategy(self, semantic_matcher, sample_video_segments, sample_text_segments):
        """Test optimal matching strategy."""
        # Create a similarity matrix where optimal matching is clear
        similarity_matrix = np.array([
            [0.9, 0.1, 0.2],  # Video 0 best matches Text 0
            [0.3, 0.8, 0.1],  # Video 1 best matches Text 1
            [0.2, 0.1, 0.7]   # Video 2 best matches Text 2
        ])
        
        matches = semantic_matcher._optimal_matching(
            similarity_matrix, sample_video_segments, sample_text_segments
        )
        
        # Should find optimal one-to-one matching
        assert len(matches) == 3
        
        # Check that each video and text segment is matched only once
        video_indices = [match.video_segment_idx for match in matches]
        text_indices = [match.text_segment_idx for match in matches]
        
        assert len(set(video_indices)) == len(video_indices)  # No duplicates
        assert len(set(text_indices)) == len(text_indices)    # No duplicates
    
    def test_greedy_matching_strategy(self, semantic_matcher, sample_video_segments, sample_text_segments):
        """Test greedy matching strategy."""
        similarity_matrix = np.array([
            [0.9, 0.1, 0.2],
            [0.3, 0.8, 0.1],
            [0.2, 0.1, 0.7]
        ])
        
        matches = semantic_matcher._greedy_matching(
            similarity_matrix, sample_video_segments, sample_text_segments
        )
        
        assert len(matches) > 0
        assert all(match.confidence >= semantic_matcher.min_confidence for match in matches)
    
    def test_threshold_matching_strategy(self, semantic_matcher, sample_video_segments, sample_text_segments):
        """Test threshold matching strategy."""
        similarity_matrix = np.array([
            [0.9, 0.1, 0.2],
            [0.3, 0.8, 0.1],
            [0.2, 0.1, 0.7]
        ])
        
        matches = semantic_matcher._threshold_matching(
            similarity_matrix, sample_video_segments, sample_text_segments
        )
        
        # Should include all matches above threshold
        expected_matches = np.sum(similarity_matrix >= semantic_matcher.min_confidence)
        assert len(matches) == expected_matches
    
    def test_classify_match_strength(self, semantic_matcher):
        """Test match strength classification."""
        high_confidence = semantic_matcher.high_confidence + 0.1
        medium_confidence = 0.5
        low_confidence = semantic_matcher.min_confidence + 0.1
        
        assert semantic_matcher._classify_match_strength(high_confidence) == "high_confidence"
        assert semantic_matcher._classify_match_strength(medium_confidence) == "medium_confidence"
        assert semantic_matcher._classify_match_strength(low_confidence) == "low_confidence"
    
    def test_compute_matching_metrics(self, semantic_matcher):
        """Test matching metrics computation."""
        # Create sample matches
        matches = [
            VideoTextMatch(0, 0, 0.0, 2.0, 0, 50, 0.9, {}),
            VideoTextMatch(1, 1, 2.0, 4.0, 50, 100, 0.7, {}),
            VideoTextMatch(2, 2, 4.0, 6.0, 100, 150, 0.5, {})
        ]
        
        metrics = semantic_matcher.compute_matching_metrics(matches)
        
        assert "avg_confidence" in metrics
        assert "high_confidence_ratio" in metrics
        assert "total_matches" in metrics
        assert "min_confidence" in metrics
        assert "max_confidence" in metrics
        assert "std_confidence" in metrics
        
        assert metrics["total_matches"] == 3
        assert metrics["avg_confidence"] == 0.7  # (0.9 + 0.7 + 0.5) / 3
        assert metrics["min_confidence"] == 0.5
        assert metrics["max_confidence"] == 0.9
    
    def test_compute_matching_metrics_empty(self, semantic_matcher):
        """Test matching metrics with empty matches."""
        metrics = semantic_matcher.compute_matching_metrics([])
        
        assert metrics["avg_confidence"] == 0.0
        assert metrics["high_confidence_ratio"] == 0.0
        assert metrics["total_matches"] == 0
        assert metrics["coverage_ratio"] == 0.0
    
    def test_filter_matches_by_confidence(self, semantic_matcher):
        """Test filtering matches by confidence."""
        matches = [
            VideoTextMatch(0, 0, 0.0, 2.0, 0, 50, 0.9, {}),
            VideoTextMatch(1, 1, 2.0, 4.0, 50, 100, 0.7, {}),
            VideoTextMatch(2, 2, 4.0, 6.0, 100, 150, 0.2, {})
        ]
        
        filtered = semantic_matcher.filter_matches_by_confidence(matches, min_confidence=0.5)
        
        assert len(filtered) == 2
        assert all(match.confidence >= 0.5 for match in filtered)
    
    def test_get_best_matches(self, semantic_matcher):
        """Test getting best matches."""
        matches = [
            VideoTextMatch(0, 0, 0.0, 2.0, 0, 50, 0.5, {}),
            VideoTextMatch(1, 1, 2.0, 4.0, 50, 100, 0.9, {}),
            VideoTextMatch(2, 2, 4.0, 6.0, 100, 150, 0.7, {}),
            VideoTextMatch(3, 3, 6.0, 8.0, 150, 200, 0.3, {})
        ]
        
        best_matches = semantic_matcher.get_best_matches(matches, top_k=2)
        
        assert len(best_matches) == 2
        assert best_matches[0].confidence == 0.9  # Highest confidence first
        assert best_matches[1].confidence == 0.7  # Second highest
    
    def test_unknown_matching_strategy(self, semantic_matcher, sample_video_segments, sample_text_segments):
        """Test unknown matching strategy raises error."""
        with pytest.raises(ValueError):
            semantic_matcher.match_video_to_text(
                sample_video_segments, 
                sample_text_segments, 
                matching_strategy="unknown_strategy"
            )
    
    def test_add_explanations(self, semantic_matcher, sample_video_segments, sample_text_segments):
        """Test adding explanations to matches."""
        matches = [
            VideoTextMatch(0, 0, 0.0, 2.0, 0, 50, 0.8, {})
        ]
        
        matches_with_explanations = semantic_matcher._add_explanations(
            matches, sample_video_segments, sample_text_segments
        )
        
        assert len(matches_with_explanations) == 1
        
        explanation = matches_with_explanations[0].explanation
        assert "match_type" in explanation
        assert "video_duration" in explanation
        assert "text_length" in explanation
        assert "text_preview" in explanation
        assert "keywords" in explanation
        assert "temporal_alignment" in explanation
    
    def test_matching_with_low_confidence_threshold(self, semantic_matcher):
        """Test matching with very low confidence threshold."""
        original_min_confidence = semantic_matcher.min_confidence
        semantic_matcher.min_confidence = 0.1
        
        try:
            similarity_matrix = np.array([
                [0.2, 0.15, 0.12],
                [0.18, 0.25, 0.11],
                [0.16, 0.13, 0.22]
            ])
            
            video_segments = [MagicMock() for _ in range(3)]
            text_segments = [MagicMock() for _ in range(3)]
            
            matches = semantic_matcher._optimal_matching(
                similarity_matrix, video_segments, text_segments
            )
            
            assert len(matches) == 3  # All matches should be above 0.1 threshold
            
        finally:
            semantic_matcher.min_confidence = original_min_confidence
    
    def test_videotext_match_properties(self):
        """Test VideoTextMatch dataclass properties."""
        match = VideoTextMatch(
            video_segment_idx=1,
            text_segment_idx=2,
            video_start=1.0,
            video_end=3.0,
            text_start=25,
            text_end=75,
            confidence=0.85,
            explanation={"test": "explanation"}
        )
        
        assert match.video_segment_idx == 1
        assert match.text_segment_idx == 2
        assert match.video_start == 1.0
        assert match.video_end == 3.0
        assert match.text_start == 25
        assert match.text_end == 75
        assert match.confidence == 0.85
        assert match.explanation == {"test": "explanation"}