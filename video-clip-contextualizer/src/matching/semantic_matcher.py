import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from scipy.optimize import linear_sum_assignment

from ..models.clip_encoder import CLIPVideoEncoder
from ..processing.video_processor import VideoSegment
from ..processing.text_processor import TextSegment
from ..config import get_config


@dataclass
class MatchResult:
    video_segment: VideoSegment
    text_segment: TextSegment
    confidence: float
    similarity_score: float
    explanation: Dict[str, any]


@dataclass
class VideoTextMatch:
    video_segment_idx: int
    text_segment_idx: int
    video_start: float
    video_end: float
    text_start: int
    text_end: int
    confidence: float
    explanation: Dict[str, any]


class SemanticMatcher:
    """Core semantic matching engine using CLIP embeddings."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize CLIP encoder
        self.clip_encoder = CLIPVideoEncoder()
        
        # Confidence thresholds
        self.min_confidence = 0.3
        self.high_confidence = 0.7
        
    def match_video_to_text(
        self, 
        video_segments: List[VideoSegment], 
        text_segments: List[TextSegment],
        matching_strategy: str = "optimal"
    ) -> List[VideoTextMatch]:
        """
        Match video segments to text segments using semantic similarity.
        
        Args:
            video_segments: List of video segments
            text_segments: List of text segments
            matching_strategy: Strategy for matching ("optimal", "greedy", "threshold")
            
        Returns:
            List of matches with confidence scores
        """
        if not video_segments or not text_segments:
            return []
        
        # Generate embeddings for video segments
        video_embeddings = self._get_video_embeddings(video_segments)
        
        # Generate embeddings for text segments
        text_embeddings = self._get_text_embeddings(text_segments)
        
        # Compute similarity matrix
        similarity_matrix = self._compute_similarity_matrix(video_embeddings, text_embeddings)
        
        # Apply matching strategy
        matches = self._apply_matching_strategy(
            similarity_matrix, 
            video_segments, 
            text_segments, 
            matching_strategy
        )
        
        # Generate explanations
        matches = self._add_explanations(matches, video_segments, text_segments)
        
        return matches
    
    def _get_video_embeddings(self, video_segments: List[VideoSegment]) -> torch.Tensor:
        """Generate embeddings for video segments."""
        embeddings = []
        
        for segment in video_segments:
            # Preprocess frames
            frames = self.clip_encoder.model.processor.image_processor.preprocess(
                segment.frames, return_tensors="pt"
            )
            
            # Get embedding
            embedding = self.clip_encoder.get_video_embedding(segment.frames)
            embeddings.append(embedding)
        
        return torch.stack(embeddings)
    
    def _get_text_embeddings(self, text_segments: List[TextSegment]) -> torch.Tensor:
        """Generate embeddings for text segments."""
        texts = [segment.text for segment in text_segments]
        return self.clip_encoder.encode_text(texts)
    
    def _compute_similarity_matrix(self, video_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> np.ndarray:
        """Compute similarity matrix between video and text embeddings."""
        similarity = self.clip_encoder.compute_similarity(video_embeddings, text_embeddings)
        return similarity.cpu().numpy()
    
    def _apply_matching_strategy(
        self, 
        similarity_matrix: np.ndarray, 
        video_segments: List[VideoSegment], 
        text_segments: List[TextSegment], 
        strategy: str
    ) -> List[VideoTextMatch]:
        """Apply matching strategy to find best matches."""
        
        if strategy == "optimal":
            return self._optimal_matching(similarity_matrix, video_segments, text_segments)
        elif strategy == "greedy":
            return self._greedy_matching(similarity_matrix, video_segments, text_segments)
        elif strategy == "threshold":
            return self._threshold_matching(similarity_matrix, video_segments, text_segments)
        else:
            raise ValueError(f"Unknown matching strategy: {strategy}")
    
    def _optimal_matching(
        self, 
        similarity_matrix: np.ndarray, 
        video_segments: List[VideoSegment], 
        text_segments: List[TextSegment]
    ) -> List[VideoTextMatch]:
        """Use Hungarian algorithm for optimal matching."""
        
        # Convert similarity to cost (negative similarity)
        cost_matrix = -similarity_matrix
        
        # Apply Hungarian algorithm
        video_indices, text_indices = linear_sum_assignment(cost_matrix)
        
        matches = []
        for v_idx, t_idx in zip(video_indices, text_indices):
            confidence = similarity_matrix[v_idx, t_idx]
            
            # Only include matches above minimum confidence
            if confidence >= self.min_confidence:
                match = VideoTextMatch(
                    video_segment_idx=v_idx,
                    text_segment_idx=t_idx,
                    video_start=video_segments[v_idx].start_time,
                    video_end=video_segments[v_idx].end_time,
                    text_start=text_segments[t_idx].start_pos,
                    text_end=text_segments[t_idx].end_pos,
                    confidence=float(confidence),
                    explanation={}
                )
                matches.append(match)
        
        return matches
    
    def _greedy_matching(
        self, 
        similarity_matrix: np.ndarray, 
        video_segments: List[VideoSegment], 
        text_segments: List[TextSegment]
    ) -> List[VideoTextMatch]:
        """Greedy matching: select highest similarity matches."""
        
        matches = []
        used_video = set()
        used_text = set()
        
        # Get sorted indices by similarity
        indices = np.unravel_index(
            np.argsort(similarity_matrix.ravel())[::-1], 
            similarity_matrix.shape
        )
        
        for v_idx, t_idx in zip(indices[0], indices[1]):
            if v_idx in used_video or t_idx in used_text:
                continue
                
            confidence = similarity_matrix[v_idx, t_idx]
            
            if confidence >= self.min_confidence:
                match = VideoTextMatch(
                    video_segment_idx=v_idx,
                    text_segment_idx=t_idx,
                    video_start=video_segments[v_idx].start_time,
                    video_end=video_segments[v_idx].end_time,
                    text_start=text_segments[t_idx].start_pos,
                    text_end=text_segments[t_idx].end_pos,
                    confidence=float(confidence),
                    explanation={}
                )
                matches.append(match)
                used_video.add(v_idx)
                used_text.add(t_idx)
        
        return matches
    
    def _threshold_matching(
        self, 
        similarity_matrix: np.ndarray, 
        video_segments: List[VideoSegment], 
        text_segments: List[TextSegment]
    ) -> List[VideoTextMatch]:
        """Match all pairs above confidence threshold."""
        
        matches = []
        
        for v_idx in range(len(video_segments)):
            for t_idx in range(len(text_segments)):
                confidence = similarity_matrix[v_idx, t_idx]
                
                if confidence >= self.min_confidence:
                    match = VideoTextMatch(
                        video_segment_idx=v_idx,
                        text_segment_idx=t_idx,
                        video_start=video_segments[v_idx].start_time,
                        video_end=video_segments[v_idx].end_time,
                        text_start=text_segments[t_idx].start_pos,
                        text_end=text_segments[t_idx].end_pos,
                        confidence=float(confidence),
                        explanation={}
                    )
                    matches.append(match)
        
        return matches
    
    def _add_explanations(
        self, 
        matches: List[VideoTextMatch], 
        video_segments: List[VideoSegment], 
        text_segments: List[TextSegment]
    ) -> List[VideoTextMatch]:
        """Add explanations to matches."""
        
        for match in matches:
            video_segment = video_segments[match.video_segment_idx]
            text_segment = text_segments[match.text_segment_idx]
            
            # Generate explanation
            explanation = {
                "match_type": self._classify_match_strength(match.confidence),
                "video_duration": video_segment.duration,
                "text_length": len(text_segment.text),
                "text_preview": text_segment.text[:100] + "..." if len(text_segment.text) > 100 else text_segment.text,
                "keywords": text_segment.tokens[:5] if text_segment.tokens else [],
                "temporal_alignment": {
                    "video_position": match.video_start / video_segments[-1].end_time if video_segments else 0,
                    "text_position": match.text_start / text_segments[-1].end_pos if text_segments else 0
                }
            }
            
            match.explanation = explanation
        
        return matches
    
    def _classify_match_strength(self, confidence: float) -> str:
        """Classify match strength based on confidence."""
        if confidence >= self.high_confidence:
            return "high_confidence"
        elif confidence >= 0.5:
            return "medium_confidence"
        else:
            return "low_confidence"
    
    def compute_matching_metrics(self, matches: List[VideoTextMatch]) -> Dict[str, float]:
        """Compute metrics for matching quality."""
        if not matches:
            return {
                "avg_confidence": 0.0,
                "high_confidence_ratio": 0.0,
                "total_matches": 0,
                "coverage_ratio": 0.0
            }
        
        confidences = [match.confidence for match in matches]
        
        return {
            "avg_confidence": np.mean(confidences),
            "high_confidence_ratio": sum(1 for c in confidences if c >= self.high_confidence) / len(confidences),
            "total_matches": len(matches),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
            "std_confidence": np.std(confidences)
        }
    
    def filter_matches_by_confidence(self, matches: List[VideoTextMatch], min_confidence: float = None) -> List[VideoTextMatch]:
        """Filter matches by confidence threshold."""
        threshold = min_confidence or self.min_confidence
        return [match for match in matches if match.confidence >= threshold]
    
    def get_best_matches(self, matches: List[VideoTextMatch], top_k: int = 10) -> List[VideoTextMatch]:
        """Get top-k matches by confidence."""
        sorted_matches = sorted(matches, key=lambda x: x.confidence, reverse=True)
        return sorted_matches[:top_k]