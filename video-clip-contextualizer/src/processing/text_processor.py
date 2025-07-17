import re
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from dataclasses import dataclass

from ..config import get_config


@dataclass
class TextSegment:
    text: str
    start_pos: int
    end_pos: int
    embedding: np.ndarray = None
    tokens: List[str] = None


class TextProcessor:
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize sentence transformer model - force CPU usage
        self.text_encoder = SentenceTransformer(
            self.config.models.text_encoder,
            device="cpu"
        )
        
    def segment_text(self, text: str, max_segment_length: int = 200) -> List[TextSegment]:
        """
        Segment text into meaningful chunks for matching.
        
        Args:
            text: Input text script
            max_segment_length: Maximum characters per segment
            
        Returns:
            List of TextSegment objects
        """
        # Clean and normalize text
        cleaned_text = self._clean_text(text)
        
        # Split into sentences
        sentences = self._split_into_sentences(cleaned_text)
        
        # Group sentences into segments
        segments = []
        current_segment = ""
        current_start = 0
        
        for sentence in sentences:
            # If adding this sentence would exceed max length, create a segment
            if len(current_segment) + len(sentence) > max_segment_length and current_segment:
                segment = TextSegment(
                    text=current_segment.strip(),
                    start_pos=current_start,
                    end_pos=current_start + len(current_segment),
                    tokens=self._tokenize(current_segment)
                )
                segments.append(segment)
                
                # Start new segment
                current_start = current_start + len(current_segment)
                current_segment = sentence
            else:
                current_segment += " " + sentence if current_segment else sentence
        
        # Add final segment
        if current_segment:
            segment = TextSegment(
                text=current_segment.strip(),
                start_pos=current_start,
                end_pos=current_start + len(current_segment),
                tokens=self._tokenize(current_segment)
            )
            segments.append(segment)
        
        return segments
    
    def generate_embeddings(self, segments: List[TextSegment]) -> List[TextSegment]:
        """
        Generate embeddings for text segments.
        
        Args:
            segments: List of text segments
            
        Returns:
            Segments with embeddings added
        """
        if not segments:
            return segments
        
        # Extract text from segments
        texts = [segment.text for segment in segments]
        
        # Generate embeddings
        embeddings = self.text_encoder.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Add embeddings to segments
        for segment, embedding in zip(segments, embeddings):
            segment.embedding = embedding
        
        return segments
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        
        # Normalize case
        text = text.strip()
        
        return text
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting on punctuation
        sentences = re.split(r'[.!?]+', text)
        
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Simple word tokenization
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """
        Extract key terms from text for matching explanations.
        
        Args:
            text: Input text
            top_k: Number of top keywords to return
            
        Returns:
            List of keywords
        """
        # Simple frequency-based keyword extraction
        tokens = self._tokenize(text)
        
        # Filter out common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'can', 'must', 'shall', 'this', 'that', 'these', 'those'
        }
        
        filtered_tokens = [token for token in tokens if token not in stop_words]
        
        # Count frequency
        word_freq = {}
        for token in filtered_tokens:
            word_freq[token] = word_freq.get(token, 0) + 1
        
        # Sort by frequency and return top k
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, freq in sorted_words[:top_k]]
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        embeddings = self.text_encoder.encode([text1, text2], convert_to_numpy=True)
        
        # Compute cosine similarity
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        
        return float(similarity)
    
    def get_text_statistics(self, text: str) -> Dict[str, Any]:
        """Get statistics about the text."""
        tokens = self._tokenize(text)
        sentences = self._split_into_sentences(text)
        
        return {
            "character_count": len(text),
            "word_count": len(tokens),
            "sentence_count": len(sentences),
            "avg_sentence_length": np.mean([len(s.split()) for s in sentences]) if sentences else 0,
            "unique_words": len(set(tokens)),
            "vocabulary_richness": len(set(tokens)) / len(tokens) if tokens else 0
        }