import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import logging

from ..config import get_config


class CLIPVideoEncoder:
    """CLIP-based video encoder for semantic video understanding."""
    
    def __init__(self, model_name: str = None):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize CLIP model - force CPU usage
        model_name = model_name or self.config.models.video_encoder
        self.device = torch.device("cpu")
        
        try:
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Skip half precision for CPU compatibility
                
            self.logger.info(f"Loaded CLIP model: {model_name} on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load CLIP model: {str(e)}")
            raise
    
    def encode_frames(self, frames: np.ndarray, batch_size: int = 8) -> torch.Tensor:
        """
        Encode video frames into embeddings.
        
        Args:
            frames: Array of frames [num_frames, height, width, channels]
            batch_size: Batch size for processing
            
        Returns:
            Frame embeddings tensor
        """
        if len(frames) == 0:
            return torch.empty(0, self.model.config.projection_dim).to(self.device)
        
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(frames), batch_size):
                batch_frames = frames[i:i + batch_size]
                
                # Convert frames to PIL Images
                images = [Image.fromarray(frame.astype(np.uint8)) for frame in batch_frames]
                
                # Process images
                inputs = self.processor(images=images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get embeddings
                image_features = self.model.get_image_features(**inputs)
                
                # Normalize embeddings
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                embeddings.append(image_features)
        
        return torch.cat(embeddings, dim=0)
    
    def encode_text(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Encode text descriptions into embeddings.
        
        Args:
            texts: List of text descriptions
            batch_size: Batch size for processing
            
        Returns:
            Text embeddings tensor
        """
        if not texts:
            return torch.empty(0, self.model.config.projection_dim).to(self.device)
        
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Process texts
                inputs = self.processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get embeddings
                text_features = self.model.get_text_features(**inputs)
                
                # Normalize embeddings
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                embeddings.append(text_features)
        
        return torch.cat(embeddings, dim=0)
    
    def compute_similarity(self, video_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between video and text embeddings.
        
        Args:
            video_embeddings: Video embeddings tensor
            text_embeddings: Text embeddings tensor
            
        Returns:
            Similarity matrix
        """
        # Compute cosine similarity
        similarity = torch.matmul(video_embeddings, text_embeddings.T)
        return similarity
    
    def aggregate_frame_embeddings(self, frame_embeddings: torch.Tensor, method: str = "mean") -> torch.Tensor:
        """
        Aggregate frame embeddings into a single video embedding.
        
        Args:
            frame_embeddings: Frame embeddings tensor [num_frames, embedding_dim]
            method: Aggregation method ("mean", "max", "attention")
            
        Returns:
            Aggregated video embedding
        """
        if len(frame_embeddings) == 0:
            return torch.zeros(self.model.config.projection_dim).to(self.device)
        
        if method == "mean":
            return torch.mean(frame_embeddings, dim=0)
        elif method == "max":
            return torch.max(frame_embeddings, dim=0)[0]
        elif method == "attention":
            # Simple attention mechanism
            attention_weights = torch.softmax(torch.sum(frame_embeddings ** 2, dim=1), dim=0)
            return torch.sum(frame_embeddings * attention_weights.unsqueeze(1), dim=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def get_video_embedding(self, frames: np.ndarray, aggregation_method: str = "attention") -> torch.Tensor:
        """
        Get a single embedding for a video segment.
        
        Args:
            frames: Video frames
            aggregation_method: How to aggregate frame embeddings
            
        Returns:
            Video embedding
        """
        frame_embeddings = self.encode_frames(frames)
        return self.aggregate_frame_embeddings(frame_embeddings, aggregation_method)
    
    def batch_encode_videos(self, video_frames_list: List[np.ndarray]) -> List[torch.Tensor]:
        """
        Encode multiple video segments efficiently.
        
        Args:
            video_frames_list: List of video frame arrays
            
        Returns:
            List of video embeddings
        """
        embeddings = []
        
        for frames in video_frames_list:
            embedding = self.get_video_embedding(frames)
            embeddings.append(embedding)
        
        return embeddings
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "model_name": self.config.models.video_encoder,
            "device": str(self.device),
            "precision": self.config.processing.precision,
            "embedding_dim": self.model.config.projection_dim,
            "max_text_length": self.model.config.text_config.max_position_embeddings
        }