import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import numpy as np
from typing import List, Dict, Any
import logging

from ..config import get_config


class BLIP2Captioner:
    """BLIP-2 model for generating video frame captions and descriptions."""
    
    def __init__(self, model_name: str = None):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize BLIP-2 model - force CPU usage
        model_name = model_name or self.config.models.blip2_model
        self.device = torch.device("cpu")
        
        try:
            self.processor = Blip2Processor.from_pretrained(model_name)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_name, 
                torch_dtype=torch.float32  # Force float32 for CPU compatibility
            )
            self.model.to(self.device)
            self.model.eval()
            
            self.logger.info(f"Loaded BLIP-2 model: {model_name} on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load BLIP-2 model: {str(e)}")
            raise
    
    def generate_frame_captions(self, frames: np.ndarray, max_new_tokens: int = 50) -> List[str]:
        """
        Generate captions for video frames.
        
        Args:
            frames: Array of frames [num_frames, height, width, channels]
            max_new_tokens: Maximum tokens for caption generation
            
        Returns:
            List of captions for each frame
        """
        if len(frames) == 0:
            return []
        
        captions = []
        
        with torch.no_grad():
            for frame in frames:
                # Convert frame to PIL Image
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                
                image = Image.fromarray(frame)
                
                # Process image
                inputs = self.processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate caption
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=3,
                    early_stopping=True
                )
                
                # Decode caption
                caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
                captions.append(caption)
        
        return captions
    
    def generate_video_description(self, frames: np.ndarray, sample_rate: int = 5) -> str:
        """
        Generate a comprehensive description of the video segment.
        
        Args:
            frames: Array of frames
            sample_rate: Sample every Nth frame for description
            
        Returns:
            Video description
        """
        if len(frames) == 0:
            return "No visual content detected."
        
        # Sample frames to reduce computation
        sampled_frames = frames[::sample_rate]
        
        # Generate captions for sampled frames
        captions = self.generate_frame_captions(sampled_frames)
        
        if not captions:
            return "Unable to generate video description."
        
        # Combine captions into a coherent description
        description = self._combine_captions(captions)
        
        return description
    
    def _combine_captions(self, captions: List[str]) -> str:
        """
        Combine individual frame captions into a coherent video description.
        
        Args:
            captions: List of frame captions
            
        Returns:
            Combined description
        """
        if not captions:
            return "No captions available."
        
        if len(captions) == 1:
            return captions[0]
        
        # Simple approach: combine unique captions
        unique_captions = []
        seen = set()
        
        for caption in captions:
            if caption not in seen:
                unique_captions.append(caption)
                seen.add(caption)
        
        # Join with temporal indicators
        if len(unique_captions) == 1:
            return unique_captions[0]
        elif len(unique_captions) == 2:
            return f"{unique_captions[0]}, then {unique_captions[1]}"
        else:
            return f"{unique_captions[0]}, followed by {', '.join(unique_captions[1:-1])}, and finally {unique_captions[-1]}"
    
    def generate_conditional_caption(self, frames: np.ndarray, prompt: str, max_new_tokens: int = 50) -> str:
        """
        Generate caption conditioned on a text prompt.
        
        Args:
            frames: Array of frames
            prompt: Text prompt to condition the caption
            max_new_tokens: Maximum tokens for generation
            
        Returns:
            Conditional caption
        """
        if len(frames) == 0:
            return "No visual content for the given prompt."
        
        # Use the middle frame as representative
        middle_idx = len(frames) // 2
        frame = frames[middle_idx]
        
        # Convert to PIL Image
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        
        image = Image.fromarray(frame)
        
        with torch.no_grad():
            # Process with prompt
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate conditional caption
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=3,
                early_stopping=True
            )
            
            # Decode caption
            caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            
            return caption
    
    def analyze_visual_content(self, frames: np.ndarray) -> Dict[str, Any]:
        """
        Analyze visual content and provide detailed insights.
        
        Args:
            frames: Array of frames
            
        Returns:
            Dictionary with visual analysis
        """
        if len(frames) == 0:
            return {"error": "No frames to analyze"}
        
        # Generate captions for analysis
        sample_frames = frames[::max(1, len(frames) // 3)]  # Sample up to 3 frames
        captions = self.generate_frame_captions(sample_frames)
        
        # Generate overall description
        description = self.generate_video_description(frames)
        
        # Extract visual elements (simplified)
        visual_elements = self._extract_visual_elements(captions)
        
        return {
            "overall_description": description,
            "frame_captions": captions,
            "visual_elements": visual_elements,
            "frame_count": len(frames),
            "analysis_method": "BLIP-2 captioning"
        }
    
    def _extract_visual_elements(self, captions: List[str]) -> Dict[str, List[str]]:
        """
        Extract visual elements from captions.
        
        Args:
            captions: List of captions
            
        Returns:
            Dictionary of visual elements
        """
        # Simple keyword extraction for common visual elements
        objects = set()
        actions = set()
        locations = set()
        
        # Common object keywords
        object_keywords = {
            'person', 'people', 'man', 'woman', 'child', 'car', 'building', 'tree', 'dog', 'cat',
            'chair', 'table', 'phone', 'book', 'computer', 'window', 'door', 'hand', 'face'
        }
        
        # Common action keywords
        action_keywords = {
            'walking', 'running', 'sitting', 'standing', 'talking', 'eating', 'drinking', 'reading',
            'writing', 'typing', 'looking', 'holding', 'wearing', 'smiling', 'jumping', 'dancing'
        }
        
        # Common location keywords
        location_keywords = {
            'room', 'office', 'kitchen', 'bedroom', 'street', 'park', 'beach', 'city', 'house',
            'car', 'train', 'airplane', 'restaurant', 'store', 'school', 'hospital'
        }
        
        for caption in captions:
            words = caption.lower().split()
            for word in words:
                if word in object_keywords:
                    objects.add(word)
                elif word in action_keywords:
                    actions.add(word)
                elif word in location_keywords:
                    locations.add(word)
        
        return {
            "objects": list(objects),
            "actions": list(actions),
            "locations": list(locations)
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.config.models.blip2_model,
            "device": str(self.device),
            "precision": self.config.processing.precision,
            "capabilities": [
                "frame_captioning",
                "video_description",
                "conditional_captioning",
                "visual_analysis"
            ]
        }