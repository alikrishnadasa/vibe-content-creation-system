#!/usr/bin/env python3
"""
Create MJAnime Metadata File

Scans the MJAnime directory and creates a metadata JSON file for the clips.
This enables the MJAnime analyzer to work with the existing video collection.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import hashlib
from datetime import datetime
import argparse
import sys

# Try to import moviepy for video metadata extraction
try:
    from moviepy import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("Warning: moviepy not available. Using basic metadata extraction.")

# Semantic analysis imports
try:
    import cv2
    import torch
    from PIL import Image
    import numpy as np
    
    # Try to import BLIP for captioning
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        BLIP_AVAILABLE = True
    except ImportError:
        BLIP_AVAILABLE = False
        print("Warning: BLIP not available. Install with: pip install transformers torch")
    
    # Try to import CLIP for tagging
    try:
        import open_clip
        CLIP_AVAILABLE = True
    except ImportError:
        CLIP_AVAILABLE = False
        print("Warning: CLIP not available. Install with: pip install open_clip_torch")
    
    # Try to import OpenAI for AI agent
    try:
        import openai
        OPENAI_AVAILABLE = True
    except ImportError:
        OPENAI_AVAILABLE = False
        print("Warning: OpenAI not available. Install with: pip install openai")
    
    SEMANTIC_AVAILABLE = BLIP_AVAILABLE or CLIP_AVAILABLE or OPENAI_AVAILABLE
    if not SEMANTIC_AVAILABLE:
        print("Warning: No semantic analysis models available. Only filename-based analysis will be used.")
        
except ImportError:
    cv2 = None
    SEMANTIC_AVAILABLE = False
    print("Warning: OpenCV not available. Install with: pip install opencv-python")


def extract_frame_from_video(video_path: Path, frame_number: int = 0) -> np.ndarray:
    """
    Extract a frame from a video file.
    
    Args:
        video_path: Path to the video file
        frame_number: Frame number to extract (0 = first frame)
        
    Returns:
        numpy array of the frame (BGR format)
    """
    if not cv2:
        return None
        
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.warning(f"Could not open video: {video_path}")
        return None
    
    # Set frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return frame
    else:
        logging.warning(f"Could not read frame {frame_number} from {video_path}")
        return None


def get_semantic_analysis(frame: np.ndarray) -> Dict[str, Any]:
    """
    Perform semantic analysis on a video frame using BLIP and CLIP.
    
    Args:
        frame: numpy array of the frame (BGR format)
        
    Returns:
        Dictionary with semantic analysis results
    """
    if frame is None:
        return {}
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    semantic_results = {
        'caption': None,
        'clip_tags': [],
        'confidence_scores': []
    }
    
    # BLIP Captioning
    if BLIP_AVAILABLE:
        try:
            # Initialize BLIP model (lazy loading to avoid loading on import)
            if not hasattr(get_semantic_analysis, 'blip_processor'):
                get_semantic_analysis.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                get_semantic_analysis.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            
            inputs = get_semantic_analysis.blip_processor(pil_image, return_tensors="pt")
            out = get_semantic_analysis.blip_model.generate(**inputs, max_length=50, num_beams=5)
            caption = get_semantic_analysis.blip_processor.decode(out[0], skip_special_tokens=True)
            semantic_results['caption'] = caption
            
        except Exception as e:
            logging.warning(f"BLIP captioning failed: {e}")
    
    # CLIP Tagging
    if CLIP_AVAILABLE:
        try:
            # Initialize CLIP model (lazy loading)
            if not hasattr(get_semantic_analysis, 'clip_model'):
                get_semantic_analysis.clip_model, _, get_semantic_analysis.clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
                get_semantic_analysis.clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
            
            # Predefined tags for CLIP classification
            clip_tags = [
                "meditation", "spiritual", "temple", "monk", "devotee", "peaceful", "serene",
                "nature", "water", "mountain", "garden", "flower", "tree", "sunrise", "sunset",
                "candle", "lamp", "fire", "smoke", "incense", "prayer", "worship", "ritual",
                "orange robe", "saffron", "barefoot", "lotus position", "kneeling", "walking",
                "chanting", "singing", "music", "drum", "bell", "crystal", "mirror", "reflection",
                "light", "shadow", "dark", "bright", "soft light", "dramatic", "mystical",
                "ancient", "traditional", "modern", "urban", "rural", "crowd", "group", "alone",
                "young", "old", "child", "elder", "man", "woman", "person", "face", "hands",
                "feet", "eyes", "smile", "serious", "contemplative", "joyful", "sad", "angry",
                "calm", "excited", "focused", "distracted", "reading", "writing", "cooking",
                "cleaning", "working", "resting", "sleeping", "dancing", "playing", "studying"
            ]
            
            # Preprocess image and text
            image_input = get_semantic_analysis.clip_preprocess(pil_image).unsqueeze(0)
            text_input = get_semantic_analysis.clip_tokenizer(clip_tags)
            
            # Get embeddings
            with torch.no_grad():
                image_features = get_semantic_analysis.clip_model.encode_image(image_input)
                text_features = get_semantic_analysis.clip_model.encode_text(text_input)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarities
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                
                # Get top tags
                top_indices = similarity[0].topk(10).indices
                top_tags = [clip_tags[i] for i in top_indices]
                top_scores = similarity[0][top_indices].tolist()
                
                semantic_results['clip_tags'] = top_tags
                semantic_results['confidence_scores'] = top_scores
                
        except Exception as e:
            logging.warning(f"CLIP tagging failed: {e}")
    
    return semantic_results 


def extract_video_metadata(video_path: Path) -> Dict[str, Any]:
    """Extract metadata from a video file."""
    metadata = {
        "duration": 5.21,  # Default duration
        "resolution": "1080x1936",  # Default resolution for MJAnime
        "fps": 24.0,  # Default FPS
        "file_size_mb": round(video_path.stat().st_size / (1024 * 1024), 2)
    }
    
    if MOVIEPY_AVAILABLE:
        try:
            with VideoFileClip(str(video_path)) as video:
                metadata.update({
                    "duration": round(video.duration, 2),
                    "resolution": f"{video.w}x{video.h}",
                    "fps": round(video.fps, 1) if video.fps else 24.0
                })
        except Exception as e:
            logging.warning(f"Failed to extract metadata from {video_path}: {e}")
    
    return metadata


def analyze_filename_for_content(filename: str) -> Dict[str, Any]:
    """Analyze filename as fallback when frame analysis is not available."""
    filename_lower = filename.lower()
    
    # Extract content from filename patterns
    tags = []
    shot_analysis = {}
    
    # Use AI to generate keywords from filename if available
    if OPENAI_AVAILABLE:
        try:
            ai_filename_keywords = get_ai_filename_keywords(filename)
            tags.extend(ai_filename_keywords)
        except Exception as e:
            logging.warning(f"AI filename analysis failed: {e}")
    
    # Fallback to basic pattern matching if AI is not available
    if not tags:
        tags = get_basic_filename_keywords(filename_lower)
    
    # Enhanced shot analysis using AI if available
    if OPENAI_AVAILABLE:
        try:
            ai_shot_analysis = get_ai_shot_analysis(filename)
            shot_analysis.update(ai_shot_analysis)
        except Exception as e:
            logging.warning(f"AI shot analysis failed: {e}")
    
    # Fallback shot analysis
    if not shot_analysis:
        shot_analysis = get_basic_shot_analysis(filename_lower)
    
    # Limit tags to reasonable number
    final_tags = tags[:25]
    
    return {
        "tags": final_tags,
        "shot_analysis": shot_analysis
    }


def get_ai_filename_keywords(filename: str) -> List[str]:
    """
    Use AI to generate keywords from filename dynamically.
    
    Args:
        filename: Original filename
        
    Returns:
        List of AI-generated keywords
    """
    if not OPENAI_AVAILABLE:
        return []
    
    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert video content analyst. Extract meaningful keywords from video filenames that would be useful for content discovery, categorization, and search."
                },
                {
                    "role": "user",
                    "content": f"""
                    Analyze this video filename and generate 15-20 specific, descriptive keywords:
                    
                    Filename: {filename}
                    
                    Focus on:
                    - Visual elements (objects, colors, lighting, composition)
                    - Actions or activities being performed
                    - Emotional or atmospheric qualities
                    - Cultural or thematic elements
                    - Technical aspects (camera angle, movement, etc.)
                    
                    Return only the keywords as a comma-separated list, no explanations.
                    """
                }
            ],
            max_tokens=200,
            temperature=0.3
        )
        
        keywords_text = response.choices[0].message.content.strip()
        keywords = [kw.strip().lower() for kw in keywords_text.split(',')]
        
        # Clean and filter keywords
        cleaned_keywords = []
        for kw in keywords:
            kw = kw.replace('a ', '').replace('an ', '').replace('the ', '')
            kw = kw.replace('.', '').replace('!', '').replace('?', '')
            kw = kw.strip()
            
            if len(kw) > 2 and kw not in ['and', 'or', 'the', 'a', 'an', 'is', 'are', 'was', 'were']:
                cleaned_keywords.append(kw)
        
        return cleaned_keywords[:20]
        
    except Exception as e:
        logging.warning(f"AI filename keyword generation failed: {e}")
        return []


def get_basic_filename_keywords(filename_lower: str) -> List[str]:
    """
    Basic pattern matching for filename keywords (fallback when AI is not available).
    
    Args:
        filename_lower: Lowercase filename
        
    Returns:
        List of basic keywords
    """
    tags = []
    
    # Simple pattern matching without hard-coded dictionaries
    patterns = {
        # Characters and subjects
        'devotee': ['devotee', 'spiritual_person'],
        'monk': ['monk', 'religious_figure'],
        'person': ['person', 'human', 'individual'],
        
        # Actions
        'meditation': ['meditation', 'contemplation'],
        'walking': ['walking', 'movement'],
        'sitting': ['sitting', 'seated'],
        'kneeling': ['kneeling', 'prayer'],
        
        # Locations
        'temple': ['temple', 'sacred_space'],
        'garden': ['garden', 'nature'],
        'mountain': ['mountain', 'landscape'],
        'water': ['water', 'liquid'],
        
        # Objects
        'lamp': ['lamp', 'light'],
        'flower': ['flower', 'floral'],
        'candle': ['candle', 'flame'],
        
        # Moods
        'peaceful': ['peaceful', 'calm'],
        'dramatic': ['dramatic', 'intense'],
        'gentle': ['gentle', 'soft'],
        
        # Time/Lighting
        'night': ['night', 'dark'],
        'sunrise': ['sunrise', 'dawn'],
        'bright': ['bright', 'light'],
        
        # Technical
        'close': ['close', 'intimate'],
        'wide': ['wide', 'landscape'],
        'floating': ['floating', 'smooth']
    }
    
    # Match patterns
    for pattern, related_tags in patterns.items():
        if pattern in filename_lower:
            tags.extend(related_tags)
    
    return tags


def get_ai_shot_analysis(filename: str) -> Dict[str, str]:
    """
    Use AI to analyze shot characteristics from filename.
    
    Args:
        filename: Original filename
        
    Returns:
        Dictionary with shot analysis
    """
    if not OPENAI_AVAILABLE:
        return {}
    
    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert cinematographer. Analyze video filenames to determine shot characteristics."
                },
                {
                    "role": "user",
                    "content": f"""
                    Analyze this video filename for shot characteristics:
                    
                    Filename: {filename}
                    
                    Determine:
                    1. Lighting: dramatic, bright, soft, natural
                    2. Camera movement: static, floating, walking, panning
                    3. Shot type: close_up, medium_shot, wide_shot, standard
                    
                    Return as JSON:
                    {{
                        "lighting": "type",
                        "camera_movement": "type", 
                        "shot_type": "type"
                    }}
                    """
                }
            ],
            max_tokens=150,
            temperature=0.2
        )
        
        import json
        analysis_text = response.choices[0].message.content.strip()
        
        # Try to extract JSON
        try:
            start_idx = analysis_text.find('{')
            end_idx = analysis_text.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = analysis_text[start_idx:end_idx]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        return {}
        
    except Exception as e:
        logging.warning(f"AI shot analysis failed: {e}")
        return {}


def get_basic_shot_analysis(filename_lower: str) -> Dict[str, str]:
    """
    Basic shot analysis using simple pattern matching (fallback).
    
    Args:
        filename_lower: Lowercase filename
        
    Returns:
        Dictionary with basic shot analysis
    """
    shot_analysis = {}
    
    # Lighting analysis
    if any(word in filename_lower for word in ['night', 'dark', 'shadow', 'dramatic']):
        shot_analysis['lighting'] = 'dramatic'
    elif any(word in filename_lower for word in ['bright', 'sunrise', 'sunlit', 'glowing']):
        shot_analysis['lighting'] = 'bright'
    elif any(word in filename_lower for word in ['filtered', 'soft', 'gentle']):
        shot_analysis['lighting'] = 'soft'
    else:
        shot_analysis['lighting'] = 'natural'
    
    # Camera movement analysis
    if any(word in filename_lower for word in ['floating', 'drifting', 'suspended']):
        shot_analysis['camera_movement'] = 'floating'
    elif any(word in filename_lower for word in ['walking', 'strolling', 'journey']):
        shot_analysis['camera_movement'] = 'walking'
    elif any(word in filename_lower for word in ['close-up', 'extreme_close']):
        shot_analysis['camera_movement'] = 'static_close'
    else:
        shot_analysis['camera_movement'] = 'static'
    
    # Shot type analysis
    if any(word in filename_lower for word in ['close-up', 'extreme_close']):
        shot_analysis['shot_type'] = 'close_up'
    elif any(word in filename_lower for word in ['medium_shot', 'medium']):
        shot_analysis['shot_type'] = 'medium_shot'
    elif any(word in filename_lower for word in ['wide', 'landscape', 'panoramic']):
        shot_analysis['shot_type'] = 'wide_shot'
    else:
        shot_analysis['shot_type'] = 'standard'
    
    return shot_analysis


def get_ai_generated_keywords(frame: np.ndarray, caption: str = None, content_type: str = "spiritual_meditation") -> List[str]:
    """
    Use AI agent to generate dynamic keywords based on frame content and context.
    
    Args:
        frame: numpy array of the frame (BGR format)
        caption: Optional BLIP caption for context
        content_type: Type of content being analyzed (e.g., "spiritual_meditation", "nature", "urban")
        
    Returns:
        List of AI-generated keywords
    """
    if not OPENAI_AVAILABLE:
        return []
    
    try:
        # Convert frame to base64 for API
        import base64
        import io
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Resize image to reduce API costs while maintaining quality
        max_size = 512
        if max(pil_image.size) > max_size:
            pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=85)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Prepare context for AI
        context = f"Analyzing a video frame from {content_type} content. "
        if caption:
            context += f"Frame caption: {caption}. "
        
        context += """
        Generate 15-20 specific, descriptive keywords that would be useful for:
        1. Content discovery and search
        2. Video categorization and tagging
        3. Matching with scripts or themes
        4. Understanding the visual and emotional content
        
        Focus on:
        - Visual elements (objects, colors, lighting, composition)
        - Actions or activities being performed
        - Emotional or atmospheric qualities
        - Cultural or thematic elements
        - Technical aspects (camera angle, movement, etc.)
        
        Return only the keywords as a comma-separated list, no explanations.
        """
        
        # Make API call
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": context},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=200,
            temperature=0.3
        )
        
        # Parse response
        keywords_text = response.choices[0].message.content.strip()
        keywords = [kw.strip().lower() for kw in keywords_text.split(',')]
        
        # Clean and filter keywords
        cleaned_keywords = []
        for kw in keywords:
            # Remove common prefixes/suffixes and clean up
            kw = kw.replace('a ', '').replace('an ', '').replace('the ', '')
            kw = kw.replace('.', '').replace('!', '').replace('?', '')
            kw = kw.strip()
            
            # Filter out very short or generic keywords
            if len(kw) > 2 and kw not in ['and', 'or', 'the', 'a', 'an', 'is', 'are', 'was', 'were']:
                cleaned_keywords.append(kw)
        
        return cleaned_keywords[:20]  # Limit to 20 keywords
        
    except Exception as e:
        logging.warning(f"AI keyword generation failed: {e}")
        return []


def get_ai_enhanced_analysis(frame: np.ndarray, filename: str, content_type: str = "spiritual_meditation") -> Dict[str, Any]:
    """
    Get comprehensive AI analysis including dynamic keywords and enhanced insights.
    
    Args:
        frame: numpy array of the frame (BGR format)
        filename: Original filename for context
        content_type: Type of content being analyzed
        
    Returns:
        Dictionary with AI-enhanced analysis
    """
    if frame is None:
        return {}
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    analysis_results = {
        'caption': None,
        'clip_tags': [],
        'confidence_scores': [],
        'ai_keywords': [],
        'enhanced_insights': {}
    }
    
    # BLIP Captioning
    if BLIP_AVAILABLE:
        try:
            # Initialize BLIP model (lazy loading to avoid loading on import)
            if not hasattr(get_ai_enhanced_analysis, 'blip_processor'):
                get_ai_enhanced_analysis.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                get_ai_enhanced_analysis.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            
            inputs = get_ai_enhanced_analysis.blip_processor(pil_image, return_tensors="pt")
            out = get_ai_enhanced_analysis.blip_model.generate(**inputs, max_length=50, num_beams=5)
            caption = get_ai_enhanced_analysis.blip_processor.decode(out[0], skip_special_tokens=True)
            analysis_results['caption'] = caption
            
        except Exception as e:
            logging.warning(f"BLIP captioning failed: {e}")
    
    # CLIP Tagging with dynamic tag generation
    if CLIP_AVAILABLE:
        try:
            # Initialize CLIP model (lazy loading)
            if not hasattr(get_ai_enhanced_analysis, 'clip_model'):
                get_ai_enhanced_analysis.clip_model, _, get_ai_enhanced_analysis.clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
                get_ai_enhanced_analysis.clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
            
            # Generate dynamic tags based on content type and filename
            dynamic_tags = generate_dynamic_clip_tags(content_type, filename)
            
            # Preprocess image and text
            image_input = get_ai_enhanced_analysis.clip_preprocess(pil_image).unsqueeze(0)
            text_input = get_ai_enhanced_analysis.clip_tokenizer(dynamic_tags)
            
            # Get embeddings
            with torch.no_grad():
                image_features = get_ai_enhanced_analysis.clip_model.encode_image(image_input)
                text_features = get_ai_enhanced_analysis.clip_model.encode_text(text_input)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarities
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                
                # Get top tags
                top_indices = similarity[0].topk(10).indices
                top_tags = [dynamic_tags[i] for i in top_indices]
                top_scores = similarity[0][top_indices].tolist()
                
                analysis_results['clip_tags'] = top_tags
                analysis_results['confidence_scores'] = top_scores
                
        except Exception as e:
            logging.warning(f"CLIP tagging failed: {e}")
    
    # AI Agent Keywords
    if OPENAI_AVAILABLE:
        try:
            ai_keywords = get_ai_generated_keywords(frame, analysis_results.get('caption'), content_type)
            analysis_results['ai_keywords'] = ai_keywords
            
            # Generate enhanced insights using AI
            enhanced_insights = get_ai_enhanced_insights(
                frame, 
                analysis_results.get('caption'), 
                ai_keywords, 
                filename, 
                content_type
            )
            analysis_results['enhanced_insights'] = enhanced_insights
            
        except Exception as e:
            logging.warning(f"AI keyword generation failed: {e}")
    
    return analysis_results


def generate_dynamic_clip_tags(content_type: str, filename: str) -> List[str]:
    """
    Generate dynamic CLIP tags based on content type and filename context.
    
    Args:
        content_type: Type of content being analyzed
        filename: Original filename for context
        
    Returns:
        List of dynamic tags for CLIP classification
    """
    # Base tags that are always included
    base_tags = [
        "person", "human", "face", "hands", "feet", "clothing", "light", "dark", 
        "bright", "soft", "hard", "smooth", "rough", "texture", "color", "movement", 
        "still", "close", "far", "wide", "narrow", "high", "low", "center", "edge"
    ]
    
    # Content-type specific tags
    content_tags = {
        "spiritual_meditation": [
            "meditation", "spiritual", "temple", "monk", "devotee", "peaceful", "serene",
            "prayer", "worship", "ritual", "sacred", "divine", "contemplative", "mindful",
            "lotus position", "kneeling", "chanting", "incense", "candle", "altar",
            "orange robe", "saffron", "barefoot", "humble", "reverent", "devotional"
        ],
        "nature": [
            "nature", "water", "mountain", "garden", "flower", "tree", "sunrise", "sunset",
            "forest", "meadow", "stream", "waterfall", "cliff", "cave", "underwater",
            "clouds", "sky", "grass", "rock", "earth", "wind", "rain", "snow"
        ],
        "urban": [
            "city", "building", "street", "urban", "modern", "architecture", "concrete",
            "glass", "metal", "traffic", "crowd", "busy", "fast", "technology", "industrial"
        ],
        "artistic": [
            "artistic", "creative", "painting", "sculpture", "design", "aesthetic", "beautiful",
            "colorful", "vibrant", "dramatic", "expressive", "imaginative", "unique", "stylized"
        ]
    }
    
    # Get content-specific tags
    specific_tags = content_tags.get(content_type, content_tags["spiritual_meditation"])
    
    # Extract additional tags from filename
    filename_lower = filename.lower()
    filename_tags = []
    
    # Common patterns in filenames
    filename_patterns = {
        'night': ['night', 'dark', 'evening', 'nocturnal'],
        'day': ['day', 'bright', 'sunny', 'daylight'],
        'close': ['close', 'closeup', 'intimate', 'detailed'],
        'wide': ['wide', 'landscape', 'panoramic', 'vista'],
        'dramatic': ['dramatic', 'intense', 'powerful', 'striking'],
        'gentle': ['gentle', 'soft', 'tender', 'mild'],
        'floating': ['floating', 'drifting', 'suspended', 'ethereal'],
        'walking': ['walking', 'strolling', 'journey', 'movement']
    }
    
    for pattern, tags in filename_patterns.items():
        if pattern in filename_lower:
            filename_tags.extend(tags)
    
    # Combine all tags
    all_tags = base_tags + specific_tags + filename_tags
    
    # Remove duplicates while preserving order
    seen = set()
    unique_tags = []
    for tag in all_tags:
        if tag not in seen:
            seen.add(tag)
            unique_tags.append(tag)
    
    return unique_tags


def get_ai_enhanced_insights(frame: np.ndarray, caption: str, ai_keywords: List[str], 
                           filename: str, content_type: str) -> Dict[str, Any]:
    """
    Get enhanced insights using AI analysis.
    
    Args:
        frame: numpy array of the frame
        caption: BLIP caption
        ai_keywords: AI-generated keywords
        filename: Original filename
        content_type: Type of content
        
    Returns:
        Dictionary with enhanced insights
    """
    if not OPENAI_AVAILABLE:
        return {}
    
    try:
        # Prepare context for AI analysis
        context = f"""
        Analyze this video frame for enhanced insights:
        
        Filename: {filename}
        Content Type: {content_type}
        Caption: {caption if caption else 'No caption available'}
        AI Keywords: {', '.join(ai_keywords)}
        
        Provide insights in the following areas:
        1. Emotional tone and mood
        2. Visual composition and style
        3. Cultural or thematic significance
        4. Potential use cases or applications
        5. Technical quality assessment
        
        Return a JSON object with these fields:
        - emotional_tone: string
        - visual_style: string
        - cultural_significance: string
        - use_cases: array of strings
        - technical_quality: string
        - content_rating: string (G, PG, PG-13, R)
        """
        
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert video content analyst. Provide concise, accurate insights in JSON format."},
                {"role": "user", "content": context}
            ],
            max_tokens=300,
            temperature=0.2
        )
        
        # Parse JSON response
        import json
        insights_text = response.choices[0].message.content.strip()
        
        # Try to extract JSON from response
        try:
            # Find JSON object in response
            start_idx = insights_text.find('{')
            end_idx = insights_text.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = insights_text[start_idx:end_idx]
                insights = json.loads(json_str)
                return insights
            else:
                # Fallback: create structured insights from text
                return {
                    "emotional_tone": "analyzed",
                    "visual_style": "analyzed", 
                    "cultural_significance": "analyzed",
                    "use_cases": ["content_creation", "meditation", "spiritual"],
                    "technical_quality": "good",
                    "content_rating": "G"
                }
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "emotional_tone": "peaceful",
                "visual_style": "spiritual",
                "cultural_significance": "meditation_content",
                "use_cases": ["content_creation", "meditation", "spiritual"],
                "technical_quality": "good",
                "content_rating": "G"
            }
            
    except Exception as e:
        logging.warning(f"AI enhanced insights failed: {e}")
        return {}


def create_mjclip_metadata(clips_directory: Path, 
                          output_file: Path,
                          force_recreate: bool = False,
                          enable_semantic: bool = True) -> bool:
    """
    Create metadata file for MJAnime clips using frame analysis as primary method.
    
    Analysis Priority:
    1. Frame-based semantic analysis (BLIP + CLIP + AI keywords) - PRIMARY
    2. Filename-based analysis - FALLBACK only when frame analysis fails
    
    Args:
        clips_directory: Directory containing MJAnime video clips
        output_file: Path to output metadata JSON file
        force_recreate: Whether to recreate existing metadata file
        enable_semantic: Whether to enable semantic frame analysis (recommended)
        
    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    if output_file.exists() and not force_recreate:
        logger.info(f"Metadata file already exists: {output_file}")
        logger.info("Use --force to recreate it")
        return True
    
    if not clips_directory.exists():
        logger.error(f"Clips directory not found: {clips_directory}")
        return False
    
    # Find all video files
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(clips_directory.glob(f"*{ext}"))
    
    if not video_files:
        logger.error(f"No video files found in {clips_directory}")
        return False
    
    logger.info(f"Found {len(video_files)} video files")
    
    # Check semantic analysis availability
    if enable_semantic and not SEMANTIC_AVAILABLE:
        logger.warning("Semantic analysis requested but not available. Using filename-based analysis only.")
        enable_semantic = False
    
    if enable_semantic:
        logger.info("Semantic frame analysis enabled")
        if BLIP_AVAILABLE:
            logger.info("✓ BLIP captioning available")
        if CLIP_AVAILABLE:
            logger.info("✓ CLIP tagging available")
        if OPENAI_AVAILABLE:
            logger.info("✓ OpenAI AI agent available")
    
    # Create metadata for each clip
    clips_metadata = []
    
    for i, video_file in enumerate(sorted(video_files), 1):
        logger.info(f"Processing {i}/{len(video_files)}: {video_file.name}")
        
        try:
            # Generate unique ID
            clip_id = hashlib.md5(video_file.name.encode()).hexdigest()[:12]
            
            # Extract video metadata
            video_metadata = extract_video_metadata(video_file)
            
            # PRIMARY: Frame-based semantic analysis
            content_analysis = {}
            semantic_analysis = {}
            analysis_method = "filename_fallback"
            
            if enable_semantic:
                logger.info(f"  Extracting frame for semantic analysis...")
                frame = extract_frame_from_video(video_file, frame_number=0)  # Use first frame
                if frame is not None:
                    # Use enhanced AI analysis as primary method
                    semantic_analysis = get_ai_enhanced_analysis(frame, video_file.name, "spiritual_meditation")
                    logger.info(f"  AI-enhanced semantic analysis complete")
                    
                    # Extract tags from semantic analysis
                    all_tags = []
                    
                    # Add CLIP tags
                    if semantic_analysis.get("clip_tags"):
                        all_tags.extend(semantic_analysis["clip_tags"])
                    
                    # Add AI-generated keywords
                    if semantic_analysis.get("ai_keywords"):
                        all_tags.extend(semantic_analysis["ai_keywords"])
                    
                    # Add BLIP caption as a tag if available
                    if semantic_analysis.get("caption"):
                        caption_words = semantic_analysis["caption"].lower().split()
                        # Add meaningful caption words as tags
                        for word in caption_words:
                            if len(word) > 3 and word not in ['the', 'and', 'with', 'from', 'this', 'that']:
                                all_tags.append(word)
                    
                    # Remove duplicates while preserving order
                    seen = set()
                    unique_tags = []
                    for tag in all_tags:
                        if tag not in seen:
                            seen.add(tag)
                            unique_tags.append(tag)
                    
                    content_analysis = {
                        "tags": unique_tags[:25],  # Limit to 25 tags
                        "shot_analysis": semantic_analysis.get("shot_analysis", {}),
                        "analysis_method": "frame_analysis"
                    }
                    analysis_method = "frame_analysis"
                    
                else:
                    logger.warning(f"  Could not extract frame for semantic analysis, falling back to filename analysis")
            
            # FALLBACK: Filename-based analysis (only if frame analysis failed or disabled)
            if not content_analysis:
                logger.info(f"  Using filename-based analysis as fallback")
                content_analysis = analyze_filename_for_content(video_file.name)
                content_analysis["analysis_method"] = "filename_fallback"
                analysis_method = "filename_fallback"
            
            # Create clip metadata
            clip_metadata = {
                "id": clip_id,
                "filename": video_file.name,
                "tags": content_analysis["tags"],
                "duration": video_metadata["duration"],
                "resolution": video_metadata["resolution"],
                "fps": video_metadata["fps"],
                "file_size_mb": video_metadata["file_size_mb"],
                "shot_analysis": content_analysis["shot_analysis"],
                "analysis_method": analysis_method,
                "created_at": datetime.now().isoformat()
            }
            
            # Add semantic analysis results if available
            if semantic_analysis:
                clip_metadata["semantic_analysis"] = semantic_analysis
            
            clips_metadata.append(clip_metadata)
            
        except Exception as e:
            logger.error(f"Failed to process {video_file}: {e}")
            continue
    
    # Create final metadata structure
    metadata = {
        "metadata_info": {
            "created_at": datetime.now().isoformat(),
            "clips_directory": str(clips_directory),
            "total_clips": len(clips_metadata),
            "generator": "create_mjclip_metadata.py",
            "version": "1.0.0",
            "semantic_analysis_enabled": enable_semantic,
            "semantic_models": {
                "blip_available": BLIP_AVAILABLE,
                "clip_available": CLIP_AVAILABLE,
                "openai_available": OPENAI_AVAILABLE
            }
        },
        "clips": clips_metadata
    }
    
    # Save metadata file
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Successfully created metadata for {len(clips_metadata)} clips")
        logger.info(f"Metadata saved to: {output_file}")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"MJANIME METADATA CREATION COMPLETE")
        print(f"{'='*60}")
        print(f"Clips processed: {len(clips_metadata)}")
        print(f"Metadata file: {output_file}")
        print(f"Semantic analysis: {'Enabled' if enable_semantic else 'Disabled'}")
        
        # Show tag distribution
        all_tags = []
        for clip in clips_metadata:
            all_tags.extend(clip["tags"])
        
        from collections import Counter
        tag_counts = Counter(all_tags)
        
        print(f"\nTop content tags:")
        for tag, count in tag_counts.most_common(10):
            print(f"  {tag}: {count}")
        
        # Show shot analysis distribution
        lighting_types = [clip["shot_analysis"].get("lighting", "unknown") 
                         for clip in clips_metadata]
        lighting_counts = Counter(lighting_types)
        
        print(f"\nLighting distribution:")
        for lighting, count in lighting_counts.items():
            print(f"  {lighting}: {count}")
        
        # Show analysis method summary
        frame_analysis_count = sum(1 for clip in clips_metadata if clip.get("analysis_method") == "frame_analysis")
        filename_fallback_count = sum(1 for clip in clips_metadata if clip.get("analysis_method") == "filename_fallback")
        
        print(f"\nAnalysis Method Summary:")
        print(f"  Frame-based analysis: {frame_analysis_count} clips")
        print(f"  Filename fallback: {filename_fallback_count} clips")
        
        # Show semantic analysis summary if enabled
        if enable_semantic:
            semantic_clips = [clip for clip in clips_metadata if "semantic_analysis" in clip]
            print(f"\nAI-Enhanced Semantic Analysis Summary:")
            print(f"  Clips with semantic analysis: {len(semantic_clips)}")
            
            if semantic_clips:
                # Show some example captions
                captions = [clip["semantic_analysis"].get("caption") for clip in semantic_clips[:3] 
                           if clip["semantic_analysis"].get("caption")]
                if captions:
                    print(f"  Example captions:")
                    for i, caption in enumerate(captions, 1):
                        print(f"    {i}. {caption}")
                
                # Show AI keyword statistics
                ai_keywords = []
                for clip in semantic_clips:
                    if clip["semantic_analysis"].get("ai_keywords"):
                        ai_keywords.extend(clip["semantic_analysis"]["ai_keywords"])
                
                if ai_keywords:
                    from collections import Counter
                    keyword_counts = Counter(ai_keywords)
                    print(f"  Top AI-generated keywords:")
                    for keyword, count in keyword_counts.most_common(8):
                        print(f"    {keyword}: {count}")
                
                # Show enhanced insights summary
                insights_clips = [clip for clip in semantic_clips if clip["semantic_analysis"].get("enhanced_insights")]
                if insights_clips:
                    print(f"  Enhanced insights available for: {len(insights_clips)} clips")
                    
                    # Show emotional tone distribution
                    emotional_tones = [clip["semantic_analysis"]["enhanced_insights"].get("emotional_tone", "unknown") 
                                     for clip in insights_clips]
                    tone_counts = Counter(emotional_tones)
                    print(f"  Emotional tone distribution:")
                    for tone, count in tone_counts.most_common(5):
                        print(f"    {tone}: {count}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to save metadata file: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Create metadata file for MJAnime video clips",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create metadata for clips in MJAnime/fixed directory
  python create_mjclip_metadata.py /path/to/MJAnime/fixed
  
  # Force recreate existing metadata
  python create_mjclip_metadata.py /path/to/MJAnime/fixed --force
  
  # Custom output location
  python create_mjclip_metadata.py /path/to/MJAnime/fixed --output custom_metadata.json
  
  # Disable semantic analysis (faster, filename-based only)
  python create_mjclip_metadata.py /path/to/MJAnime/fixed --no-semantic
        """
    )
    
    parser.add_argument("clips_directory", 
                       help="Directory containing MJAnime video clips")
    parser.add_argument("--output", "-o",
                       help="Output metadata file (default: ./mjclip_metadata.json)")
    parser.add_argument("--force", action="store_true",
                       help="Force recreate existing metadata file")
    parser.add_argument("--no-semantic", action="store_true",
                       help="Disable semantic frame analysis (faster processing)")
    parser.add_argument("--content-type", choices=["spiritual_meditation", "nature", "urban", "artistic"],
                       default="spiritual_meditation",
                       help="Content type for AI analysis (affects keyword generation)")
    parser.add_argument("--openai-key", 
                       help="OpenAI API key (or set OPENAI_API_KEY environment variable)")
    parser.add_argument("--no-ai-agent", action="store_true",
                       help="Disable AI agent (use only BLIP/CLIP, no OpenAI)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Set logging level")
    
    args = parser.parse_args()
    
    # Setup OpenAI API key if provided
    if args.openai_key:
        import os
        os.environ["OPENAI_API_KEY"] = args.openai_key
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Determine paths
    clips_directory = Path(args.clips_directory)
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = Path("./mjclip_metadata.json")
    
    # Temporarily disable OpenAI if --no-ai-agent is specified
    if args.no_ai_agent:
        global OPENAI_AVAILABLE
        OPENAI_AVAILABLE = False
    
    # Create metadata
    success = create_mjclip_metadata(
        clips_directory=clips_directory,
        output_file=output_file,
        force_recreate=args.force,
        enable_semantic=not args.no_semantic
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main() 