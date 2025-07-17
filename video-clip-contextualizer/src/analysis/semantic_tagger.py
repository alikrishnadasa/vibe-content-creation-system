"""
Semantic Video Tagger

Extracts comprehensive semantic tags from video content including objects, people,
environment, and vibe/mood analysis using BLIP-2 and CLIP models.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Set, Optional
import logging
from collections import Counter
import re

from ..models.blip2_captioner import BLIP2Captioner
from ..models.clip_encoder import CLIPVideoEncoder
from ..config import get_config


class SemanticTagger:
    """Extract semantic tags from video content."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self.blip2_captioner = BLIP2Captioner()
        self.clip_encoder = CLIPVideoEncoder()
        
        # Semantic categories with keywords
        self.semantic_categories = {
            "objects": {
                # Kitchen/Food
                "kitchen": ["kitchen", "stove", "oven", "sink", "refrigerator", "microwave", "counter", "cabinet"],
                "food": ["food", "meal", "dish", "cooking", "ingredients", "vegetables", "meat", "bread", "fruit"],
                "cookware": ["pan", "pot", "knife", "spoon", "fork", "plate", "bowl", "cutting", "board"],
                
                # Technology
                "electronics": ["computer", "laptop", "phone", "tablet", "screen", "monitor", "keyboard", "mouse"],
                "devices": ["camera", "microphone", "speaker", "headphones", "remote", "controller"],
                
                # Furniture/Home
                "furniture": ["chair", "table", "sofa", "couch", "bed", "desk", "shelf", "cabinet"],
                "home": ["house", "room", "bedroom", "living", "bathroom", "window", "door", "wall"],
                
                # Transportation
                "vehicles": ["car", "truck", "bus", "bike", "bicycle", "motorcycle", "train", "airplane"],
                
                # Nature
                "nature": ["tree", "flower", "plant", "grass", "rock", "mountain", "river", "lake"],
                "animals": ["dog", "cat", "bird", "horse", "cow", "fish", "rabbit", "deer"],
                
                # Sports/Fitness
                "sports": ["ball", "equipment", "weights", "dumbbells", "mat", "treadmill", "bike"],
                "fitness": ["gym", "workout", "exercise", "training", "fitness", "health"],
                
                # Tools/Equipment
                "tools": ["hammer", "screwdriver", "drill", "saw", "wrench", "tool", "equipment"],
                "office": ["document", "paper", "pen", "pencil", "notebook", "file", "folder"]
            },
            
            "people": {
                "roles": ["chef", "teacher", "student", "doctor", "nurse", "worker", "employee", "manager"],
                "age_groups": ["child", "kid", "teenager", "adult", "elderly", "senior", "young", "old"],
                "gender": ["man", "woman", "male", "female", "boy", "girl"],
                "groups": ["family", "couple", "team", "group", "crowd", "audience", "class"],
                "actions": ["person", "people", "someone", "individual", "human", "figure"]
            },
            
            "environments": {
                "indoor": ["indoor", "inside", "interior", "room", "building", "house", "office"],
                "outdoor": ["outdoor", "outside", "exterior", "street", "park", "garden", "yard"],
                "locations": {
                    "home": ["home", "house", "apartment", "residence", "domestic"],
                    "work": ["office", "workplace", "business", "corporate", "professional"],
                    "education": ["school", "classroom", "university", "college", "library", "educational"],
                    "medical": ["hospital", "clinic", "medical", "healthcare", "doctor", "dental"],
                    "retail": ["store", "shop", "market", "mall", "shopping", "restaurant", "cafe"],
                    "recreation": ["park", "playground", "gym", "sports", "entertainment", "leisure"],
                    "nature": ["forest", "beach", "mountain", "countryside", "wilderness", "natural"],
                    "urban": ["city", "downtown", "urban", "street", "sidewalk", "building", "metropolitan"]
                },
                "settings": ["kitchen", "bedroom", "living room", "bathroom", "garage", "basement", "attic"]
            },
            
            "vibes": {
                "mood": {
                    "positive": ["happy", "joyful", "cheerful", "excited", "energetic", "upbeat", "bright"],
                    "calm": ["peaceful", "serene", "relaxed", "tranquil", "quiet", "gentle", "soothing"],
                    "serious": ["serious", "focused", "concentrated", "professional", "formal", "business"],
                    "intense": ["intense", "dramatic", "powerful", "strong", "bold", "dynamic"],
                    "playful": ["playful", "fun", "amusing", "entertaining", "lighthearted", "casual"],
                    "romantic": ["romantic", "intimate", "loving", "tender", "warm", "cozy"],
                    "mysterious": ["mysterious", "dark", "moody", "atmospheric", "enigmatic"],
                    "negative": ["sad", "angry", "frustrated", "worried", "anxious", "stressed"]
                },
                "energy": {
                    "high": ["energetic", "active", "fast", "quick", "rapid", "dynamic", "lively"],
                    "moderate": ["steady", "normal", "regular", "moderate", "balanced"],
                    "low": ["slow", "calm", "gentle", "relaxed", "quiet", "still", "peaceful"]
                },
                "atmosphere": {
                    "bright": ["bright", "sunny", "cheerful", "light", "illuminated", "vibrant"],
                    "warm": ["warm", "cozy", "comfortable", "inviting", "friendly", "welcoming"],
                    "cool": ["cool", "fresh", "crisp", "clean", "modern", "sleek"],
                    "dark": ["dark", "dim", "shadowy", "moody", "dramatic", "noir"]
                }
            },
            
            "activities": {
                "cooking": ["cooking", "preparing", "chopping", "mixing", "baking", "frying", "boiling"],
                "eating": ["eating", "drinking", "tasting", "consuming", "dining", "snacking"],
                "working": ["working", "typing", "writing", "reading", "studying", "computing"],
                "exercising": ["exercising", "running", "walking", "jumping", "stretching", "lifting"],
                "socializing": ["talking", "conversation", "meeting", "discussing", "chatting"],
                "learning": ["learning", "studying", "teaching", "explaining", "demonstrating"],
                "creating": ["creating", "making", "building", "crafting", "designing", "drawing"],
                "relaxing": ["relaxing", "resting", "sitting", "lying", "sleeping", "meditating"]
            }
        }
        
        # Vibe analysis prompts for BLIP-2
        self.vibe_prompts = [
            "What is the overall mood and atmosphere of this scene?",
            "Describe the emotional tone and feeling of this video.",
            "What kind of energy and vibe does this scene convey?",
            "How would you describe the lighting and visual mood?"
        ]
    
    def analyze_video_semantics(self, frames: np.ndarray, video_duration: float) -> Dict[str, Any]:
        """
        Comprehensive semantic analysis of video frames.
        
        Args:
            frames: Video frames array
            video_duration: Duration of video segment
            
        Returns:
            Dictionary with semantic tags and analysis
        """
        if len(frames) == 0:
            return self._empty_analysis()
        
        try:
            # Generate captions using BLIP-2
            captions = self.blip2_captioner.generate_frame_captions(frames)
            overall_description = self.blip2_captioner.generate_video_description(frames)
            
            # Get visual analysis
            visual_analysis = self.blip2_captioner.analyze_visual_content(frames)
            
            # Generate vibe analysis
            vibe_analysis = self._analyze_vibe(frames)
            
            # Extract semantic tags from all text sources
            all_text = " ".join(captions + [overall_description])
            semantic_tags = self._extract_semantic_tags(all_text, visual_analysis)
            
            # Get CLIP embeddings for similarity matching
            clip_embedding = self.clip_encoder.get_video_embedding(frames)
            
            return {
                "semantic_tags": semantic_tags,
                "descriptions": {
                    "overall": overall_description,
                    "frame_captions": captions,
                    "visual_elements": visual_analysis.get("visual_elements", {})
                },
                "vibe_analysis": vibe_analysis,
                "technical_info": {
                    "frame_count": len(frames),
                    "duration": video_duration,
                    "analysis_method": "BLIP-2 + CLIP",
                    "embedding_dim": clip_embedding.shape[0] if clip_embedding is not None else 0
                },
                "confidence_scores": self._calculate_tag_confidence(semantic_tags, all_text)
            }
            
        except Exception as e:
            self.logger.error(f"Error in semantic analysis: {str(e)}")
            return self._empty_analysis()
    
    def _analyze_vibe(self, frames: np.ndarray) -> Dict[str, Any]:
        """Analyze the emotional vibe and atmosphere of the video."""
        try:
            vibe_descriptions = []
            
            # Use different prompts to get various aspects of the vibe
            for prompt in self.vibe_prompts:
                vibe_desc = self.blip2_captioner.generate_conditional_caption(frames, prompt)
                if vibe_desc and vibe_desc.strip():
                    vibe_descriptions.append(vibe_desc.strip())
            
            # Analyze vibe text for mood indicators
            vibe_text = " ".join(vibe_descriptions).lower()
            vibe_tags = self._extract_vibe_tags(vibe_text)
            
            return {
                "descriptions": vibe_descriptions,
                "detected_vibes": vibe_tags,
                "overall_mood": self._determine_overall_mood(vibe_tags),
                "energy_level": self._determine_energy_level(vibe_tags),
                "atmosphere": self._determine_atmosphere(vibe_tags)
            }
            
        except Exception as e:
            self.logger.error(f"Error in vibe analysis: {str(e)}")
            return {
                "descriptions": [],
                "detected_vibes": {},
                "overall_mood": "neutral",
                "energy_level": "moderate",
                "atmosphere": "neutral"
            }
    
    def _extract_semantic_tags(self, text: str, visual_analysis: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract semantic tags from text and visual analysis."""
        text_lower = text.lower()
        semantic_tags = {
            "objects": [],
            "people": [],
            "environments": [],
            "activities": [],
            "specific_items": []
        }
        
        # Extract from visual analysis if available
        if "visual_elements" in visual_analysis:
            elements = visual_analysis["visual_elements"]
            semantic_tags["objects"].extend(elements.get("objects", []))
            semantic_tags["environments"].extend(elements.get("locations", []))
            semantic_tags["activities"].extend(elements.get("actions", []))
        
        # Extract objects
        for category, keywords in self.semantic_categories["objects"].items():
            found_items = [item for item in keywords if item in text_lower]
            if found_items:
                semantic_tags["objects"].extend(found_items)
                semantic_tags["specific_items"].append(category)
        
        # Extract people-related tags
        for category, keywords in self.semantic_categories["people"].items():
            found_items = [item for item in keywords if item in text_lower]
            semantic_tags["people"].extend(found_items)
        
        # Extract environment tags
        for category, keywords in self.semantic_categories["environments"].items():
            if isinstance(keywords, dict):
                for subcategory, subkeywords in keywords.items():
                    found_items = [item for item in subkeywords if item in text_lower]
                    if found_items:
                        semantic_tags["environments"].extend(found_items)
                        semantic_tags["specific_items"].append(subcategory)
            else:
                found_items = [item for item in keywords if item in text_lower]
                semantic_tags["environments"].extend(found_items)
        
        # Extract activities
        for activity, keywords in self.semantic_categories["activities"].items():
            found_items = [item for item in keywords if item in text_lower]
            if found_items:
                semantic_tags["activities"].extend(found_items)
                semantic_tags["specific_items"].append(activity)
        
        # Remove duplicates and sort
        for key in semantic_tags:
            semantic_tags[key] = sorted(list(set(semantic_tags[key])))
        
        return semantic_tags
    
    def _extract_vibe_tags(self, text: str) -> Dict[str, List[str]]:
        """Extract vibe-related tags from text."""
        vibe_tags = {
            "mood": [],
            "energy": [],
            "atmosphere": []
        }
        
        # Extract mood tags
        for mood_type, keywords in self.semantic_categories["vibes"]["mood"].items():
            found_moods = [keyword for keyword in keywords if keyword in text]
            if found_moods:
                vibe_tags["mood"].extend([mood_type] * len(found_moods))
        
        # Extract energy tags
        for energy_type, keywords in self.semantic_categories["vibes"]["energy"].items():
            found_energy = [keyword for keyword in keywords if keyword in text]
            if found_energy:
                vibe_tags["energy"].extend([energy_type] * len(found_energy))
        
        # Extract atmosphere tags
        for atmosphere_type, keywords in self.semantic_categories["vibes"]["atmosphere"].items():
            found_atmosphere = [keyword for keyword in keywords if keyword in text]
            if found_atmosphere:
                vibe_tags["atmosphere"].extend([atmosphere_type] * len(found_atmosphere))
        
        return vibe_tags
    
    def _determine_overall_mood(self, vibe_tags: Dict[str, List[str]]) -> str:
        """Determine the overall mood from vibe tags."""
        if not vibe_tags.get("mood"):
            return "neutral"
        
        mood_counts = Counter(vibe_tags["mood"])
        most_common_mood = mood_counts.most_common(1)[0][0]
        return most_common_mood
    
    def _determine_energy_level(self, vibe_tags: Dict[str, List[str]]) -> str:
        """Determine the energy level from vibe tags."""
        if not vibe_tags.get("energy"):
            return "moderate"
        
        energy_counts = Counter(vibe_tags["energy"])
        most_common_energy = energy_counts.most_common(1)[0][0]
        return most_common_energy
    
    def _determine_atmosphere(self, vibe_tags: Dict[str, List[str]]) -> str:
        """Determine the atmosphere from vibe tags."""
        if not vibe_tags.get("atmosphere"):
            return "neutral"
        
        atmosphere_counts = Counter(vibe_tags["atmosphere"])
        most_common_atmosphere = atmosphere_counts.most_common(1)[0][0]
        return most_common_atmosphere
    
    def _calculate_tag_confidence(self, semantic_tags: Dict[str, List[str]], source_text: str) -> Dict[str, float]:
        """Calculate confidence scores for extracted tags."""
        confidence_scores = {}
        
        for category, tags in semantic_tags.items():
            if not tags:
                confidence_scores[category] = 0.0
                continue
            
            # Simple confidence based on frequency and text length
            tag_mentions = sum(source_text.lower().count(tag.lower()) for tag in tags)
            text_length = len(source_text.split())
            
            if text_length > 0:
                confidence = min(tag_mentions / text_length * 10, 1.0)  # Normalize to 0-1
            else:
                confidence = 0.0
            
            confidence_scores[category] = round(confidence, 3)
        
        return confidence_scores
    
    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis structure."""
        return {
            "semantic_tags": {
                "objects": [],
                "people": [],
                "environments": [],
                "activities": [],
                "specific_items": []
            },
            "descriptions": {
                "overall": "No visual content detected",
                "frame_captions": [],
                "visual_elements": {}
            },
            "vibe_analysis": {
                "descriptions": [],
                "detected_vibes": {},
                "overall_mood": "neutral",
                "energy_level": "moderate",
                "atmosphere": "neutral"
            },
            "technical_info": {
                "frame_count": 0,
                "duration": 0.0,
                "analysis_method": "No analysis performed",
                "embedding_dim": 0
            },
            "confidence_scores": {}
        }
    
    def get_tag_summary(self, semantic_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get a concise summary of the most important tags."""
        tags = semantic_analysis.get("semantic_tags", {})
        vibe = semantic_analysis.get("vibe_analysis", {})
        
        # Get top tags from each category
        summary = {
            "primary_objects": tags.get("objects", [])[:5],
            "people_present": "yes" if tags.get("people") else "no",
            "environment_type": tags.get("environments", [])[:3],
            "main_activity": tags.get("activities", [])[:3],
            "mood": vibe.get("overall_mood", "neutral"),
            "energy": vibe.get("energy_level", "moderate"),
            "atmosphere": vibe.get("atmosphere", "neutral"),
            "scene_type": self._categorize_scene(tags),
            "complexity": "high" if sum(len(v) for v in tags.values()) > 15 else "moderate" if sum(len(v) for v in tags.values()) > 8 else "simple"
        }
        
        return summary
    
    def _categorize_scene(self, tags: Dict[str, List[str]]) -> str:
        """Categorize the type of scene based on tags."""
        objects = set(tags.get("objects", []))
        environments = set(tags.get("environments", []))
        activities = set(tags.get("activities", []))
        
        # Kitchen/cooking scene
        if any(item in objects for item in ["pan", "knife", "stove", "kitchen"]) or "cooking" in activities:
            return "cooking"
        
        # Workout/fitness scene
        if any(item in objects for item in ["weights", "gym", "mat"]) or "exercising" in activities:
            return "fitness"
        
        # Office/work scene
        if any(item in objects for item in ["computer", "desk", "office"]) or "working" in activities:
            return "office"
        
        # Nature/outdoor scene
        if any(item in environments for item in ["outdoor", "nature", "park"]):
            return "nature"
        
        # Home/domestic scene
        if any(item in environments for item in ["home", "house", "indoor"]):
            return "domestic"
        
        # Education scene
        if any(item in environments for item in ["school", "classroom"]) or "learning" in activities:
            return "educational"
        
        return "general"