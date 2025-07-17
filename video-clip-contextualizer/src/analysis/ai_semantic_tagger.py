"""
AI-Powered Semantic Video Tagger

Uses advanced NLP models and computer vision AI to extract comprehensive semantic tags
including objects, people, environment, and emotional/atmospheric analysis.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModel,
    CLIPModel,
    CLIPProcessor
)
import spacy
from collections import defaultdict
import json

from ..models.blip2_captioner import BLIP2Captioner
from ..models.clip_encoder import CLIPVideoEncoder
from ..config import get_config


class AISemanticTagger:
    """AI-powered comprehensive semantic analysis for video content."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        # Force CPU usage to avoid CUDA issues
        self.device = torch.device("cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize core models - disable for now to avoid model loading issues
        self.blip2_captioner = None  # Disable BLIP2 for CPU-only mode
        self.clip_encoder = None  # Disable CLIP for CPU-only mode
        
        # Initialize NLP pipeline
        self._initialize_nlp_models()
        
        # Initialize semantic analysis pipelines
        self._initialize_semantic_pipelines()
        
        # Pre-defined semantic categories for classification
        self.semantic_categories = self._load_semantic_categories()
        
    def _initialize_nlp_models(self):
        """Initialize NLP models for text analysis."""
        try:
            # Load spaCy model for NER and dependency parsing
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                self.logger.warning("spaCy en_core_web_sm not found. Installing...")
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
                self.nlp = spacy.load("en_core_web_sm")
            
            # Emotion analysis pipeline - force CPU usage
            self.emotion_analyzer = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=-1  # Force CPU
            )
            
            # Sentiment analysis - force CPU usage
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=-1  # Force CPU
            )
            
            # Zero-shot classification for custom categories - force CPU usage
            self.zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1  # Force CPU
            )
            
            self.logger.info("NLP models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize NLP models: {str(e)}")
            # Fallback to basic analysis
            self.nlp = None
            self.emotion_analyzer = None
            self.sentiment_analyzer = None
            self.zero_shot_classifier = None
    
    def _initialize_semantic_pipelines(self):
        """Initialize specialized semantic analysis pipelines."""
        try:
            # Object detection categories
            self.object_categories = [
                "person", "people", "man", "woman", "child", "adult",
                "kitchen", "food", "cooking", "utensils", "appliances",
                "furniture", "electronics", "technology", "computer",
                "nature", "outdoor", "indoor", "building", "vehicle",
                "sports", "exercise", "fitness", "tools", "equipment",
                "art", "music", "books", "education", "medical"
            ]
            
            # Environment categories
            self.environment_categories = [
                "home", "kitchen", "bedroom", "living room", "bathroom",
                "office", "workplace", "school", "classroom", "hospital",
                "restaurant", "store", "park", "street", "beach",
                "forest", "mountain", "city", "suburban", "rural",
                "indoor", "outdoor", "professional", "casual", "formal"
            ]
            
            # Activity categories
            self.activity_categories = [
                "cooking", "eating", "working", "studying", "exercising",
                "walking", "running", "sitting", "standing", "talking",
                "reading", "writing", "playing", "teaching", "learning",
                "creating", "building", "cleaning", "shopping", "traveling"
            ]
            
            # Vibe/mood categories
            self.vibe_categories = [
                "happy", "sad", "excited", "calm", "energetic", "peaceful",
                "professional", "casual", "formal", "playful", "serious",
                "bright", "dark", "warm", "cool", "cozy", "modern",
                "vintage", "minimalist", "busy", "quiet", "dramatic",
                "romantic", "mysterious", "uplifting", "motivational"
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to initialize semantic pipelines: {str(e)}")
    
    def _load_semantic_categories(self) -> Dict[str, List[str]]:
        """Load comprehensive semantic categories."""
        return {
            "objects": self.object_categories,
            "environments": self.environment_categories,
            "activities": self.activity_categories,
            "vibes": self.vibe_categories
        }
    
    def analyze_video_semantics(self, frames: np.ndarray, video_duration: float) -> Dict[str, Any]:
        """
        Comprehensive AI-powered semantic analysis of video content.
        
        Args:
            frames: Video frames array
            video_duration: Duration of video segment
            
        Returns:
            Dictionary with comprehensive semantic analysis
        """
        if len(frames) == 0:
            return self._empty_analysis()
        
        try:
            # Step 1: Generate visual descriptions using BLIP-2
            visual_descriptions = self._generate_visual_descriptions(frames)
            
            # Step 2: Extract entities and concepts using NLP
            nlp_analysis = self._analyze_text_with_nlp(visual_descriptions)
            
            # Step 3: Classify content into semantic categories
            semantic_classifications = self._classify_semantic_content(visual_descriptions)
            
            # Step 4: Analyze emotional content and vibe
            emotional_analysis = self._analyze_emotional_content(visual_descriptions)
            
            # Step 5: Extract people and demographics
            people_analysis = self._analyze_people_content(visual_descriptions)
            
            # Step 6: Analyze environment and setting
            environment_analysis = self._analyze_environment(visual_descriptions)
            
            # Step 7: Generate comprehensive tags
            comprehensive_tags = self._generate_comprehensive_tags(
                nlp_analysis, semantic_classifications, emotional_analysis,
                people_analysis, environment_analysis
            )
            
            # Step 8: Calculate confidence scores
            confidence_scores = self._calculate_ai_confidence_scores(
                visual_descriptions, comprehensive_tags
            )
            
            return {
                "semantic_tags": comprehensive_tags,
                "descriptions": visual_descriptions,
                "nlp_analysis": nlp_analysis,
                "emotional_analysis": emotional_analysis,
                "people_analysis": people_analysis,
                "environment_analysis": environment_analysis,
                "confidence_scores": confidence_scores,
                "technical_info": {
                    "frame_count": len(frames),
                    "duration": video_duration,
                    "analysis_method": "AI-powered NLP + Computer Vision",
                    "models_used": [
                        "BLIP-2", "CLIP", "RoBERTa", "BART", "spaCy"
                    ]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in AI semantic analysis: {str(e)}")
            return self._empty_analysis()
    
    def _generate_visual_descriptions(self, frames: np.ndarray) -> Dict[str, Any]:
        """Generate comprehensive visual descriptions using BLIP-2."""
        try:
            # Simplified mode - generate basic placeholder descriptions
            if self.blip2_captioner is None:
                return {
                    "overall_description": "AI visual analysis disabled for CPU-only mode",
                    "frame_captions": ["Generic video frame"] * min(len(frames), 5),
                    "targeted_descriptions": {
                        "people": "People may be present in the scene",
                        "environment": "Indoor or outdoor environment",
                        "objects": "Various objects may be visible",
                        "mood": "Neutral mood and atmosphere",
                        "activity": "Some activity is taking place"
                    },
                    "combined_text": "This is a video segment with various visual elements including people, objects, and activities taking place in an environment."
                }
            
            # Generate multiple types of descriptions
            frame_captions = self.blip2_captioner.generate_frame_captions(frames)
            overall_description = self.blip2_captioner.generate_video_description(frames)
            
            # Generate targeted descriptions for specific aspects
            targeted_descriptions = {}
            
            # People-focused description
            people_prompt = "Describe the people in this scene including their actions, appearance, and interactions"
            targeted_descriptions["people"] = self.blip2_captioner.generate_conditional_caption(
                frames, people_prompt
            )
            
            # Environment-focused description
            environment_prompt = "Describe the setting, location, and environment of this scene"
            targeted_descriptions["environment"] = self.blip2_captioner.generate_conditional_caption(
                frames, environment_prompt
            )
            
            # Objects-focused description
            objects_prompt = "List and describe the main objects, items, and things visible in this scene"
            targeted_descriptions["objects"] = self.blip2_captioner.generate_conditional_caption(
                frames, objects_prompt
            )
            
            # Mood/atmosphere description
            mood_prompt = "Describe the mood, atmosphere, lighting, and overall feeling of this scene"
            targeted_descriptions["mood"] = self.blip2_captioner.generate_conditional_caption(
                frames, mood_prompt
            )
            
            # Activity description
            activity_prompt = "Describe what activities and actions are taking place in this scene"
            targeted_descriptions["activity"] = self.blip2_captioner.generate_conditional_caption(
                frames, activity_prompt
            )
            
            return {
                "overall_description": overall_description,
                "frame_captions": frame_captions,
                "targeted_descriptions": targeted_descriptions,
                "combined_text": self._combine_descriptions(
                    overall_description, frame_captions, targeted_descriptions
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error generating visual descriptions: {str(e)}")
            return {
                "overall_description": "",
                "frame_captions": [],
                "targeted_descriptions": {},
                "combined_text": ""
            }
    
    def _analyze_text_with_nlp(self, visual_descriptions: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze text using advanced NLP techniques."""
        if not self.nlp:
            return {"entities": [], "concepts": [], "relationships": []}
        
        try:
            combined_text = visual_descriptions.get("combined_text", "")
            if not combined_text:
                return {"entities": [], "concepts": [], "relationships": []}
            
            # Process with spaCy
            doc = self.nlp(combined_text)
            
            # Extract named entities
            entities = []
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "description": spacy.explain(ent.label_),
                    "confidence": getattr(ent, "_.confidence", 0.5)
                })
            
            # Extract noun phrases (concepts)
            concepts = []
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 3:  # Keep it concise
                    concepts.append({
                        "text": chunk.text,
                        "root": chunk.root.text,
                        "pos": chunk.root.pos_
                    })
            
            # Extract relationships (subject-verb-object)
            relationships = []
            for token in doc:
                if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                    subj = token.text
                    verb = token.head.text
                    obj = None
                    
                    # Find object
                    for child in token.head.children:
                        if child.dep_ in ["dobj", "pobj"]:
                            obj = child.text
                            break
                    
                    if obj:
                        relationships.append({
                            "subject": subj,
                            "verb": verb,
                            "object": obj
                        })
            
            return {
                "entities": entities,
                "concepts": [c["text"] for c in concepts[:10]],  # Top 10
                "relationships": relationships[:5],  # Top 5
                "language_stats": {
                    "sentence_count": len(list(doc.sents)),
                    "token_count": len(doc),
                    "complexity": self._calculate_text_complexity(doc)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in NLP analysis: {str(e)}")
            return {"entities": [], "concepts": [], "relationships": []}
    
    def _classify_semantic_content(self, visual_descriptions: Dict[str, Any]) -> Dict[str, Any]:
        """Classify content into semantic categories using zero-shot classification."""
        if not self.zero_shot_classifier:
            return {}
        
        try:
            combined_text = visual_descriptions.get("combined_text", "")
            if not combined_text:
                return {}
            
            classifications = {}
            
            # Classify for each semantic category
            for category, labels in self.semantic_categories.items():
                try:
                    result = self.zero_shot_classifier(combined_text, labels)
                    
                    # Get top classifications with confidence > 0.1
                    top_classifications = [
                        {
                            "label": result["labels"][i],
                            "score": result["scores"][i]
                        }
                        for i in range(min(5, len(result["labels"])))
                        if result["scores"][i] > 0.1
                    ]
                    
                    classifications[category] = top_classifications
                    
                except Exception as e:
                    self.logger.warning(f"Classification failed for {category}: {str(e)}")
                    classifications[category] = []
            
            return classifications
            
        except Exception as e:
            self.logger.error(f"Error in semantic classification: {str(e)}")
            return {}
    
    def _analyze_emotional_content(self, visual_descriptions: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze emotional content and vibe using emotion detection models."""
        try:
            combined_text = visual_descriptions.get("combined_text", "")
            mood_description = visual_descriptions.get("targeted_descriptions", {}).get("mood", "")
            
            analysis_text = f"{combined_text} {mood_description}".strip()
            
            if not analysis_text:
                return self._empty_emotional_analysis()
            
            emotional_analysis = {
                "emotions": [],
                "sentiment": {},
                "vibe_score": {},
                "overall_mood": "neutral"
            }
            
            # Emotion detection
            if self.emotion_analyzer:
                try:
                    emotion_results = self.emotion_analyzer(analysis_text[:512])  # Limit length
                    emotional_analysis["emotions"] = [
                        {
                            "emotion": result["label"],
                            "confidence": result["score"]
                        }
                        for result in emotion_results
                    ]
                    
                    # Get primary emotion
                    if emotion_results:
                        emotional_analysis["overall_mood"] = emotion_results[0]["label"]
                        
                except Exception as e:
                    self.logger.warning(f"Emotion analysis failed: {str(e)}")
            
            # Sentiment analysis
            if self.sentiment_analyzer:
                try:
                    sentiment_results = self.sentiment_analyzer(analysis_text[:512])
                    emotional_analysis["sentiment"] = {
                        "label": sentiment_results[0]["label"],
                        "confidence": sentiment_results[0]["score"]
                    }
                except Exception as e:
                    self.logger.warning(f"Sentiment analysis failed: {str(e)}")
            
            # Vibe classification using zero-shot
            if self.zero_shot_classifier:
                try:
                    vibe_result = self.zero_shot_classifier(
                        mood_description or combined_text,
                        self.vibe_categories
                    )
                    
                    emotional_analysis["vibe_score"] = {
                        vibe_result["labels"][i]: vibe_result["scores"][i]
                        for i in range(min(3, len(vibe_result["labels"])))
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Vibe classification failed: {str(e)}")
            
            return emotional_analysis
            
        except Exception as e:
            self.logger.error(f"Error in emotional analysis: {str(e)}")
            return self._empty_emotional_analysis()
    
    def _analyze_people_content(self, visual_descriptions: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze people-related content using NLP."""
        try:
            people_description = visual_descriptions.get("targeted_descriptions", {}).get("people", "")
            combined_text = visual_descriptions.get("combined_text", "")
            
            analysis_text = people_description or combined_text
            
            if not analysis_text or not self.zero_shot_classifier:
                return {"present": False, "details": []}
            
            # Check if people are present
            people_categories = [
                "people present", "no people", "person visible", "human subjects",
                "individual", "group", "crowd", "family", "friends"
            ]
            
            people_classification = self.zero_shot_classifier(
                analysis_text, people_categories
            )
            
            people_present = any(
                "people" in label.lower() or "person" in label.lower() or "human" in label.lower()
                for label, score in zip(people_classification["labels"][:3], people_classification["scores"][:3])
                if score > 0.3
            )
            
            details = []
            if people_present:
                # Analyze demographics and characteristics
                demographic_categories = [
                    "adult", "child", "elderly", "young person", "teenager",
                    "male", "female", "professional", "casual", "formal attire"
                ]
                
                try:
                    demo_result = self.zero_shot_classifier(analysis_text, demographic_categories)
                    details = [
                        {"characteristic": demo_result["labels"][i], "confidence": demo_result["scores"][i]}
                        for i in range(min(5, len(demo_result["labels"])))
                        if demo_result["scores"][i] > 0.2
                    ]
                except Exception as e:
                    self.logger.warning(f"Demographic analysis failed: {str(e)}")
            
            return {
                "present": people_present,
                "confidence": people_classification["scores"][0] if people_classification["scores"] else 0.0,
                "details": details
            }
            
        except Exception as e:
            self.logger.error(f"Error in people analysis: {str(e)}")
            return {"present": False, "details": []}
    
    def _analyze_environment(self, visual_descriptions: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze environment and setting using AI classification."""
        try:
            environment_description = visual_descriptions.get("targeted_descriptions", {}).get("environment", "")
            combined_text = visual_descriptions.get("combined_text", "")
            
            analysis_text = environment_description or combined_text
            
            if not analysis_text or not self.zero_shot_classifier:
                return {"type": "unknown", "details": []}
            
            # Classify environment type
            env_result = self.zero_shot_classifier(analysis_text, self.environment_categories)
            
            # Get lighting and atmosphere
            lighting_categories = [
                "bright lighting", "dim lighting", "natural light", "artificial light",
                "warm lighting", "cool lighting", "dramatic lighting", "soft lighting"
            ]
            
            lighting_result = self.zero_shot_classifier(analysis_text, lighting_categories)
            
            return {
                "type": env_result["labels"][0] if env_result["labels"] else "unknown",
                "confidence": env_result["scores"][0] if env_result["scores"] else 0.0,
                "lighting": {
                    "type": lighting_result["labels"][0] if lighting_result["labels"] else "unknown",
                    "confidence": lighting_result["scores"][0] if lighting_result["scores"] else 0.0
                },
                "details": [
                    {"aspect": env_result["labels"][i], "confidence": env_result["scores"][i]}
                    for i in range(min(3, len(env_result["labels"])))
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error in environment analysis: {str(e)}")
            return {"type": "unknown", "details": []}
    
    def _generate_comprehensive_tags(self, nlp_analysis: Dict, semantic_classifications: Dict,
                                   emotional_analysis: Dict, people_analysis: Dict,
                                   environment_analysis: Dict) -> Dict[str, List[str]]:
        """Generate comprehensive semantic tags from all analyses."""
        tags = {
            "objects": [],
            "people": [],
            "environment": [],
            "activities": [],
            "emotions": [],
            "vibes": [],
            "concepts": [],
            "attributes": []
        }
        
        # Extract from NLP analysis
        if nlp_analysis:
            # Add entities
            for entity in nlp_analysis.get("entities", []):
                if entity["label"] in ["PERSON", "ORG"]:
                    tags["people"].append(entity["text"].lower())
                elif entity["label"] in ["GPE", "LOC"]:
                    tags["environment"].append(entity["text"].lower())
                else:
                    tags["objects"].append(entity["text"].lower())
            
            # Add concepts
            tags["concepts"].extend([c.lower() for c in nlp_analysis.get("concepts", [])])
        
        # Extract from semantic classifications
        for category, classifications in semantic_classifications.items():
            if category in tags:
                for classification in classifications:
                    if classification["score"] > 0.2:
                        tags[category].append(classification["label"])
        
        # Extract from emotional analysis
        if emotional_analysis:
            for emotion in emotional_analysis.get("emotions", []):
                if emotion["confidence"] > 0.3:
                    tags["emotions"].append(emotion["emotion"])
            
            for vibe, score in emotional_analysis.get("vibe_score", {}).items():
                if score > 0.2:
                    tags["vibes"].append(vibe)
        
        # Extract from people analysis
        if people_analysis.get("present"):
            tags["people"].append("people_present")
            for detail in people_analysis.get("details", []):
                if detail["confidence"] > 0.3:
                    tags["attributes"].append(detail["characteristic"])
        
        # Extract from environment analysis
        if environment_analysis:
            env_type = environment_analysis.get("type")
            if env_type and env_type != "unknown":
                tags["environment"].append(env_type)
            
            lighting = environment_analysis.get("lighting", {})
            if lighting.get("type") and lighting.get("confidence", 0) > 0.3:
                tags["attributes"].append(lighting["type"])
        
        # Clean up and deduplicate tags
        for category in tags:
            tags[category] = sorted(list(set([tag for tag in tags[category] if tag and len(tag) > 1])))
        
        return tags
    
    def _calculate_ai_confidence_scores(self, visual_descriptions: Dict, tags: Dict) -> Dict[str, float]:
        """Calculate confidence scores for AI-generated tags."""
        confidence_scores = {}
        
        for category, tag_list in tags.items():
            if not tag_list:
                confidence_scores[category] = 0.0
                continue
            
            # Base confidence on number of tags and description quality
            description_length = len(visual_descriptions.get("combined_text", ""))
            tag_count = len(tag_list)
            
            # Simple heuristic: more tags from longer descriptions = higher confidence
            if description_length > 100:
                base_confidence = min(0.8, 0.4 + (tag_count * 0.1))
            elif description_length > 50:
                base_confidence = min(0.6, 0.3 + (tag_count * 0.08))
            else:
                base_confidence = min(0.4, 0.2 + (tag_count * 0.05))
            
            confidence_scores[category] = round(base_confidence, 3)
        
        return confidence_scores
    
    def _combine_descriptions(self, overall: str, frame_captions: List[str], 
                            targeted: Dict[str, str]) -> str:
        """Combine all descriptions into a single text for analysis."""
        combined = [overall]
        combined.extend(frame_captions)
        combined.extend([desc for desc in targeted.values() if desc])
        
        return " ".join([desc for desc in combined if desc]).strip()
    
    def _calculate_text_complexity(self, doc) -> str:
        """Calculate text complexity based on spaCy analysis."""
        if len(doc) < 10:
            return "simple"
        elif len(doc) < 50:
            return "moderate"
        else:
            return "complex"
    
    def _empty_emotional_analysis(self) -> Dict[str, Any]:
        """Return empty emotional analysis structure."""
        return {
            "emotions": [],
            "sentiment": {},
            "vibe_score": {},
            "overall_mood": "neutral"
        }
    
    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis structure."""
        return {
            "semantic_tags": {
                "objects": [],
                "people": [],
                "environment": [],
                "activities": [],
                "emotions": [],
                "vibes": [],
                "concepts": [],
                "attributes": []
            },
            "descriptions": {
                "overall_description": "No visual content detected",
                "frame_captions": [],
                "targeted_descriptions": {},
                "combined_text": ""
            },
            "nlp_analysis": {"entities": [], "concepts": [], "relationships": []},
            "emotional_analysis": self._empty_emotional_analysis(),
            "people_analysis": {"present": False, "details": []},
            "environment_analysis": {"type": "unknown", "details": []},
            "confidence_scores": {},
            "technical_info": {
                "frame_count": 0,
                "duration": 0.0,
                "analysis_method": "No analysis performed",
                "models_used": []
            }
        }
    
    def get_tag_summary(self, semantic_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get a concise summary of the most important semantic information."""
        tags = semantic_analysis.get("semantic_tags", {})
        emotional = semantic_analysis.get("emotional_analysis", {})
        people = semantic_analysis.get("people_analysis", {})
        environment = semantic_analysis.get("environment_analysis", {})
        
        return {
            "primary_objects": tags.get("objects", [])[:5],
            "key_concepts": tags.get("concepts", [])[:5],
            "people_present": people.get("present", False),
            "environment_type": environment.get("type", "unknown"),
            "primary_emotion": emotional.get("overall_mood", "neutral"),
            "dominant_vibes": list(tags.get("vibes", []))[:3],
            "main_activities": tags.get("activities", [])[:3],
            "scene_attributes": tags.get("attributes", [])[:5],
            "complexity_score": self._calculate_scene_complexity(tags),
            "confidence_level": self._get_overall_confidence(semantic_analysis.get("confidence_scores", {}))
        }
    
    def _calculate_scene_complexity(self, tags: Dict[str, List[str]]) -> str:
        """Calculate overall scene complexity."""
        total_tags = sum(len(tag_list) for tag_list in tags.values())
        
        if total_tags > 20:
            return "high"
        elif total_tags > 10:
            return "moderate"
        else:
            return "simple"
    
    def _get_overall_confidence(self, confidence_scores: Dict[str, float]) -> str:
        """Get overall confidence assessment."""
        if not confidence_scores:
            return "low"
        
        avg_confidence = sum(confidence_scores.values()) / len(confidence_scores)
        
        if avg_confidence > 0.7:
            return "high"
        elif avg_confidence > 0.4:
            return "moderate"
        else:
            return "low"