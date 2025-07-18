#!/usr/bin/env python3
"""
Enhanced Audio Script Analyzer

Uses actual whisper transcriptions for semantic analysis instead of filename-based guessing.
Implements phrase-level matching, expanded emotions, and context-aware processing.
"""

import os
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np
from dataclasses import dataclass, field
import hashlib
import re
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

@dataclass
class EnhancedScriptAnalysis:
    """Enhanced analysis results for a single audio script."""
    filename: str
    duration: float
    primary_emotion: str
    emotional_intensity: float
    themes: List[str]
    semantic_phrases: List[str]  # NEW: Key phrases from actual content
    content_keywords: List[str]  # NEW: Important keywords from transcript
    emotional_progression: List[Tuple[str, float]]  # NEW: Emotion changes over time
    thematic_categories: Dict[str, float]  # NEW: Weighted theme categories
    context_markers: List[str]  # NEW: Context clues for clip selection
    raw_features: Dict[str, Any] = field(repr=False)

class EnhancedAudioScriptAnalyzer:
    """Enhanced analyzer that uses actual whisper transcriptions for semantic analysis"""
    
    def __init__(self, scripts_directory: str, whisper_cache_directory: str = "cache/whisper"):
        """
        Initialize the enhanced script analyzer
        
        Args:
            scripts_directory: Path to directory containing audio scripts
            whisper_cache_directory: Path to whisper transcription cache
        """
        self.scripts_directory = Path(scripts_directory)
        self.whisper_cache_directory = Path(whisper_cache_directory)
        self.analyses: Dict[str, EnhancedScriptAnalysis] = {}
        self.loaded = False
        
        # EXPANDED EMOTIONAL CATEGORIES (from 5 to 12)
        self.emotion_mapping = {
            'anxiety': {
                'primary_emotion': 'anxiety',
                'intensity': 0.8,
                'keywords': ['anxiety', 'anxious', 'worried', 'tense', 'stressed', 'scared', 'fear', 'terrifying', 'panic'],
                'themes': ['mental_health', 'struggle', 'inner_conflict', 'fear_based_living']
            },
            'seeking': {
                'primary_emotion': 'seeking',
                'intensity': 0.7,
                'keywords': ['searching', 'looking', 'seeking', 'finding', 'journey', 'path', 'quest', 'purpose'],
                'themes': ['spiritual_search', 'purpose_seeking', 'life_direction', 'growth']
            },
            'peace': {
                'primary_emotion': 'peace',
                'intensity': 0.6,
                'keywords': ['peace', 'peaceful', 'calm', 'tranquil', 'serene', 'quiet', 'stillness'],
                'themes': ['inner_peace', 'tranquility', 'meditation', 'serenity']
            },
            'awakening': {
                'primary_emotion': 'awakening',
                'intensity': 0.8,
                'keywords': ['awakening', 'realization', 'enlightenment', 'understanding', 'clarity', 'insight'],
                'themes': ['spiritual_awakening', 'consciousness', 'enlightenment', 'transformation']
            },
            'contemplative': {  # NEW
                'primary_emotion': 'contemplative',
                'intensity': 0.6,
                'keywords': ['contemplative', 'thinking', 'reflection', 'introspective', 'pondering', 'meditation'],
                'themes': ['deep_thought', 'self_reflection', 'philosophical', 'introspection']
            },
            'transformative': {  # NEW
                'primary_emotion': 'transformative',
                'intensity': 0.8,
                'keywords': ['transformation', 'change', 'becoming', 'evolution', 'growth', 'metamorphosis'],
                'themes': ['personal_growth', 'life_change', 'spiritual_development', 'evolution']
            },
            'communal': {  # NEW
                'primary_emotion': 'communal',
                'intensity': 0.7,
                'keywords': ['community', 'together', 'shared', 'collective', 'fellowship', 'brotherhood'],
                'themes': ['community', 'togetherness', 'shared_experience', 'belonging']
            },
            'transcendent': {  # NEW
                'primary_emotion': 'transcendent',
                'intensity': 0.9,
                'keywords': ['transcendent', 'beyond', 'divine', 'eternal', 'infinite', 'unlimited', 'soul'],
                'themes': ['transcendence', 'divinity', 'eternal_perspective', 'soul_consciousness']
            },
            'grounding': {  # NEW
                'primary_emotion': 'grounding',
                'intensity': 0.5,
                'keywords': ['grounded', 'foundation', 'stable', 'rooted', 'secure', 'anchored'],
                'themes': ['stability', 'foundation', 'security', 'rootedness']
            },
            'struggle': {  # NEW (extracted from anxiety but distinct)
                'primary_emotion': 'struggle',
                'intensity': 0.8,
                'keywords': ['struggle', 'difficulty', 'challenge', 'hardship', 'obstacle', 'problem'],
                'themes': ['life_challenges', 'obstacles', 'difficulty', 'overcoming']
            },
            'liberation': {  # NEW
                'primary_emotion': 'liberation',
                'intensity': 0.8,
                'keywords': ['freedom', 'liberation', 'free', 'release', 'escape', 'breakthrough'],
                'themes': ['freedom', 'liberation', 'breaking_free', 'independence']
            },
            'devotional': {  # NEW
                'primary_emotion': 'devotional',
                'intensity': 0.7,
                'keywords': ['devotion', 'love', 'worship', 'surrender', 'service', 'krishna', 'god'],
                'themes': ['devotion', 'spiritual_love', 'worship', 'surrender_to_divine']
            }
        }
        
        # ENHANCED PHRASE-LEVEL PATTERNS
        self.semantic_phrase_patterns = {
            'spiritual_identity': [
                r'eternal soul', r'spiritual being', r'divine nature', r'true self',
                r'soul consciousness', r'spiritual identity', r'who you actually are'
            ],
            'divine_connection': [
                r'love.*krishna', r'serve.*krishna', r'krishna consciousness', 
                r'relationship.*god', r'connection.*divine', r'spiritual relationship'
            ],
            'material_illusion': [
                r'meaningless universe', r'random.*body', r'material existence',
                r'temporary things', r'illusion.*happiness', r'external.*stuff'
            ],
            'inner_transformation': [
                r'inner.*change', r'spiritual.*growth', r'consciousness.*shift',
                r'awakening.*process', r'transformation.*journey', r'spiritual.*evolution'
            ],
            'life_purpose': [
                r'human.*life.*purpose', r'meant to.*serve', r'spiritual.*realization',
                r'highest.*perfection', r'life.*meaning', r'eternal.*purpose'
            ],
            'modern_struggles': [
                r'phone.*addiction', r'digital.*detox', r'comfort zone', r'fear.*failure',
                r'social.*media', r'technology.*problems', r'modern.*life'
            ]
        }
        
        # THEMATIC CATEGORIES with weights
        self.thematic_categories = {
            'spiritual_philosophy': ['soul', 'consciousness', 'divine', 'eternal', 'spiritual', 'transcendent'],
            'practical_spirituality': ['meditation', 'chanting', 'service', 'practice', 'discipline'],
            'psychological_states': ['anxiety', 'peace', 'fear', 'love', 'joy', 'suffering'],
            'life_guidance': ['purpose', 'meaning', 'direction', 'path', 'journey', 'growth'],
            'modern_challenges': ['technology', 'phone', 'comfort', 'safety', 'approval', 'materialism'],
            'relationships': ['community', 'friendship', 'connection', 'isolation', 'belonging'],
            'transformation': ['change', 'growth', 'evolution', 'becoming', 'development']
        }
    
    async def analyze_scripts(self) -> bool:
        """Analyze all audio scripts using whisper transcriptions"""
        try:
            if not self.scripts_directory.exists():
                logger.error(f"Scripts directory not found: {self.scripts_directory}")
                return False
            
            # Find all WAV files
            wav_files = list(self.scripts_directory.glob("*.wav"))
            logger.info(f"Found {len(wav_files)} audio scripts to analyze")
            
            analyzed_count = 0
            for wav_file in wav_files:
                analysis = await self._analyze_single_script_enhanced(wav_file)
                if analysis:
                    self.analyses[wav_file.stem] = analysis
                    analyzed_count += 1
            
            self.loaded = True
            logger.info(f"✅ Successfully analyzed {analyzed_count} audio scripts with enhanced semantic analysis")
            return True
            
        except Exception as e:
            logger.error(f"Failed to analyze scripts: {e}")
            return False
    
    async def _analyze_single_script_enhanced(self, wav_file: Path) -> Optional[EnhancedScriptAnalysis]:
        """Analyze a single audio script using whisper transcription"""
        try:
            # Load whisper transcription
            whisper_cache_path = self.whisper_cache_directory / f"{wav_file.stem}_base.json"
            
            if not whisper_cache_path.exists():
                logger.warning(f"No whisper transcription found for {wav_file.stem}, falling back to filename analysis")
                return await self._fallback_filename_analysis(wav_file)
            
            with open(whisper_cache_path, 'r') as f:
                whisper_data = json.load(f)
            
            transcript = whisper_data.get('text', '')
            duration = whisper_data.get('duration', 0.0)
            
            if not transcript:
                logger.warning(f"Empty transcript for {wav_file.stem}, falling back to filename analysis")
                return await self._fallback_filename_analysis(wav_file)
            
            # ENHANCED SEMANTIC ANALYSIS
            semantic_analysis = self._analyze_transcript_semantics(transcript)
            
            # Determine emotions from actual content
            emotions = self._determine_emotions_from_content(transcript, semantic_analysis)
            primary_emotion = emotions[0][0] if emotions else 'neutral'
            emotional_intensity = emotions[0][1] if emotions else 0.5
            
            # Extract themes from content
            themes = self._extract_themes_from_content(transcript, semantic_analysis)
            
            # Generate context markers for clip selection
            context_markers = self._generate_context_markers(transcript, semantic_analysis)
            
            analysis = EnhancedScriptAnalysis(
                filename=wav_file.name,
                duration=duration,
                primary_emotion=primary_emotion,
                emotional_intensity=emotional_intensity,
                themes=themes,
                semantic_phrases=semantic_analysis['key_phrases'],
                content_keywords=semantic_analysis['keywords'],
                emotional_progression=emotions,
                thematic_categories=semantic_analysis['thematic_weights'],
                context_markers=context_markers,
                raw_features={
                    'filepath': str(wav_file),
                    'transcript': transcript,
                    'whisper_confidence': self._calculate_transcript_confidence(whisper_data),
                    'semantic_analysis': semantic_analysis
                }
            )
            
            logger.debug(f"Enhanced analysis for {wav_file.name}: {primary_emotion} (intensity: {emotional_intensity:.2f})")
            logger.debug(f"Key themes: {', '.join(themes[:3])}")
            logger.debug(f"Key phrases: {', '.join(semantic_analysis['key_phrases'][:2])}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze {wav_file}: {e}")
            return None
    
    def _analyze_transcript_semantics(self, transcript: str) -> Dict[str, Any]:
        """Perform deep semantic analysis of transcript"""
        words = transcript.lower().split()
        word_count = Counter(words)
        
        # Extract key phrases using patterns
        key_phrases = []
        for category, patterns in self.semantic_phrase_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, transcript.lower())
                key_phrases.extend([f"{category}:{match}" for match in matches])
        
        # Extract important keywords (filter out common words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'}
        
        keywords = [word for word, count in word_count.most_common(20) 
                   if len(word) > 3 and word not in stop_words]
        
        # Calculate thematic category weights
        thematic_weights = {}
        for category, category_words in self.thematic_categories.items():
            weight = sum(word_count.get(word, 0) for word in category_words)
            weight += sum(len(re.findall(word, transcript.lower())) for word in category_words) * 0.5
            thematic_weights[category] = weight / len(words) if words else 0
        
        return {
            'key_phrases': key_phrases[:10],  # Top 10 key phrases
            'keywords': keywords[:15],  # Top 15 keywords
            'thematic_weights': thematic_weights,
            'word_count': len(words),
            'unique_words': len(word_count)
        }
    
    def _determine_emotions_from_content(self, transcript: str, semantic_analysis: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Determine emotions from actual transcript content with confidence scores"""
        transcript_lower = transcript.lower()
        emotion_scores = []
        
        for emotion, config in self.emotion_mapping.items():
            score = 0.0
            
            # Keyword matching with frequency weighting
            for keyword in config['keywords']:
                # Count occurrences and weight by frequency
                occurrences = len(re.findall(r'\b' + re.escape(keyword) + r'\b', transcript_lower))
                score += occurrences * 2.0
            
            # Theme matching
            for theme in config['themes']:
                theme_words = theme.replace('_', ' ').split()
                for word in theme_words:
                    if word in transcript_lower:
                        score += 1.0
            
            # Phrase pattern bonus
            phrase_bonus = sum(1.5 for phrase in semantic_analysis['key_phrases'] 
                             if any(theme in phrase for theme in config['themes']))
            score += phrase_bonus
            
            # Thematic category alignment
            relevant_categories = ['spiritual_philosophy', 'psychological_states', 'life_guidance']
            for category in relevant_categories:
                if category in semantic_analysis['thematic_weights']:
                    score += semantic_analysis['thematic_weights'][category] * 10
            
            # Normalize score by transcript length
            normalized_score = score / max(len(transcript_lower.split()), 1)
            
            if normalized_score > 0:
                emotion_scores.append((emotion, min(normalized_score, 1.0)))
        
        # Sort by score and return top emotions
        emotion_scores.sort(key=lambda x: x[1], reverse=True)
        return emotion_scores[:3] if emotion_scores else [('neutral', 0.5)]
    
    def _extract_themes_from_content(self, transcript: str, semantic_analysis: Dict[str, Any]) -> List[str]:
        """Extract themes based on content analysis"""
        themes = set()
        
        # Add themes from highest-weighted categories
        sorted_categories = sorted(semantic_analysis['thematic_weights'].items(), 
                                 key=lambda x: x[1], reverse=True)
        
        for category, weight in sorted_categories[:4]:  # Top 4 categories
            if weight > 0.01:  # Minimum threshold
                themes.add(category)
        
        # Add specific themes based on key phrases
        for phrase in semantic_analysis['key_phrases']:
            if ':' in phrase:
                category = phrase.split(':')[0]
                themes.add(category)
        
        # Add themes based on high-frequency meaningful keywords
        meaningful_keywords = semantic_analysis['keywords'][:5]
        for keyword in meaningful_keywords:
            if len(keyword) > 5:  # Longer keywords often represent themes
                themes.add(keyword)
        
        return list(themes)[:8]  # Limit to 8 themes
    
    def _generate_context_markers(self, transcript: str, semantic_analysis: Dict[str, Any]) -> List[str]:
        """Generate context markers for intelligent clip selection"""
        markers = []
        
        # Temporal markers
        if any(word in transcript.lower() for word in ['before', 'after', 'when', 'during', 'while']):
            markers.append('temporal_narrative')
        
        # Emotional progression markers
        if any(word in transcript.lower() for word in ['but', 'however', 'yet', 'although']):
            markers.append('emotional_contrast')
        
        # Action-oriented markers
        if any(word in transcript.lower() for word in ['start', 'begin', 'try', 'practice', 'do']):
            markers.append('actionable_advice')
        
        # Personal experience markers
        if any(phrase in transcript.lower() for phrase in ['i used to', 'when i', 'i became', 'i learned']):
            markers.append('personal_testimony')
        
        # Philosophical markers
        if any(word in transcript.lower() for word in ['consciousness', 'soul', 'divine', 'eternal', 'spiritual']):
            markers.append('philosophical_depth')
        
        # Modern context markers
        if any(word in transcript.lower() for word in ['phone', 'digital', 'technology', 'modern']):
            markers.append('contemporary_relevance')
        
        return markers
    
    def _calculate_transcript_confidence(self, whisper_data: Dict[str, Any]) -> float:
        """Calculate average confidence from whisper word-level data"""
        words = whisper_data.get('words', [])
        if not words:
            return 0.5
        
        confidences = [word.get('confidence', 0.5) for word in words]
        return sum(confidences) / len(confidences)
    
    async def _fallback_filename_analysis(self, wav_file: Path) -> Optional[EnhancedScriptAnalysis]:
        """Fallback to filename-based analysis when whisper data unavailable"""
        logger.info(f"Using fallback filename analysis for {wav_file.stem}")
        
        # Use original filename-based logic
        emotion_info = self._determine_emotion_from_filename(wav_file.stem)
        
        return EnhancedScriptAnalysis(
            filename=wav_file.name,
            duration=0.0,  # Unknown without audio analysis
            primary_emotion=emotion_info['primary_emotion'],
            emotional_intensity=emotion_info['intensity'],
            themes=emotion_info['themes'].copy(),
            semantic_phrases=[],
            content_keywords=[],
            emotional_progression=[(emotion_info['primary_emotion'], emotion_info['intensity'])],
            thematic_categories={},
            context_markers=['filename_based'],
            raw_features={
                'filepath': str(wav_file),
                'analysis_type': 'filename_fallback',
                'emotional_tags': emotion_info['tags'].copy()
            }
        )
    
    def _determine_emotion_from_filename(self, filename_stem: str) -> Dict[str, Any]:
        """Original filename-based emotion determination (fallback)"""
        filename_lower = filename_stem.lower()
        
        # Legacy mappings for fallback
        legacy_mapping = {
            'anxiety': {'primary_emotion': 'anxiety', 'intensity': 0.8, 'tags': ['anxious', 'worried'], 'themes': ['mental_health', 'struggle']},
            'safe': {'primary_emotion': 'peace', 'intensity': 0.7, 'tags': ['secure', 'protected'], 'themes': ['safety', 'security']},
            'phone': {'primary_emotion': 'struggle', 'intensity': 0.6, 'tags': ['distracted', 'addicted'], 'themes': ['technology', 'modern_challenges']},
            'before': {'primary_emotion': 'transformative', 'intensity': 0.7, 'tags': ['transformation', 'change'], 'themes': ['personal_growth', 'transformation']},
        }
        
        for pattern, emotion_info in legacy_mapping.items():
            if pattern in filename_lower or any(f"{pattern}{i}" in filename_lower for i in range(1, 10)):
                return emotion_info
        
        return {'primary_emotion': 'neutral', 'intensity': 0.5, 'tags': ['general'], 'themes': ['general_content']}
    
    def get_script_analysis(self, script_name: str) -> Optional[EnhancedScriptAnalysis]:
        """Get enhanced analysis for a specific script"""
        script_key = script_name.replace('.wav', '')
        return self.analyses.get(script_key)
    
    def get_scripts_by_emotion(self, emotion: str) -> List[EnhancedScriptAnalysis]:
        """Get scripts that match a specific emotional category"""
        if not self.loaded:
            raise RuntimeError("Scripts not analyzed. Call analyze_scripts() first.")
        
        return [analysis for analysis in self.analyses.values() 
                if analysis.primary_emotion == emotion.lower()]
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about analyzed scripts"""
        if not self.loaded:
            raise RuntimeError("Scripts not analyzed. Call analyze_scripts() first.")
        
        total_scripts = len(self.analyses)
        
        # Emotional distribution
        emotion_counts = Counter(analysis.primary_emotion for analysis in self.analyses.values())
        
        # Theme distribution
        theme_counts = Counter()
        for analysis in self.analyses.values():
            theme_counts.update(analysis.themes)
        
        # Semantic phrase analysis
        phrase_counts = Counter()
        for analysis in self.analyses.values():
            phrase_counts.update(analysis.semantic_phrases)
        
        # Context marker distribution
        context_counts = Counter()
        for analysis in self.analyses.values():
            context_counts.update(analysis.context_markers)
        
        return {
            'total_scripts': total_scripts,
            'emotion_distribution': dict(emotion_counts),
            'theme_distribution': dict(theme_counts.most_common(15)),
            'semantic_phrases': dict(phrase_counts.most_common(10)),
            'context_markers': dict(context_counts),
            'enhanced_analysis_count': sum(1 for a in self.analyses.values() if a.semantic_phrases),
            'fallback_analysis_count': sum(1 for a in self.analyses.values() if not a.semantic_phrases)
        }