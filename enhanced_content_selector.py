#!/usr/bin/env python3
"""
Enhanced Content Selector

Implements phrase-level semantic matching, context-aware clip selection,
and dynamic learning from successful combinations.
"""

import logging
import asyncio
import json
import time
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
import random
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
import re

from enhanced_script_analyzer import EnhancedScriptAnalysis

logger = logging.getLogger(__name__)

@dataclass
class EnhancedSelectionCriteria:
    """Enhanced criteria for intelligent clip selection"""
    emotion: str
    intensity: float
    themes: List[str]
    duration_target: float
    music_beats: List[float]
    
    # NEW ENHANCED FIELDS
    semantic_phrases: List[str] = field(default_factory=list)
    content_keywords: List[str] = field(default_factory=list)
    thematic_weights: Dict[str, float] = field(default_factory=dict)
    context_markers: List[str] = field(default_factory=list)
    
    # Original fields
    visual_variety: bool = True
    lighting_preference: Optional[str] = None
    movement_preference: Optional[str] = None
    sync_event_type: str = 'beat'
    use_percussive_sync: bool = False

@dataclass
class ContextualSequence:
    """Enhanced sequence with contextual flow analysis"""
    clips: List[Any]  # ClipMetadata objects
    total_duration: float
    relevance_score: float
    visual_variety_score: float
    
    # NEW ENHANCED FIELDS
    semantic_coherence_score: float  # How well clips flow semantically
    contextual_flow_score: float     # Narrative/emotional flow
    phrase_match_score: float        # Phrase-level semantic matching
    thematic_alignment_score: float  # Theme consistency
    
    music_sync_points: List[Dict[str, float]]
    sequence_hash: str
    selection_timestamp: float
    
    # Context tracking
    emotional_progression: List[str] = field(default_factory=list)
    narrative_markers: List[str] = field(default_factory=list)

class EnhancedContentSelector:
    """Enhanced clip selection engine with semantic intelligence and context awareness"""
    
    def __init__(self, content_database, learning_cache_path: str = "cache/selection_learning.json"):
        """
        Initialize enhanced content selector
        
        Args:
            content_database: ContentDatabase instance
            learning_cache_path: Path to save selection learning data
        """
        self.content_database = content_database
        self.learning_cache_path = Path(learning_cache_path)
        self.selection_history: List[str] = []
        
        # DYNAMIC LEARNING SYSTEM
        self.success_correlations = defaultdict(float)  # Track successful clip-script combinations
        self.phrase_clip_mappings = defaultdict(list)   # Track which clips work with which phrases
        self.contextual_flow_patterns = defaultdict(list)  # Track successful clip sequences
        
        # Load existing learning data
        self._load_learning_data()
        
        # ENHANCED EMOTION-VISUAL MAPPING (expanded from 5 to 12 emotions)
        self.emotion_visual_mapping = {
            'anxiety': {
                'preferred_lighting': ['dramatic', 'dark', 'natural'],
                'preferred_movement': ['dynamic', 'static'],
                'preferred_shots': ['close_up', 'medium_shot'],
                'visual_keywords': ['shadows', 'dramatic', 'tense', 'conflicted', 'struggle'],
                'context_flow': ['tension_building', 'inner_conflict', 'breakthrough']
            },
            'peace': {
                'preferred_lighting': ['natural', 'soft', 'bright'],
                'preferred_movement': ['static', 'gentle'],
                'preferred_shots': ['wide_shot', 'medium_shot'],
                'visual_keywords': ['serene', 'meditation', 'calm', 'lotus', 'floating', 'tranquil'],
                'context_flow': ['settling', 'centering', 'harmony']
            },
            'seeking': {
                'preferred_lighting': ['natural', 'dramatic'],
                'preferred_movement': ['dynamic', 'gentle'],
                'preferred_shots': ['medium_shot', 'wide_shot'],
                'visual_keywords': ['contemplative', 'introspective', 'journey', 'path', 'searching'],
                'context_flow': ['questioning', 'exploring', 'discovering']
            },
            'awakening': {
                'preferred_lighting': ['bright', 'natural', 'soft'],
                'preferred_movement': ['dynamic', 'static'],
                'preferred_shots': ['wide_shot', 'close_up'],
                'visual_keywords': ['bright', 'temple', 'spiritual', 'realization', 'enlightenment'],
                'context_flow': ['revelation', 'understanding', 'transformation']
            },
            'contemplative': {  # NEW
                'preferred_lighting': ['soft', 'natural'],
                'preferred_movement': ['static', 'gentle'],
                'preferred_shots': ['medium_shot', 'close_up'],
                'visual_keywords': ['meditation', 'reflection', 'thoughtful', 'introspective', 'quiet'],
                'context_flow': ['deep_thought', 'reflection', 'processing']
            },
            'transformative': {  # NEW
                'preferred_lighting': ['bright', 'dramatic'],
                'preferred_movement': ['dynamic'],
                'preferred_shots': ['wide_shot', 'medium_shot'],
                'visual_keywords': ['change', 'growth', 'evolution', 'becoming', 'transformation'],
                'context_flow': ['change_initiation', 'transformation_process', 'new_beginning']
            },
            'communal': {  # NEW
                'preferred_lighting': ['natural', 'bright'],
                'preferred_movement': ['dynamic', 'gentle'],
                'preferred_shots': ['wide_shot'],
                'visual_keywords': ['group', 'community', 'together', 'gathering', 'fellowship'],
                'context_flow': ['gathering', 'sharing', 'unity']
            },
            'transcendent': {  # NEW
                'preferred_lighting': ['bright', 'dramatic'],
                'preferred_movement': ['static', 'gentle'],
                'preferred_shots': ['wide_shot', 'medium_shot'],
                'visual_keywords': ['divine', 'eternal', 'soul', 'infinite', 'celestial', 'floating'],
                'context_flow': ['elevation', 'transcendence', 'divine_connection']
            },
            'grounding': {  # NEW
                'preferred_lighting': ['natural'],
                'preferred_movement': ['static'],
                'preferred_shots': ['medium_shot', 'close_up'],
                'visual_keywords': ['stable', 'foundation', 'rooted', 'secure', 'solid'],
                'context_flow': ['stabilizing', 'centering', 'securing']
            },
            'struggle': {  # NEW
                'preferred_lighting': ['dramatic', 'dark'],
                'preferred_movement': ['dynamic'],
                'preferred_shots': ['close_up', 'medium_shot'],
                'visual_keywords': ['challenge', 'difficulty', 'obstacle', 'effort', 'determination'],
                'context_flow': ['challenge_presentation', 'struggle_process', 'effort']
            },
            'liberation': {  # NEW
                'preferred_lighting': ['bright', 'natural'],
                'preferred_movement': ['dynamic'],
                'preferred_shots': ['wide_shot'],
                'visual_keywords': ['freedom', 'release', 'open', 'breakthrough', 'escape'],
                'context_flow': ['breaking_free', 'liberation_moment', 'freedom_expression']
            },
            'devotional': {  # NEW
                'preferred_lighting': ['soft', 'natural'],
                'preferred_movement': ['gentle', 'static'],
                'preferred_shots': ['close_up', 'medium_shot'],
                'visual_keywords': ['devotion', 'worship', 'love', 'service', 'surrender', 'offering'],
                'context_flow': ['devotional_approach', 'loving_service', 'surrender']
            },
            'neutral': {
                'preferred_lighting': ['natural'],
                'preferred_movement': ['static'],
                'preferred_shots': ['medium_shot'],
                'visual_keywords': ['general', 'peaceful'],
                'context_flow': ['neutral', 'balanced']
            }
        }
        
        # PHRASE-LEVEL SEMANTIC MATCHING PATTERNS
        self.phrase_semantic_mappings = {
            'spiritual_identity': {
                'clip_tags': ['soul', 'spiritual', 'divine', 'eternal', 'identity', 'consciousness'],
                'visual_themes': ['meditation', 'contemplation', 'inner_focus', 'realization'],
                'weight_multiplier': 3.0
            },
            'divine_connection': {
                'clip_tags': ['krishna', 'devotion', 'worship', 'service', 'love', 'surrender'],
                'visual_themes': ['devotional', 'spiritual', 'sacred', 'temple', 'worship'],
                'weight_multiplier': 2.5
            },
            'material_illusion': {
                'clip_tags': ['material', 'temporary', 'illusion', 'external', 'physical'],
                'visual_themes': ['contrast', 'material_world', 'temporary_things'],
                'weight_multiplier': 2.0
            },
            'inner_transformation': {
                'clip_tags': ['growth', 'change', 'transformation', 'evolution', 'development'],
                'visual_themes': ['transformation', 'growth', 'change', 'becoming'],
                'weight_multiplier': 2.5
            },
            'life_purpose': {
                'clip_tags': ['purpose', 'meaning', 'realization', 'calling', 'destiny'],
                'visual_themes': ['purpose', 'direction', 'clarity', 'understanding'],
                'weight_multiplier': 3.0
            },
            'modern_struggles': {
                'clip_tags': ['phone', 'technology', 'modern', 'digital', 'addiction'],
                'visual_themes': ['modern_life', 'technology', 'contemporary'],
                'weight_multiplier': 2.0
            }
        }
    
    async def select_clips_for_script(self, 
                               script_analysis: EnhancedScriptAnalysis,
                               clip_count: Optional[int] = None,
                               script_duration: Optional[float] = None,
                               music_beats: Optional[List[float]] = None,
                               min_clip_duration: float = 2.0,
                               variation_seed: Optional[int] = None,
                               sync_event_type: str = 'beat',
                               use_percussive_sync: bool = False) -> ContextualSequence:
        """
        Enhanced clip selection with semantic phrase matching and context awareness
        """
        # Enhanced seed generation for better variety
        if variation_seed is not None:
            enhanced_seed = variation_seed + int(time.time() * 1000) % 10000
            random.seed(enhanced_seed)
            logger.debug(f"Using enhanced seed: {enhanced_seed} (base: {variation_seed})")
        
        # Calculate clip count with beat synchronization
        if clip_count is None:
            if script_duration is not None and music_beats:
                relevant_beats = [beat for beat in music_beats if beat <= script_duration]
                if len(relevant_beats) >= 2:
                    grouped_beats = [relevant_beats[i] for i in range(0, len(relevant_beats), 6)]
                    if grouped_beats[-1] != relevant_beats[-1]:
                        grouped_beats.append(relevant_beats[-1])
                    clip_count = len(grouped_beats) - 1
                    avg_duration = script_duration / clip_count if clip_count > 0 else min_clip_duration
                    logger.info(f"Beat-synchronized clips every 6 beats: {clip_count} clips (avg {avg_duration:.1f}s)")
                else:
                    clip_count = max(1, int(script_duration / min_clip_duration))
                    logger.warning(f"Insufficient beats ({len(relevant_beats)}), using fixed {min_clip_duration}s clips")
            else:
                clip_count = max(1, int((script_duration or 60) / min_clip_duration))
        
        # Create enhanced selection criteria
        criteria = EnhancedSelectionCriteria(
            emotion=script_analysis.primary_emotion,
            intensity=script_analysis.emotional_intensity,
            themes=script_analysis.themes,
            duration_target=script_duration or (min_clip_duration * clip_count),
            music_beats=music_beats or [],
            semantic_phrases=script_analysis.semantic_phrases,
            content_keywords=script_analysis.content_keywords,
            thematic_weights=script_analysis.thematic_categories,
            context_markers=script_analysis.context_markers,
            visual_variety=True,
            sync_event_type=sync_event_type,
            use_percussive_sync=use_percussive_sync
        )
        
        # Get candidate clips with enhanced scoring
        candidate_clips = self._get_enhanced_candidate_clips(criteria)
        
        if len(candidate_clips) < clip_count:
            logger.warning(f"Only {len(candidate_clips)} clips available for {script_analysis.filename}")
            clip_count = len(candidate_clips)
        
        # Select optimal sequence with context awareness
        selected_clips = self._select_contextual_sequence(candidate_clips, criteria, clip_count)
        
        # Calculate enhanced metrics
        metrics = self._calculate_enhanced_metrics(selected_clips, criteria, script_analysis)
        
        # Generate music sync points
        sync_points = self._generate_music_sync_points(selected_clips, criteria.music_beats, sync_event_type)
        
        # Create sequence hash
        sequence_hash = self._generate_sequence_hash(selected_clips)
        
        # Build contextual sequence
        sequence = ContextualSequence(
            clips=selected_clips,
            total_duration=sum(clip.duration for clip in selected_clips),
            relevance_score=metrics['relevance_score'],
            visual_variety_score=metrics['visual_variety_score'],
            semantic_coherence_score=metrics['semantic_coherence_score'],
            contextual_flow_score=metrics['contextual_flow_score'],
            phrase_match_score=metrics['phrase_match_score'],
            thematic_alignment_score=metrics['thematic_alignment_score'],
            music_sync_points=sync_points,
            sequence_hash=sequence_hash,
            selection_timestamp=time.time(),
            emotional_progression=metrics['emotional_progression'],
            narrative_markers=metrics['narrative_markers']
        )
        
        # Learn from this selection
        self._record_selection_for_learning(sequence, script_analysis)
        
        logger.info(f"Enhanced selection for {script_analysis.filename}: "
                   f"relevance={metrics['relevance_score']:.3f}, "
                   f"semantic={metrics['semantic_coherence_score']:.3f}, "
                   f"flow={metrics['contextual_flow_score']:.3f}")
        
        return sequence
    
    def _get_enhanced_candidate_clips(self, criteria: EnhancedSelectionCriteria) -> List[Any]:
        """Get clips using enhanced semantic matching with phrase-level analysis"""
        all_clips = list(self.content_database.clips_loader.clips.values())
        logger.info(f"Analyzing {len(all_clips)} clips with enhanced semantic matching")
        
        scored_clips = []
        
        for clip in all_clips:
            score = 0.0
            clip_text = ' '.join(clip.tags).lower()
            
            # 1. PHRASE-LEVEL SEMANTIC MATCHING (highest weight)
            phrase_score = self._calculate_phrase_match_score(clip, criteria.semantic_phrases)
            score += phrase_score * 4.0  # 40% weight
            
            # 2. ENHANCED THEMATIC MATCHING
            thematic_score = self._calculate_thematic_alignment_score(clip, criteria.thematic_weights)
            score += thematic_score * 3.0  # 30% weight
            
            # 3. KEYWORD CONTENT MATCHING
            keyword_score = self._calculate_keyword_match_score(clip, criteria.content_keywords)
            score += keyword_score * 2.0  # 20% weight
            
            # 4. LEARNED CORRELATION BONUS
            correlation_bonus = self._get_learned_correlation_score(clip, criteria)
            score += correlation_bonus * 1.0  # 10% weight
            
            # 5. VISUAL CHARACTERISTICS (secondary)
            visual_score = self._calculate_visual_match_score(clip, criteria.emotion)
            score += visual_score * 0.5  # 5% weight (reduced from original)
            
            # 6. CONTEXT MARKERS BONUS
            context_bonus = self._calculate_context_bonus(clip, criteria.context_markers)
            score += context_bonus * 0.5  # 5% weight
            
            scored_clips.append((clip, score))
        
        # Sort by enhanced relevance score
        scored_clips.sort(key=lambda x: x[1], reverse=True)
        
        # Log top scoring insights
        if scored_clips:
            top_clip, top_score = scored_clips[0]
            logger.debug(f"Top clip: {top_clip.filename} (score: {top_score:.3f})")
        
        return [clip for clip, score in scored_clips]
    
    def _calculate_phrase_match_score(self, clip: Any, semantic_phrases: List[str]) -> float:
        """Calculate how well clip matches semantic phrases"""
        if not semantic_phrases:
            return 0.0
        
        clip_text = ' '.join(clip.tags).lower()
        phrase_score = 0.0
        
        for phrase in semantic_phrases:
            if ':' in phrase:
                category, content = phrase.split(':', 1)
                
                # Check if category has mapping
                if category in self.phrase_semantic_mappings:
                    mapping = self.phrase_semantic_mappings[category]
                    
                    # Check for tag matches
                    tag_matches = sum(1 for tag in mapping['clip_tags'] if tag in clip_text)
                    
                    # Check for visual theme matches
                    theme_matches = sum(1 for theme in mapping['visual_themes'] if theme in clip_text)
                    
                    # Calculate weighted score
                    category_score = (tag_matches + theme_matches) * mapping['weight_multiplier']
                    phrase_score += category_score
                    
                    # Check learned mappings
                    if category in self.phrase_clip_mappings:
                        if clip.id in self.phrase_clip_mappings[category]:
                            phrase_score += 2.0  # Learned correlation bonus
        
        return min(phrase_score / max(len(semantic_phrases), 1), 1.0)
    
    def _calculate_thematic_alignment_score(self, clip: Any, thematic_weights: Dict[str, float]) -> float:
        """Calculate thematic alignment between clip and script"""
        if not thematic_weights:
            return 0.0
        
        clip_text = ' '.join(clip.tags).lower()
        alignment_score = 0.0
        
        for theme_category, weight in thematic_weights.items():
            if weight > 0:
                # Map theme categories to clip characteristics
                category_keywords = {
                    'spiritual_philosophy': ['soul', 'consciousness', 'divine', 'eternal', 'spiritual'],
                    'practical_spirituality': ['meditation', 'chanting', 'service', 'practice', 'temple'],
                    'psychological_states': ['peaceful', 'dramatic', 'contemplative', 'serene'],
                    'life_guidance': ['journey', 'path', 'realization', 'growth', 'transformation'],
                    'modern_challenges': ['phone', 'technology', 'modern', 'contemporary'],
                    'relationships': ['group', 'community', 'gathering', 'together', 'fellowship'],
                    'transformation': ['change', 'growth', 'evolution', 'becoming', 'development']
                }
                
                if theme_category in category_keywords:
                    category_matches = sum(1 for keyword in category_keywords[theme_category] 
                                         if keyword in clip_text)
                    alignment_score += category_matches * weight * 2.0
        
        return min(alignment_score, 1.0)
    
    def _calculate_keyword_match_score(self, clip: Any, content_keywords: List[str]) -> float:
        """Calculate keyword content matching score"""
        if not content_keywords:
            return 0.0
        
        clip_text = ' '.join(clip.tags).lower()
        matches = sum(1 for keyword in content_keywords if keyword in clip_text)
        
        return min(matches / len(content_keywords), 1.0)
    
    def _get_learned_correlation_score(self, clip: Any, criteria: EnhancedSelectionCriteria) -> float:
        """Get score bonus from learned correlations"""
        correlation_key = f"{criteria.emotion}:{clip.id}"
        return min(self.success_correlations.get(correlation_key, 0.0), 1.0)
    
    def _calculate_visual_match_score(self, clip: Any, emotion: str) -> float:
        """Calculate visual characteristics matching (original logic, reduced weight)"""
        if emotion not in self.emotion_visual_mapping:
            return 0.0
        
        visual_prefs = self.emotion_visual_mapping[emotion]
        score = 0.0
        
        if clip.lighting_type in visual_prefs['preferred_lighting']:
            score += 0.3
        if clip.movement_type in visual_prefs['preferred_movement']:
            score += 0.3
        if clip.shot_type in visual_prefs['preferred_shots']:
            score += 0.2
        
        # Visual keywords matching
        clip_text = ' '.join(clip.tags).lower()
        keyword_matches = sum(1 for keyword in visual_prefs['visual_keywords'] if keyword in clip_text)
        score += min(keyword_matches * 0.1, 0.2)
        
        return score
    
    def _calculate_context_bonus(self, clip: Any, context_markers: List[str]) -> float:
        """Calculate context-specific bonus"""
        if not context_markers:
            return 0.0
        
        clip_text = ' '.join(clip.tags).lower()
        bonus = 0.0
        
        context_mappings = {
            'temporal_narrative': ['sequence', 'time', 'progression', 'story'],
            'emotional_contrast': ['dramatic', 'contrast', 'change', 'transition'],
            'actionable_advice': ['practice', 'action', 'doing', 'active'],
            'personal_testimony': ['personal', 'individual', 'intimate', 'experience'],
            'philosophical_depth': ['deep', 'profound', 'spiritual', 'contemplative'],
            'contemporary_relevance': ['modern', 'current', 'relevant', 'contemporary']
        }
        
        for marker in context_markers:
            if marker in context_mappings:
                marker_words = context_mappings[marker]
                matches = sum(1 for word in marker_words if word in clip_text)
                bonus += matches * 0.1
        
        return min(bonus, 0.5)
    
    def _select_contextual_sequence(self, candidates: List[Any], criteria: EnhancedSelectionCriteria, clip_count: int) -> List[Any]:
        """Select sequence with context awareness and flow optimization"""
        if len(candidates) <= clip_count:
            return candidates
        
        # Enhanced randomization
        random.shuffle(candidates)
        
        selected = []
        used_characteristics = {'lighting': set(), 'movement': set(), 'shot_type': set()}
        
        # Context flow tracking
        current_emotional_flow = []
        narrative_context = []
        
        # Select clips with contextual flow awareness
        for i in range(clip_count):
            best_clip = None
            best_score = -1
            
            # Evaluate remaining candidates for contextual fit
            for clip in candidates:
                if clip in selected:
                    continue
                
                score = 0.0
                
                # Visual variety bonus
                if (clip.lighting_type not in used_characteristics['lighting'] or
                    clip.movement_type not in used_characteristics['movement'] or
                    clip.shot_type not in used_characteristics['shot_type']):
                    score += 2.0
                
                # Enhanced contextual flow bonus
                flow_score = self._calculate_contextual_flow_score(
                    clip, selected, criteria, current_emotional_flow
                )
                score += flow_score * 3.0
                
                # Enhanced semantic relevance bonus
                semantic_score = self._calculate_enhanced_semantic_score(clip, criteria)
                score += semantic_score * 4.0  # Higher weight for semantic matching
                
                # Position-specific bonuses
                if i == 0:  # Opening clip
                    if 'opening' in ' '.join(clip.tags).lower():
                        score += 1.0
                elif i == clip_count - 1:  # Closing clip
                    if any(word in ' '.join(clip.tags).lower() 
                          for word in ['conclusion', 'peace', 'resolution']):
                        score += 1.0
                
                if score > best_score:
                    best_score = score
                    best_clip = clip
            
            if best_clip:
                selected.append(best_clip)
                used_characteristics['lighting'].add(best_clip.lighting_type)
                used_characteristics['movement'].add(best_clip.movement_type)
                used_characteristics['shot_type'].add(best_clip.shot_type)
                
                # Update flow context
                self._update_flow_context(best_clip, current_emotional_flow, narrative_context)
        
        # Final sequence optimization
        if len(selected) < clip_count:
            remaining_needed = clip_count - len(selected)
            remaining_candidates = [c for c in candidates if c not in selected]
            if remaining_candidates:
                selected.extend(random.sample(remaining_candidates, 
                                            min(remaining_needed, len(remaining_candidates))))
        
        return selected[:clip_count]
    
    def _calculate_contextual_flow_score(self, clip: Any, selected_clips: List[Any], 
                                       criteria: EnhancedSelectionCriteria, 
                                       current_flow: List[str]) -> float:
        """Calculate how well clip fits in the current contextual flow"""
        if not selected_clips:
            return 0.5  # Neutral score for first clip
        
        flow_score = 0.0
        last_clip = selected_clips[-1]
        clip_text = ' '.join(clip.tags).lower()
        last_clip_text = ' '.join(last_clip.tags).lower()
        
        # Emotional flow compatibility
        if criteria.emotion in self.emotion_visual_mapping:
            emotion_config = self.emotion_visual_mapping[criteria.emotion]
            context_flow = emotion_config.get('context_flow', [])
            
            position_in_sequence = len(selected_clips)
            expected_flow_stage = position_in_sequence % len(context_flow) if context_flow else 0
            
            if expected_flow_stage < len(context_flow):
                expected_stage = context_flow[expected_flow_stage]
                
                # Map stages to visual characteristics
                stage_mappings = {
                    'tension_building': ['dramatic', 'intense', 'building'],
                    'inner_conflict': ['conflicted', 'struggle', 'tension'],
                    'breakthrough': ['bright', 'realization', 'clarity'],
                    'settling': ['calm', 'peaceful', 'settling'],
                    'centering': ['meditation', 'focus', 'center'],
                    'harmony': ['balanced', 'harmonious', 'unified']
                }
                
                if expected_stage in stage_mappings:
                    stage_keywords = stage_mappings[expected_stage]
                    if any(keyword in clip_text for keyword in stage_keywords):
                        flow_score += 1.0
        
        # Visual transition scoring
        transition_score = self._calculate_visual_transition_score(last_clip, clip)
        flow_score += transition_score * 0.5
        
        # Learned flow patterns
        flow_pattern = f"{last_clip.id}->{clip.id}"
        if flow_pattern in self.contextual_flow_patterns:
            flow_score += 0.5
        
        return flow_score
    
    def _calculate_visual_transition_score(self, last_clip: Any, current_clip: Any) -> float:
        """Calculate visual transition smoothness"""
        score = 0.0
        
        # Lighting transitions
        lighting_transitions = {
            ('dark', 'dramatic'): 0.8,
            ('dramatic', 'natural'): 0.7,
            ('natural', 'soft'): 0.8,
            ('soft', 'bright'): 0.7,
            ('bright', 'natural'): 0.6
        }
        
        transition_key = (last_clip.lighting_type, current_clip.lighting_type)
        if transition_key in lighting_transitions:
            score += lighting_transitions[transition_key]
        elif last_clip.lighting_type == current_clip.lighting_type:
            score += 0.5  # Same lighting is neutral
        
        # Movement transitions
        if last_clip.movement_type != current_clip.movement_type:
            score += 0.3  # Variety bonus
        
        # Shot type transitions
        shot_transitions = {
            ('wide_shot', 'medium_shot'): 0.8,
            ('medium_shot', 'close_up'): 0.7,
            ('close_up', 'medium_shot'): 0.6,
            ('medium_shot', 'wide_shot'): 0.5
        }
        
        shot_key = (last_clip.shot_type, current_clip.shot_type)
        if shot_key in shot_transitions:
            score += shot_transitions[shot_key]
        
        return score
    
    def _calculate_enhanced_semantic_score(self, clip: Any, criteria: EnhancedSelectionCriteria) -> float:
        """Calculate enhanced semantic relevance score for a clip"""
        clip_text = ' '.join(clip.tags).lower()
        total_score = 0.0
        
        # Theme matching with dynamic weights (40%)
        theme_score = 0.0
        for theme in criteria.themes:
            theme_words = theme.replace('_', ' ').split()
            matches = sum(1 for word in theme_words if word in clip_text)
            if matches > 0:
                theme_score += matches / len(theme_words)
        theme_score = min(theme_score / len(criteria.themes) if criteria.themes else 0, 1.0)
        total_score += theme_score * 0.4
        
        # Phrase matching (30%)
        phrase_score = self._calculate_phrase_match_score(clip, criteria.semantic_phrases)
        total_score += phrase_score * 0.3
        
        # Keyword matching (20%)
        keyword_score = self._calculate_keyword_match_score(clip, criteria.content_keywords)
        total_score += keyword_score * 0.2
        
        # Visual-emotional alignment (10%)
        visual_score = self._calculate_visual_emotional_alignment(clip, criteria.emotion)
        total_score += visual_score * 0.1
        
        return total_score
    
    def _calculate_visual_emotional_alignment(self, clip: Any, emotion: str) -> float:
        """Calculate how well clip's visual characteristics align with emotion"""
        clip_text = ' '.join(clip.tags).lower()
        
        emotion_visual_mappings = {
            'anxiety': ['dark', 'chaotic', 'fast', 'intense', 'unsettling'],
            'peace': ['calm', 'serene', 'gentle', 'balanced', 'soft'],
            'seeking': ['dynamic', 'searching', 'movement', 'journey'],
            'awakening': ['bright', 'illuminated', 'transformation', 'radiant'],
            'contemplative': ['thoughtful', 'meditative', 'still', 'introspective'],
            'transcendent': ['elevated', 'ethereal', 'divine', 'luminous']
        }
        
        if emotion not in emotion_visual_mappings:
            return 0.5  # Neutral score for unknown emotions
        
        visual_keywords = emotion_visual_mappings[emotion]
        matches = sum(1 for keyword in visual_keywords if keyword in clip_text)
        
        return min(matches / len(visual_keywords), 1.0)

    def _update_flow_context(self, clip: Any, emotional_flow: List[str], narrative_context: List[str]):
        """Update flow context based on selected clip"""
        clip_text = ' '.join(clip.tags).lower()
        
        # Extract emotional indicators
        emotional_indicators = {
            'peaceful': ['peaceful', 'calm', 'serene', 'tranquil'],
            'dramatic': ['dramatic', 'intense', 'powerful'],
            'contemplative': ['contemplative', 'reflective', 'thoughtful'],
            'spiritual': ['spiritual', 'divine', 'sacred'],
            'active': ['dynamic', 'movement', 'action']
        }
        
        for emotion, keywords in emotional_indicators.items():
            if any(keyword in clip_text for keyword in keywords):
                emotional_flow.append(emotion)
                break
        else:
            emotional_flow.append('neutral')
        
        # Extract narrative markers
        narrative_markers = {
            'opening': ['beginning', 'start', 'introduction'],
            'development': ['development', 'progress', 'growth'],
            'climax': ['peak', 'intense', 'climax', 'breakthrough'],
            'resolution': ['resolution', 'conclusion', 'peace', 'end']
        }
        
        for marker, keywords in narrative_markers.items():
            if any(keyword in clip_text for keyword in keywords):
                narrative_context.append(marker)
                break
    
    def _calculate_enhanced_metrics(self, clips: List[Any], criteria: EnhancedSelectionCriteria, 
                                  script_analysis: EnhancedScriptAnalysis) -> Dict[str, Any]:
        """Calculate comprehensive enhanced metrics"""
        if not clips:
            return {
                'relevance_score': 0.0, 'visual_variety_score': 0.0,
                'semantic_coherence_score': 0.0, 'contextual_flow_score': 0.0,
                'phrase_match_score': 0.0, 'thematic_alignment_score': 0.0,
                'emotional_progression': [], 'narrative_markers': []
            }
        
        # Calculate individual metrics
        relevance_score = self._calculate_sequence_relevance(clips, criteria)
        visual_variety_score = self._calculate_visual_variety(clips)
        semantic_coherence_score = self._calculate_semantic_coherence(clips, criteria)
        contextual_flow_score = self._calculate_sequence_flow(clips, criteria)
        phrase_match_score = self._calculate_sequence_phrase_match(clips, criteria)
        thematic_alignment_score = self._calculate_sequence_thematic_alignment(clips, criteria)
        
        # Extract progression and markers
        emotional_progression = self._extract_emotional_progression(clips)
        narrative_markers = self._extract_narrative_markers(clips)
        
        return {
            'relevance_score': relevance_score,
            'visual_variety_score': visual_variety_score,
            'semantic_coherence_score': semantic_coherence_score,
            'contextual_flow_score': contextual_flow_score,
            'phrase_match_score': phrase_match_score,
            'thematic_alignment_score': thematic_alignment_score,
            'emotional_progression': emotional_progression,
            'narrative_markers': narrative_markers
        }
    
    def _calculate_sequence_relevance(self, clips: List[Any], criteria: EnhancedSelectionCriteria) -> float:
        """Calculate overall sequence relevance (enhanced)"""
        if not clips:
            return 0.0
        
        total_score = 0.0
        for clip in clips:
            clip_text = ' '.join(clip.tags).lower()
            
            # Semantic theme matching (40% weight)
            theme_score = 0.0
            for theme in criteria.themes:
                theme_words = theme.replace('_', ' ').split()
                theme_matches = sum(1 for word in theme_words if word in clip_text)
                theme_score += theme_matches
            total_score += theme_score * 0.4
            
            # Phrase matching (30% weight)
            phrase_score = self._calculate_phrase_match_score(clip, criteria.semantic_phrases)
            total_score += phrase_score * 0.3
            
            # Keyword matching (20% weight)
            keyword_score = self._calculate_keyword_match_score(clip, criteria.content_keywords)
            total_score += keyword_score * 0.2
            
            # Visual characteristics (10% weight)
            visual_score = self._calculate_visual_match_score(clip, criteria.emotion)
            total_score += visual_score * 0.1
        
        return min(total_score / len(clips), 1.0)
    
    def _calculate_visual_variety(self, clips: List[Any]) -> float:
        """Calculate visual variety score (original logic)"""
        if len(clips) <= 1:
            return 1.0
        
        unique_lighting = len(set(clip.lighting_type for clip in clips))
        unique_movement = len(set(clip.movement_type for clip in clips))
        unique_shots = len(set(clip.shot_type for clip in clips))
        
        lighting_variety = unique_lighting / len(clips)
        movement_variety = unique_movement / len(clips)
        shot_variety = unique_shots / len(clips)
        
        return (lighting_variety * 0.5 + movement_variety * 0.3 + shot_variety * 0.2)
    
    def _calculate_semantic_coherence(self, clips: List[Any], criteria: EnhancedSelectionCriteria) -> float:
        """Calculate semantic coherence across the sequence"""
        if not clips:
            return 0.0
        
        # Measure thematic consistency
        clip_themes = []
        for clip in clips:
            clip_text = ' '.join(clip.tags).lower()
            clip_theme_scores = {}
            
            for theme_category, weight in criteria.thematic_weights.items():
                if weight > 0:
                    category_presence = sum(1 for word in clip_text.split() 
                                          if any(theme_word in word for theme_word in theme_category.split('_')))
                    clip_theme_scores[theme_category] = category_presence
            
            clip_themes.append(clip_theme_scores)
        
        # Calculate coherence as consistency of thematic presence
        if not clip_themes:
            return 0.0
        
        coherence_scores = []
        for theme_category in criteria.thematic_weights.keys():
            theme_values = [themes.get(theme_category, 0) for themes in clip_themes]
            if theme_values:
                # Coherence is higher when theme presence is consistent
                avg_presence = sum(theme_values) / len(theme_values)
                variance = sum((v - avg_presence) ** 2 for v in theme_values) / len(theme_values)
                coherence = max(0, 1 - variance)  # Lower variance = higher coherence
                coherence_scores.append(coherence)
        
        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0
    
    def _calculate_sequence_flow(self, clips: List[Any], criteria: EnhancedSelectionCriteria) -> float:
        """Calculate contextual flow quality"""
        if len(clips) <= 1:
            return 1.0
        
        flow_score = 0.0
        
        for i in range(len(clips) - 1):
            current_clip = clips[i]
            next_clip = clips[i + 1]
            
            # Visual transition score
            transition_score = self._calculate_visual_transition_score(current_clip, next_clip)
            flow_score += transition_score
            
            # Check for learned flow patterns
            flow_pattern = f"{current_clip.id}->{next_clip.id}"
            if flow_pattern in self.contextual_flow_patterns:
                flow_score += 0.5
        
        return flow_score / (len(clips) - 1)
    
    def _calculate_sequence_phrase_match(self, clips: List[Any], criteria: EnhancedSelectionCriteria) -> float:
        """Calculate phrase-level matching for entire sequence"""
        if not clips or not criteria.semantic_phrases:
            return 0.0
        
        total_phrase_score = 0.0
        for clip in clips:
            clip_phrase_score = self._calculate_phrase_match_score(clip, criteria.semantic_phrases)
            total_phrase_score += clip_phrase_score
        
        return total_phrase_score / len(clips)
    
    def _calculate_sequence_thematic_alignment(self, clips: List[Any], criteria: EnhancedSelectionCriteria) -> float:
        """Calculate thematic alignment for entire sequence"""
        if not clips or not criteria.thematic_weights:
            return 0.0
        
        total_alignment_score = 0.0
        for clip in clips:
            clip_alignment_score = self._calculate_thematic_alignment_score(clip, criteria.thematic_weights)
            total_alignment_score += clip_alignment_score
        
        return total_alignment_score / len(clips)
    
    def _extract_emotional_progression(self, clips: List[Any]) -> List[str]:
        """Extract emotional progression through the sequence"""
        progression = []
        
        emotional_indicators = {
            'peaceful': ['peaceful', 'calm', 'serene', 'tranquil'],
            'dramatic': ['dramatic', 'intense', 'powerful'],
            'contemplative': ['contemplative', 'reflective', 'thoughtful'],
            'spiritual': ['spiritual', 'divine', 'sacred'],
            'transformative': ['transformation', 'change', 'growth']
        }
        
        for clip in clips:
            clip_text = ' '.join(clip.tags).lower()
            
            for emotion, keywords in emotional_indicators.items():
                if any(keyword in clip_text for keyword in keywords):
                    progression.append(emotion)
                    break
            else:
                progression.append('neutral')
        
        return progression
    
    def _extract_narrative_markers(self, clips: List[Any]) -> List[str]:
        """Extract narrative structure markers"""
        markers = []
        
        narrative_indicators = {
            'opening': ['beginning', 'start', 'introduction'],
            'development': ['development', 'progress', 'journey'],
            'climax': ['peak', 'intense', 'breakthrough'],
            'resolution': ['resolution', 'peace', 'conclusion']
        }
        
        for i, clip in enumerate(clips):
            clip_text = ' '.join(clip.tags).lower()
            
            # Position-based markers
            if i == 0:
                markers.append('sequence_opening')
            elif i == len(clips) - 1:
                markers.append('sequence_closing')
            else:
                markers.append('sequence_development')
            
            # Content-based markers
            for marker, keywords in narrative_indicators.items():
                if any(keyword in clip_text for keyword in keywords):
                    markers.append(f'content_{marker}')
                    break
        
        return markers
    
    def _generate_music_sync_points(self, clips: List[Any], music_beats: List[float], sync_event_type: str = 'beat') -> List[Dict[str, float]]:
        """Generate music synchronization points (original logic)"""
        if not music_beats or not clips:
            return []
        
        sync_points = []
        relevant_beats = music_beats[:len(clips) + 1]
        
        for i in range(len(clips)):
            if i < len(relevant_beats) - 1:
                start_time = relevant_beats[i]
                end_time = relevant_beats[i + 1]
                duration = end_time - start_time
                
                sync_points.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'beat_index': i,
                    'sync_event_type': sync_event_type
                })
            else:
                # Fallback for clips beyond available beats
                previous_duration = sync_points[-1]['duration'] if sync_points else 2.0
                start_time = sync_points[-1]['end_time'] if sync_points else 0.0
                end_time = start_time + previous_duration
                
                sync_points.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': previous_duration,
                    'beat_index': -1,
                    'sync_event_type': sync_event_type
                })
        
        return sync_points
    
    def _generate_sequence_hash(self, clips: List[Any]) -> str:
        """Generate unique hash for clip sequence"""
        clip_ids = [clip.id for clip in clips]
        sequence_string = '|'.join(sorted(clip_ids))
        return hashlib.sha256(sequence_string.encode()).hexdigest()[:16]
    
    def _record_selection_for_learning(self, sequence: ContextualSequence, script_analysis: EnhancedScriptAnalysis):
        """Record selection data for learning system"""
        # Record success correlations (simplified - would be enhanced based on actual video performance)
        for clip in sequence.clips:
            correlation_key = f"{script_analysis.primary_emotion}:{clip.id}"
            self.success_correlations[correlation_key] += 0.1
        
        # Record phrase-clip mappings
        for phrase in script_analysis.semantic_phrases:
            if ':' in phrase:
                category = phrase.split(':', 1)[0]
                for clip in sequence.clips:
                    if clip.id not in self.phrase_clip_mappings[category]:
                        self.phrase_clip_mappings[category].append(clip.id)
        
        # Record contextual flow patterns
        for i in range(len(sequence.clips) - 1):
            flow_pattern = f"{sequence.clips[i].id}->{sequence.clips[i+1].id}"
            if flow_pattern not in self.contextual_flow_patterns:
                self.contextual_flow_patterns[flow_pattern] = []
            self.contextual_flow_patterns[flow_pattern].append(script_analysis.primary_emotion)
        
        # Save learning data periodically
        self._save_learning_data()
    
    def _load_learning_data(self):
        """Load existing learning data"""
        if self.learning_cache_path.exists():
            try:
                with open(self.learning_cache_path, 'r') as f:
                    data = json.load(f)
                
                self.success_correlations = defaultdict(float, data.get('success_correlations', {}))
                self.phrase_clip_mappings = defaultdict(list, data.get('phrase_clip_mappings', {}))
                self.contextual_flow_patterns = defaultdict(list, data.get('contextual_flow_patterns', {}))
                
                logger.info(f"Loaded learning data: {len(self.success_correlations)} correlations, "
                           f"{len(self.phrase_clip_mappings)} phrase mappings, "
                           f"{len(self.contextual_flow_patterns)} flow patterns")
            except Exception as e:
                logger.warning(f"Failed to load learning data: {e}")
    
    def _save_learning_data(self):
        """Save learning data to cache"""
        try:
            self.learning_cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'success_correlations': dict(self.success_correlations),
                'phrase_clip_mappings': dict(self.phrase_clip_mappings),
                'contextual_flow_patterns': dict(self.contextual_flow_patterns),
                'last_updated': time.time()
            }
            
            with open(self.learning_cache_path, 'w') as f:
                json.dump(data, f, indent=2)
            
        except Exception as e:
            logger.warning(f"Failed to save learning data: {e}")
    
    def get_selection_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about content selection"""
        return {
            'emotion_mappings_available': len(self.emotion_visual_mapping),
            'supported_emotions': list(self.emotion_visual_mapping.keys()),
            'phrase_semantic_categories': len(self.phrase_semantic_mappings),
            'learned_correlations': len(self.success_correlations),
            'phrase_clip_mappings': len(self.phrase_clip_mappings),
            'flow_patterns': len(self.contextual_flow_patterns),
            'selection_history_count': len(self.selection_history)
        }