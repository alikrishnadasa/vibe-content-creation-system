"""
Content Selector

Intelligent clip selection based on script analysis with music synchronization.
Maps emotional states to visual clip categories and generates optimal sequences.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
import random
import hashlib
import time

from .mjanime_loader import ClipMetadata
from .script_analyzer import ScriptAnalysis
from .music_manager import MusicManager

logger = logging.getLogger(__name__)

@dataclass
class SelectionCriteria:
    """Criteria for intelligent clip selection"""
    emotion: str
    intensity: float
    themes: List[str]
    duration_target: float
    music_beats: List[float]
    visual_variety: bool = True
    lighting_preference: Optional[str] = None
    movement_preference: Optional[str] = None
    sync_event_type: str = 'beat'  # 'beat', 'kick', 'snare', 'hihat', 'other'
    use_percussive_sync: bool = False

@dataclass
class SelectedSequence:
    """A selected sequence of clips with metadata"""
    clips: List[ClipMetadata]
    total_duration: float
    relevance_score: float
    visual_variety_score: float
    music_sync_points: List[Dict[str, float]]  # Beat-based timing information
    sequence_hash: str
    selection_timestamp: float

class ContentSelector:
    """Intelligent clip selection engine with music synchronization"""
    
    def __init__(self, content_database):
        """
        Initialize content selector
        
        Args:
            content_database: ContentDatabase instance
        """
        self.content_database = content_database
        self.selection_history: List[str] = []  # Track used sequences
        
        # Emotional mapping to visual characteristics
        self.emotion_visual_mapping = {
            'anxiety': {
                'preferred_lighting': ['dramatic', 'dark', 'natural'],
                'preferred_movement': ['dynamic', 'static'],
                'preferred_shots': ['close_up', 'medium_shot'],
                'visual_keywords': ['shadows', 'dramatic', 'tense', 'conflicted']
            },
            'peace': {
                'preferred_lighting': ['natural', 'soft'],
                'preferred_movement': ['static', 'slow'],
                'preferred_shots': ['wide_shot', 'medium_shot'],
                'visual_keywords': ['serene', 'meditation', 'calm', 'lotus', 'floating']
            },
            'seeking': {
                'preferred_lighting': ['natural', 'dramatic'],
                'preferred_movement': ['dynamic', 'static'],
                'preferred_shots': ['medium_shot', 'wide_shot'],
                'visual_keywords': ['contemplative', 'introspective', 'journey', 'path']
            },
            'awakening': {
                'preferred_lighting': ['bright', 'natural'],
                'preferred_movement': ['dynamic', 'static'],
                'preferred_shots': ['wide_shot', 'close_up'],
                'visual_keywords': ['bright', 'temple', 'spiritual', 'realization']
            },
            'neutral': {
                'preferred_lighting': ['natural'],
                'preferred_movement': ['static'],
                'preferred_shots': ['medium_shot'],
                'visual_keywords': ['general', 'peaceful']
            }
        }
    
    async def select_clips_for_script(self, 
                               script_analysis: ScriptAnalysis,
                               clip_count: Optional[int] = None,
                               script_duration: Optional[float] = None,
                               music_beats: Optional[List[float]] = None,
                               min_clip_duration: float = 2.0,
                               variation_seed: Optional[int] = None,
                               sync_event_type: str = 'beat',
                               use_percussive_sync: bool = False) -> SelectedSequence:
        """
        Select optimal clips for a script with music synchronization
        
        Args:
            script_analysis: Analysis of the audio script
            clip_count: Number of clips to select (optional, calculated from script_duration if not provided)
            script_duration: Duration of the script in seconds (used to calculate clip_count if provided)
            music_beats: Beat timestamps for synchronization
            min_clip_duration: Minimum duration for each clip
            variation_seed: Seed for variation generation
            sync_event_type: Type of percussive event to sync with ('beat', 'kick', 'snare', 'hihat', 'other')
            use_percussive_sync: Whether to use percussive event synchronization
            
        Returns:
            Selected sequence with clips and metadata
        """
        # Set variation seed for different sequences with more entropy
        if variation_seed is not None:
            # Add timestamp and more entropy for better randomization
            enhanced_seed = variation_seed + int(time.time() * 1000) % 10000
            random.seed(enhanced_seed)
            logger.debug(f"Using enhanced seed: {enhanced_seed} (base: {variation_seed})")
        
        # Calculate clip count and timing based on music beats with grouping every 6 beats
        if clip_count is None:
            if script_duration is not None and music_beats:
                # Group beats every 6 beats for fixed-length clips
                relevant_beats = [beat for beat in music_beats if beat <= script_duration]
                if len(relevant_beats) >= 2:
                    # Group every 6 beats
                    grouped_beats = [relevant_beats[i] for i in range(0, len(relevant_beats), 6)]
                    # Ensure we end at the last beat
                    if grouped_beats[-1] != relevant_beats[-1]:
                        grouped_beats.append(relevant_beats[-1])
                    clip_count = len(grouped_beats) - 1
                    avg_duration = script_duration / clip_count if clip_count > 0 else min_clip_duration
                    logger.info(f"Beat-synchronized clips every 6 beats: {clip_count} clips (avg {avg_duration:.1f}s) from {len(relevant_beats)} beats")
                else:
                    # Fallback to time-based calculation if insufficient beats
                    clip_count = max(1, int(script_duration / min_clip_duration))
                    logger.warning(f"Insufficient beats ({len(relevant_beats)}), using fixed {min_clip_duration}s clips: {clip_count} clips")
            elif script_duration is not None:
                # Fallback to time-based calculation without beats
                clip_count = max(1, int(script_duration / min_clip_duration))
                logger.info(f"No beats available, using fixed {min_clip_duration}s clips: {clip_count} clips")
            else:
                # Fallback to default if neither clip_count nor script_duration provided
                clip_count = 5
                logger.warning(f"No clip_count or script_duration provided, defaulting to {clip_count} clips")
        
        # Ensure we have valid music_beats list and group them if needed
        music_beats_list = music_beats or []
        if script_duration and music_beats_list and len(music_beats_list) >= 2:
            # Group every 6 beats for sync points
            relevant_beats = [beat for beat in music_beats_list if beat <= script_duration]
            grouped_beats = [relevant_beats[i] for i in range(0, len(relevant_beats), 6)]
            if grouped_beats[-1] != relevant_beats[-1]:
                grouped_beats.append(relevant_beats[-1])
            music_beats_list = grouped_beats
        
        # Create selection criteria
        criteria = SelectionCriteria(
            emotion=script_analysis.primary_emotion,
            intensity=script_analysis.emotional_intensity,
            themes=script_analysis.themes,
            duration_target=script_duration or (min_clip_duration * clip_count),  # Use script duration if available
            music_beats=music_beats_list,
            visual_variety=True,
            sync_event_type=sync_event_type,
            use_percussive_sync=use_percussive_sync
        )
        
        # Get candidate clips
        candidate_clips = self._get_candidate_clips(criteria)
        
        if len(candidate_clips) < clip_count:
            logger.warning(f"Only {len(candidate_clips)} clips available for {script_analysis.filename}")
            clip_count = len(candidate_clips)
        
        # Select optimal sequence
        selected_clips = self._select_optimal_sequence(candidate_clips, criteria, clip_count)
        
        # Calculate metrics
        total_duration = sum(clip.duration for clip in selected_clips)
        relevance_score = self._calculate_sequence_relevance(selected_clips, criteria)
        variety_score = self._calculate_visual_variety(selected_clips)
        
        # Generate music sync points
        sync_points = self._generate_music_sync_points(selected_clips, music_beats_list, sync_event_type)
        
        # Create sequence hash for uniqueness tracking
        sequence_hash = self._generate_sequence_hash(selected_clips)
        
        sequence = SelectedSequence(
            clips=selected_clips,
            total_duration=total_duration,
            relevance_score=relevance_score,
            visual_variety_score=variety_score,
            music_sync_points=sync_points,
            sequence_hash=sequence_hash,
            selection_timestamp=time.time()
        )
        
        logger.debug(f"Selected {len(selected_clips)} clips for {script_analysis.filename}: relevance={relevance_score:.3f}, variety={variety_score:.3f}")
        return sequence
    
    def _get_candidate_clips(self, criteria: SelectionCriteria) -> List[ClipMetadata]:
        """Get clips using semantic matching only (emotion filtering disabled)"""
        # Use ALL available clips instead of emotion-filtered clips
        all_clips = list(self.content_database.clips_loader.clips.values())
        logger.info(f"Using all {len(all_clips)} clips for selection (emotion filtering disabled)")
        
        # Apply semantic and visual scoring to ALL clips
        scored_clips = []
        
        for clip in all_clips:
            score = 0
            
            # Semantic theme matching (most important)
            clip_text = ' '.join(clip.tags).lower()
            for theme in criteria.themes:
                theme_words = theme.replace('_', ' ').split()
                theme_matches = sum(1 for word in theme_words if word in clip_text)
                score += theme_matches * 3  # Higher weight for semantic matching
            
            # Visual characteristics scoring (secondary)
            if criteria.emotion in self.emotion_visual_mapping:
                visual_prefs = self.emotion_visual_mapping[criteria.emotion]
                
                # Lighting preference
                if clip.lighting_type in visual_prefs['preferred_lighting']:
                    score += 1
                
                # Movement preference  
                if clip.movement_type in visual_prefs['preferred_movement']:
                    score += 1
                
                # Shot type preference
                if clip.shot_type in visual_prefs['preferred_shots']:
                    score += 1
                
                # Keyword matching
                keyword_matches = sum(1 for keyword in visual_prefs['visual_keywords']
                                    if keyword in clip_text)
                score += keyword_matches
            
            # Include all clips with any score (even 0) for maximum variety
            scored_clips.append((clip, score))
        
        # Sort by relevance score but keep all clips available
        scored_clips.sort(key=lambda x: x[1], reverse=True)
        return [clip for clip, score in scored_clips]
    
    def _get_related_emotions(self, emotion: str) -> List[str]:
        """Get emotions related to the primary emotion for fallback selection"""
        emotion_relationships = {
            'anxiety': ['seeking', 'neutral'],
            'peace': ['awakening', 'neutral'],
            'seeking': ['anxiety', 'awakening', 'peace'],
            'awakening': ['peace', 'seeking'],
            'neutral': ['peace', 'seeking']
        }
        return emotion_relationships.get(emotion, ['neutral'])
    
    def _select_optimal_sequence(self, 
                                candidates: List[ClipMetadata], 
                                criteria: SelectionCriteria,
                                clip_count: int) -> List[ClipMetadata]:
        """Select optimal sequence with enhanced randomization for variety"""
        if len(candidates) <= clip_count:
            return candidates
        
        # Enhanced randomization: shuffle candidates first
        random.shuffle(candidates)
        
        selected = []
        used_characteristics = {
            'lighting': set(),
            'movement': set(), 
            'shot_type': set()
        }
        
        # Strategy 1: Ensure visual variety first
        variety_candidates = []
        for clip in candidates:
            has_variety = (
                clip.lighting_type not in used_characteristics['lighting'] or
                clip.movement_type not in used_characteristics['movement'] or
                clip.shot_type not in used_characteristics['shot_type'] or
                len(selected) == 0
            )
            
            if has_variety and len(selected) < clip_count:
                selected.append(clip)
                used_characteristics['lighting'].add(clip.lighting_type)
                used_characteristics['movement'].add(clip.movement_type)
                used_characteristics['shot_type'].add(clip.shot_type)
            else:
                variety_candidates.append(clip)
        
        # Strategy 2: Fill remaining slots with semantic + random selection
        remaining_needed = clip_count - len(selected)
        if remaining_needed > 0 and variety_candidates:
            # Mix semantic relevance with randomization
            remaining_candidates = [clip for clip in variety_candidates if clip not in selected]
            
            # Take some high-relevance clips and some random clips
            high_relevance = remaining_candidates[:remaining_needed//2]
            random_selection = random.sample(
                remaining_candidates[remaining_needed//2:], 
                min(remaining_needed - len(high_relevance), len(remaining_candidates) - len(high_relevance))
            ) if len(remaining_candidates) > remaining_needed//2 else []
            
            selected.extend(high_relevance + random_selection)
        
        # Final randomization: shuffle the selected clips for sequence variety
        random.shuffle(selected)
        
        return selected[:clip_count]
    
    def _calculate_sequence_relevance(self, clips: List[ClipMetadata], criteria: SelectionCriteria) -> float:
        """Calculate overall relevance score based on semantic matching (emotion matching disabled)"""
        if not clips:
            return 0.0
        
        total_score = 0.0
        for clip in clips:
            # Primary: Semantic theme matching (heavily weighted)
            clip_text = ' '.join(clip.tags).lower()
            for theme in criteria.themes:
                theme_words = theme.replace('_', ' ').split()
                theme_matches = sum(1 for word in theme_words if word in clip_text)
                total_score += theme_matches * 0.6  # 60% weight for semantic matching
            
            # Secondary: Visual characteristics matching
            if criteria.emotion in self.emotion_visual_mapping:
                visual_prefs = self.emotion_visual_mapping[criteria.emotion]
                
                if clip.lighting_type in visual_prefs['preferred_lighting']:
                    total_score += 0.2
                if clip.movement_type in visual_prefs['preferred_movement']:
                    total_score += 0.1
                if clip.shot_type in visual_prefs['preferred_shots']:
                    total_score += 0.1
            
            # Emotion matching now has minimal weight (legacy compatibility)
            if criteria.emotion in clip.emotional_tags:
                total_score += 0.05  # Reduced from 0.4 to 0.05
        
        return total_score / len(clips)  # Average score
    
    def _calculate_visual_variety(self, clips: List[ClipMetadata]) -> float:
        """Calculate visual variety score for the sequence"""
        if len(clips) <= 1:
            return 1.0
        
        # Count unique characteristics
        unique_lighting = len(set(clip.lighting_type for clip in clips))
        unique_movement = len(set(clip.movement_type for clip in clips))
        unique_shots = len(set(clip.shot_type for clip in clips))
        
        # Calculate variety scores
        lighting_variety = unique_lighting / len(clips)
        movement_variety = unique_movement / len(clips)
        shot_variety = unique_shots / len(clips)
        
        # Weighted average (lighting is most important for variety)
        variety_score = (lighting_variety * 0.5 + 
                        movement_variety * 0.3 + 
                        shot_variety * 0.2)
        
        return variety_score
    
    def _generate_music_sync_points(self, clips: List[ClipMetadata], music_beats: List[float], sync_event_type: str = 'beat') -> List[Dict[str, float]]:
        """Generate music synchronization points with beat-based clip durations
        
        Args:
            clips: List of selected clips
            music_beats: List of beat timestamps (or percussive event timestamps)
            sync_event_type: Type of sync event ('beat', 'kick', 'snare', 'hihat', 'other')
            
        Returns:
            List of sync point dictionaries
        """
        if not music_beats or not clips:
            return []
        
        sync_points = []
        
        # Create beat-synchronized timing for clips
        relevant_beats = music_beats[:len(clips) + 1]  # Need one more beat than clips
        
        for i in range(len(clips)):
            if i < len(relevant_beats) - 1:
                # Use beat interval for this clip
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
                    'beat_index': -1,  # Indicates no beat sync
                    'sync_event_type': sync_event_type
                })
        
        return sync_points
    
    def _group_beats_for_min_duration(self, beats: List[float], min_duration: float) -> List[float]:
        """Group beats by taking every Nth beat to ensure minimum clip duration while staying on beat"""
        if not beats or len(beats) < 2:
            return beats
        
        # Calculate average beat interval
        total_duration = beats[-1] - beats[0]
        avg_beat_interval = total_duration / (len(beats) - 1)
        
        # Determine how many beats we need to span to meet minimum duration
        beats_per_clip = max(1, int(min_duration / avg_beat_interval))
        
        # Create grouped beats by taking every Nth beat
        grouped_beats = []
        for i in range(0, len(beats), beats_per_clip):
            grouped_beats.append(beats[i])
        
        # Ensure we end at the final beat
        if grouped_beats[-1] != beats[-1]:
            grouped_beats.append(beats[-1])
        
        actual_avg_duration = total_duration / (len(grouped_beats) - 1) if len(grouped_beats) > 1 else min_duration
        logger.info(f"Beat grouping: {len(beats)} beats → {len(grouped_beats)} clip boundaries ({beats_per_clip} beats per clip, avg {actual_avg_duration:.1f}s)")
        
        return grouped_beats
    
    def _generate_sequence_hash(self, clips: List[ClipMetadata]) -> str:
        """Generate unique hash for clip sequence"""
        clip_ids = [clip.id for clip in clips]
        sequence_string = '|'.join(sorted(clip_ids))
        return hashlib.sha256(sequence_string.encode()).hexdigest()[:16]
    
    def generate_multiple_sequences(self, 
                                   script_analysis: ScriptAnalysis,
                                   count: int = 5,
                                   script_duration: Optional[float] = None,
                                   music_beats: Optional[List[float]] = None) -> List[SelectedSequence]:
        """
        Generate multiple unique sequences for testing phase
        
        Args:
            script_analysis: Analysis of the audio script
            count: Number of unique sequences to generate
            script_duration: Duration of the script in seconds
            music_beats: Beat timestamps for synchronization
            
        Returns:
            List of unique selected sequences
        """
        sequences = []
        used_hashes = set()
        
        # Calculate clips needed based on script duration
        if script_duration is not None:
            clips_needed = max(3, int(script_duration / 2.0) + 1)
            target_duration = script_duration
        else:
            clips_needed = 3  # Fallback to 3 clips
            target_duration = 2.0 * clips_needed
        
        # Get a larger pool of candidate clips for variety
        criteria = SelectionCriteria(
            emotion=script_analysis.primary_emotion,
            intensity=script_analysis.emotional_intensity,
            themes=script_analysis.themes,
            duration_target=target_duration,
            music_beats=music_beats or []
        )
        
        candidate_clips = self._get_candidate_clips(criteria)
        
        if len(candidate_clips) < clips_needed * 2:  # Need enough clips for variety
            logger.warning(f"Only {len(candidate_clips)} candidate clips for {script_analysis.filename}")
        
        # Strategy 1: Different starting clips
        for start_idx in range(min(count, len(candidate_clips) - clips_needed + 1)):
            clips_subset = candidate_clips[start_idx:] + candidate_clips[:start_idx]
            selected_clips = self._select_optimal_sequence(clips_subset, criteria, clips_needed)
            
            if len(selected_clips) >= clips_needed:
                sequence = self._create_sequence_from_clips(selected_clips, criteria)
                if sequence.sequence_hash not in used_hashes:
                    sequences.append(sequence)
                    used_hashes.add(sequence.sequence_hash)
        
        # Strategy 2: Different combination approaches if we need more
        while len(sequences) < count and len(candidate_clips) >= clips_needed:
            # Try different randomization approaches
            import random
            random.seed(len(sequences) * 42 + int(time.time()) % 1000)
            
            # Random selection for variety
            if len(candidate_clips) >= clips_needed:
                random_clips = random.sample(candidate_clips, min(clips_needed, len(candidate_clips)))
                sequence = self._create_sequence_from_clips(random_clips, criteria)
                
                if sequence.sequence_hash not in used_hashes:
                    sequences.append(sequence)
                    used_hashes.add(sequence.sequence_hash)
            
            # Prevent infinite loop
            if len(sequences) >= count or len(used_hashes) >= min(count * 2, 20):
                break
        
        # Strategy 3: If still need more, try different clip counts or arrangements
        if len(sequences) < count and len(candidate_clips) >= clips_needed:
            for clips_count in range(clips_needed, min(clips_needed + 3, len(candidate_clips) + 1)):  # Try sequences with varying clip counts
                if len(candidate_clips) >= clips_count:
                    selected_clips = self._select_optimal_sequence(candidate_clips, criteria, clips_count)
                    if len(selected_clips) >= clips_needed:
                        sequence = self._create_sequence_from_clips(selected_clips, criteria)
                        if sequence.sequence_hash not in used_hashes:
                            sequences.append(sequence)
                            used_hashes.add(sequence.sequence_hash)
                        
                        if len(sequences) >= count:
                            break
        
        if len(sequences) < count:
            logger.warning(f"Only generated {len(sequences)}/{count} unique sequences for {script_analysis.filename}")
        
        # Sort by relevance score
        sequences.sort(key=lambda x: x.relevance_score, reverse=True)
        return sequences[:count]
    
    def _create_sequence_from_clips(self, clips: List[ClipMetadata], criteria: SelectionCriteria) -> SelectedSequence:
        """Create a SelectedSequence from a list of clips"""
        total_duration = sum(clip.duration for clip in clips)
        relevance_score = self._calculate_sequence_relevance(clips, criteria)
        variety_score = self._calculate_visual_variety(clips)
        sync_points = self._generate_music_sync_points(clips, criteria.music_beats)
        sequence_hash = self._generate_sequence_hash(clips)
        
        return SelectedSequence(
            clips=clips,
            total_duration=total_duration,
            relevance_score=relevance_score,
            visual_variety_score=variety_score,
            music_sync_points=sync_points,
            sequence_hash=sequence_hash,
            selection_timestamp=time.time()
        )
    
    def get_selection_stats(self) -> Dict[str, Any]:
        """Get statistics about content selection"""
        return {
            'emotion_mappings_available': len(self.emotion_visual_mapping),
            'supported_emotions': list(self.emotion_visual_mapping.keys()),
            'selection_history_count': len(self.selection_history)
        }