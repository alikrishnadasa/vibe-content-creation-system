"""
Content Database

Centralized management of clips, scripts, and music metadata.
Provides semantic search functionality and efficient indexing for content selection.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
import json
import time
import hashlib

from .mjanime_loader import MJAnimeLoader, ClipMetadata  
from .script_analyzer import AudioScriptAnalyzer, ScriptAnalysis
from .music_manager import MusicManager

logger = logging.getLogger(__name__)

@dataclass
class ContentMatch:
    """Represents a content matching result"""
    clip: ClipMetadata
    script: ScriptAnalysis
    relevance_score: float
    matching_emotions: List[str]
    matching_themes: List[str]

class ContentDatabase:
    """Centralized database for all content assets"""
    
    def __init__(self, 
                 clips_directory: str,
                 metadata_file: str, 
                 scripts_directory: str,
                 music_file: str):
        """
        Initialize the content database
        
        Args:
            clips_directory: Path to MJAnime clips
            metadata_file: Path to clips metadata JSON
            scripts_directory: Path to audio scripts 
            music_file: Path to music track
        """
        self.clips_loader = MJAnimeLoader(clips_directory, metadata_file)
        self.scripts_analyzer = AudioScriptAnalyzer(scripts_directory)
        self.music_manager = MusicManager(music_file)
        
        self.loaded = False
        self.cache_file = Path("cache/content_database_cache.json")
        self.cache_file.parent.mkdir(exist_ok=True)
        
    async def load_all_content(self) -> bool:
        """
        Load and analyze all content assets
        
        Returns:
            bool: True if successful
        """
        try:
            logger.info("Loading all content assets...")
            
            # Load clips
            if not await self.clips_loader.load_clips():
                logger.error("Failed to load MJAnime clips")
                return False
                
            # Analyze scripts
            if not await self.scripts_analyzer.analyze_scripts():
                logger.error("Failed to analyze audio scripts") 
                return False
                
            # Load music
            if not await self.music_manager.load_music_track():
                logger.error("Failed to load music track")
                return False
                
            # Analyze music beats
            if not await self.music_manager.analyze_beats():
                logger.warning("Beat analysis failed, continuing with basic music support")
            
            self.loaded = True
            logger.info("All content assets loaded successfully")
            
            # Cache the loaded content
            self._save_to_cache()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load content database: {e}")
            return False
    
    def find_matching_clips(self, script_name: str, count: int = 5) -> List[ContentMatch]:
        """
        Find clips that match well with a given script
        
        Args:
            script_name: Name of the audio script
            count: Number of clips to return
            
        Returns:
            List of content matches sorted by relevance
        """
        if not self.loaded:
            raise RuntimeError("Content database not loaded. Call load_all_content() first.")
        
        # Get script analysis
        script_analysis = self.scripts_analyzer.get_script_analysis(script_name)
        if not script_analysis:
            logger.warning(f"Script analysis not found for: {script_name}")
            return []
        
        # Get matching emotional categories
        matching_emotions = self.scripts_analyzer.get_matching_clips_emotions(script_name)
        
        # Find clips for each emotion
        candidate_clips = []
        for emotion in matching_emotions:
            clips = self.clips_loader.get_clips_by_emotion(emotion)
            candidate_clips.extend(clips)
        
        # Remove duplicates while preserving order
        seen_ids = set()
        unique_clips = []
        for clip in candidate_clips:
            if clip.id not in seen_ids:
                unique_clips.append(clip)
                seen_ids.add(clip.id)
        
        # Score and rank clips
        matches = []
        for clip in unique_clips:
            relevance_score = self._calculate_relevance_score(clip, script_analysis)
            
            # Find matching emotions and themes
            clip_emotions = set(clip.emotional_tags)
            script_emotions = {script_analysis.primary_emotion}
            matching_emotions_list = list(clip_emotions.intersection(script_emotions))
            
            # For themes, use broader matching
            matching_themes_list = []
            for theme in script_analysis.themes:
                if any(theme_word in ' '.join(clip.tags).lower() 
                       for theme_word in theme.split('_')):
                    matching_themes_list.append(theme)
            
            match = ContentMatch(
                clip=clip,
                script=script_analysis,
                relevance_score=relevance_score,
                matching_emotions=matching_emotions_list,
                matching_themes=matching_themes_list
            )
            matches.append(match)
        
        # Sort by relevance score and return top matches
        matches.sort(key=lambda x: x.relevance_score, reverse=True)
        return matches[:count]
    
    def _calculate_relevance_score(self, clip: ClipMetadata, script: ScriptAnalysis) -> float:
        """Calculate relevance score between a clip and script"""
        score = 0.0
        
        # Emotion matching (highest weight)
        if script.primary_emotion in clip.emotional_tags:
            score += 0.5
        
        # Theme matching
        clip_text = ' '.join(clip.tags).lower()
        for theme in script.themes:
            theme_words = theme.replace('_', ' ').split()
            if any(word in clip_text for word in theme_words):
                score += 0.2
        
        # Tag matching
        script_keywords = [tag.lower() for tag in script.raw_features.get('emotional_tags', [])]
        clip_tags = [tag.lower() for tag in clip.tags]
        
        matching_tags = sum(1 for keyword in script_keywords 
                           if any(keyword in tag for tag in clip_tags))
        score += matching_tags * 0.1
        
        # Lighting preference based on emotion
        if script.primary_emotion == 'anxiety' and clip.lighting_type in ['dramatic', 'dark']:
            score += 0.15
        elif script.primary_emotion == 'peace' and clip.lighting_type in ['natural', 'soft']:
            score += 0.15
        
        # Movement preference
        if script.emotional_intensity > 0.7 and clip.movement_type in ['dynamic', 'moving']:
            score += 0.1
        elif script.emotional_intensity < 0.5 and clip.movement_type == 'static':
            score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def generate_unique_sequence(self, script_name: str, sequence_length: int = 3) -> List[ClipMetadata]:
        """
        Generate a unique sequence of clips for a script
        
        Args:
            script_name: Name of the audio script
            sequence_length: Number of clips in sequence
            
        Returns:
            List of clips forming a unique sequence
        """
        if not self.loaded:
            raise RuntimeError("Content database not loaded. Call load_all_content() first.")
        
        # Get more potential clips than needed for variety
        matches = self.find_matching_clips(script_name, count=sequence_length * 3)
        
        if len(matches) < sequence_length:
            logger.warning(f"Only {len(matches)} clips available for {script_name}, need {sequence_length}")
            return [match.clip for match in matches]
        
        # Select clips ensuring variety in visual characteristics
        selected_clips = []
        used_shot_types = set()
        used_lighting = set()
        
        for match in matches:
            clip = match.clip
            
            # Prefer clips with different visual characteristics
            if (len(selected_clips) < sequence_length and
                (clip.shot_type not in used_shot_types or len(selected_clips) == 0) and
                (clip.lighting_type not in used_lighting or len(selected_clips) == 0)):
                
                selected_clips.append(clip)
                used_shot_types.add(clip.shot_type)
                used_lighting.add(clip.lighting_type)
        
        # Fill remaining slots if needed
        while len(selected_clips) < sequence_length and len(selected_clips) < len(matches):
            for match in matches:
                if match.clip not in selected_clips:
                    selected_clips.append(match.clip)
                    if len(selected_clips) >= sequence_length:
                        break
        
        return selected_clips[:sequence_length]
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the content database"""
        if not self.loaded:
            raise RuntimeError("Content database not loaded. Call load_all_content() first.")
        
        clips_stats = self.clips_loader.get_clip_stats()
        scripts_stats = self.scripts_analyzer.get_analysis_stats()
        music_stats = self.music_manager.get_track_info()
        
        return {
            'clips': clips_stats,
            'scripts': scripts_stats,
            'music': music_stats,
            'total_possible_combinations': clips_stats['total_clips'] * scripts_stats['total_scripts'],
            'loaded_timestamp': time.time()
        }
    
    def search_content(self, query: str, content_type: str = 'all') -> Dict[str, List[Any]]:
        """
        Search across all content types
        
        Args:
            query: Search query
            content_type: 'clips', 'scripts', 'music', or 'all'
            
        Returns:
            Dictionary with search results by content type
        """
        if not self.loaded:
            raise RuntimeError("Content database not loaded. Call load_all_content() first.")
        
        results = {'clips': [], 'scripts': [], 'music': []}
        query_lower = query.lower()
        
        if content_type in ['clips', 'all']:
            # Search clips by tags
            for clip in self.clips_loader.clips.values():
                if any(query_lower in tag.lower() for tag in clip.tags):
                    results['clips'].append(clip)
        
        if content_type in ['scripts', 'all']:
            # Search scripts by filename and themes
            for script in self.scripts_analyzer.analyses.values():
                if (query_lower in script.filename.lower() or
                    any(query_lower in theme.lower() for theme in script.themes) or
                    any(query_lower in tag.lower() for tag in script.emotional_tags)):
                    results['scripts'].append(script)
        
        if content_type in ['music', 'all']:
            # Search music (basic filename search for now)
            music_info = self.music_manager.get_track_info()
            if music_info and query_lower in music_info.get('filename', '').lower():
                results['music'].append(music_info)
        
        return results
    
    def _save_to_cache(self):
        """Save database metadata to cache"""
        try:
            cache_data = {
                'loaded_timestamp': time.time(),
                'clips_fingerprint': self.clips_loader.create_content_fingerprint(),
                'scripts_fingerprint': self.scripts_analyzer.create_content_fingerprint(),
                'music_file': self.music_manager.track_info.filepath if self.music_manager.track_info else None
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _load_from_cache(self) -> bool:
        """Load database metadata from cache (for future cache validation)"""
        try:
            if not self.cache_file.exists():
                return False
                
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Basic cache validation could be implemented here
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return False