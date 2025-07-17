"""
MJAnime Video Loader

Loads and indexes all 84 MJAnime clips with comprehensive metadata analysis.
Provides semantic search capabilities and efficient GPU memory management.
"""

import json
import os
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class ClipMetadata:
    """Metadata for a single video clip (MJAnime or other sources)"""
    id: str
    filename: str
    filepath: str
    tags: List[str]
    duration: float
    resolution: str
    fps: float
    file_size_mb: float
    shot_analysis: Dict[str, Any]
    created_at: str
    source_type: str = "mjanime"  # Track source: "mjanime" or "midjourney_composite"
    
    # Computed properties for content selection (with defaults)
    emotional_tags: List[str] = None
    lighting_type: str = ""
    movement_type: str = ""
    shot_type: str = ""
    
    def __post_init__(self):
        """Extract key properties for intelligent selection"""
        if self.emotional_tags is None:
            self.emotional_tags = []
            
        self.lighting_type = self.shot_analysis.get('lighting', 'natural')
        self.movement_type = self.shot_analysis.get('camera_movement', 'static')
        self.shot_type = self.shot_analysis.get('shot_type', 'medium_shot')
        
        # Categorize tags by emotional content for script matching
        self.emotional_tags = self._categorize_emotional_content()
    
    def _categorize_emotional_content(self) -> List[str]:
        """Categorize clip tags into emotional categories for content selection"""
        # Keywords matching 11 audio scripts
        anxiety_keywords = ['dramatic', 'shadows', 'dark', 'tense', 'conflicted', 'struggle', 'cliff', 'intense']
        peace_keywords = ['serene', 'meditation', 'calm', 'tranquil', 'lotus', 'gentle', 'floating', 'quiet']
        seeking_keywords = ['contemplative', 'introspective', 'searching', 'journey', 'path', 'walking']
        awakening_keywords = ['bright', 'enlightenment', 'realization', 'temple', 'spiritual', 'sacred']
        safe_keywords = ['cozy', 'warm', 'comfortable', 'shelter', 'interior', 'home', 'protected']
        social_keywords = ['group', 'crowd', 'gathering', 'festival', 'celebration', 'community']
        isolated_keywords = ['alone', 'single', 'lone', 'individual', 'solitary']
        
        emotions = []
        tag_text = ' '.join(self.tags).lower()
        
        if any(keyword in tag_text for keyword in anxiety_keywords):
            emotions.append('anxiety')
        if any(keyword in tag_text for keyword in peace_keywords):
            emotions.append('peace')
        if any(keyword in tag_text for keyword in seeking_keywords):
            emotions.append('seeking')
        if any(keyword in tag_text for keyword in awakening_keywords):
            emotions.append('awakening')
        if any(keyword in tag_text for keyword in safe_keywords):
            emotions.append('safe')
        if any(keyword in tag_text for keyword in social_keywords):
            emotions.append('social')
        if any(keyword in tag_text for keyword in isolated_keywords):
            emotions.append('isolated')
            
        return emotions if emotions else ['neutral']

class MJAnimeLoader:
    """Loads and manages video clips from multiple sources with intelligent indexing"""
    
    def __init__(self, clips_directory: str, metadata_file: str, use_unified_metadata: bool = False):
        """
        Initialize the clip loader
        
        Args:
            clips_directory: Path to directory containing clips (legacy parameter)
            metadata_file: Path to metadata JSON file
            use_unified_metadata: If True, load from unified metadata with multiple sources
        """
        self.clips_directory = Path(clips_directory)
        self.metadata_file = Path(metadata_file)
        self.use_unified_metadata = use_unified_metadata
        self.clips: Dict[str, ClipMetadata] = {}
        self.loaded = False
        
    async def load_clips(self) -> bool:
        """
        Load all clips and their metadata
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load metadata from JSON
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            clips_data = metadata.get('clips', [])
            
            if self.use_unified_metadata:
                logger.info(f"Loading {len(clips_data)} clips from unified metadata...")
                loaded_count = self._load_unified_clips(clips_data)
            else:
                logger.info(f"Loading {len(clips_data)} clips from single source...")
                loaded_count = self._load_single_source_clips(clips_data)
            
            self.loaded = True
            logger.info(f"✅ Successfully loaded {loaded_count} clips")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load clips: {e}")
            return False
    
    def _load_unified_clips(self, clips_data: List[Dict]) -> int:
        """Load clips from unified metadata with multiple sources"""
        loaded_count = 0
        for clip_data in clips_data:
            # Use file_path from unified metadata which includes full path
            clip_path = Path(clip_data.get('file_path', ''))
            
            # Verify file exists
            if not clip_path.exists():
                logger.warning(f"Clip file not found: {clip_path}")
                continue
            
            # Create clip metadata
            clip_metadata = ClipMetadata(
                id=clip_data['id'],
                filename=clip_data['filename'],
                filepath=str(clip_path),
                tags=clip_data.get('tags', []),
                duration=clip_data.get('duration', 5.21),
                resolution=clip_data.get('resolution', '1080x1936'),
                fps=clip_data.get('fps', 24.0),
                file_size_mb=clip_data.get('file_size_mb', 0),
                shot_analysis=clip_data.get('shot_analysis', {}),
                created_at=clip_data.get('created_at', ''),
                source_type=clip_data.get('source_type', 'mjanime')
            )
            
            self.clips[clip_metadata.id] = clip_metadata
            loaded_count += 1
        
        return loaded_count
    
    def _load_single_source_clips(self, clips_data: List[Dict]) -> int:
        """Load clips from single source (legacy behavior)"""
        loaded_count = 0
        for clip_data in clips_data:
            clip_path = self.clips_directory / clip_data['filename']
            
            # Verify file exists
            if not clip_path.exists():
                logger.warning(f"Clip file not found: {clip_path}")
                continue
            
            # Create clip metadata
            clip_metadata = ClipMetadata(
                id=clip_data['id'],
                filename=clip_data['filename'],
                filepath=str(clip_path),
                tags=clip_data.get('tags', []),
                duration=clip_data.get('duration', 5.21),
                resolution=clip_data.get('resolution', '1080x1936'),
                fps=clip_data.get('fps', 24.0),
                file_size_mb=clip_data.get('file_size_mb', 0),
                shot_analysis=clip_data.get('shot_analysis', {}),
                created_at=clip_data.get('created_at', ''),
                source_type='mjanime'
            )
            
            self.clips[clip_metadata.id] = clip_metadata
            loaded_count += 1
        
        return loaded_count
    
    def get_clips_by_emotion(self, emotion: str) -> List[ClipMetadata]:
        """
        Get clips that match a specific emotional category
        
        Args:
            emotion: Emotional category ('anxiety', 'peace', 'seeking', 'awakening', 'neutral')
            
        Returns:
            List of matching clip metadata
        """
        if not self.loaded:
            raise RuntimeError("Clips not loaded. Call load_clips() first.")
        
        matching_clips = []
        for clip in self.clips.values():
            if emotion.lower() in clip.emotional_tags:
                matching_clips.append(clip)
        
        return matching_clips
    
    def get_clips_by_lighting(self, lighting_type: str) -> List[ClipMetadata]:
        """Get clips with specific lighting type"""
        if not self.loaded:
            raise RuntimeError("Clips not loaded. Call load_clips() first.")
        
        return [clip for clip in self.clips.values() 
                if clip.lighting_type == lighting_type]
    
    def get_clips_by_movement(self, movement_type: str) -> List[ClipMetadata]:
        """Get clips with specific camera movement"""
        if not self.loaded:
            raise RuntimeError("Clips not loaded. Call load_clips() first.")
        
        return [clip for clip in self.clips.values() 
                if clip.movement_type == movement_type]
    
    def get_clips_by_source(self, source_type: str) -> List[ClipMetadata]:
        """Get clips from specific source"""
        if not self.loaded:
            raise RuntimeError("Clips not loaded. Call load_clips() first.")
        
        return [clip for clip in self.clips.values() 
                if clip.source_type == source_type]
    
    def get_source_stats(self) -> Dict[str, Any]:
        """Get statistics by source type"""
        if not self.loaded:
            raise RuntimeError("Clips not loaded. Call load_clips() first.")
        
        source_counts = {}
        for clip in self.clips.values():
            source = clip.source_type
            source_counts[source] = source_counts.get(source, 0) + 1
        
        return source_counts
    
    def search_clips_by_tags(self, search_tags: List[str]) -> List[Tuple[ClipMetadata, float]]:
        """
        Search clips by tags with relevance scoring
        
        Args:
            search_tags: List of tags to search for
            
        Returns:
            List of (clip, relevance_score) tuples, sorted by relevance
        """
        if not self.loaded:
            raise RuntimeError("Clips not loaded. Call load_clips() first.")
        
        results = []
        search_tags_lower = [tag.lower() for tag in search_tags]
        
        for clip in self.clips.values():
            clip_tags_lower = [tag.lower() for tag in clip.tags]
            
            # Calculate relevance score
            matches = sum(1 for tag in search_tags_lower 
                         if any(tag in clip_tag for clip_tag in clip_tags_lower))
            
            if matches > 0:
                relevance = matches / len(search_tags)
                results.append((clip, relevance))
        
        # Sort by relevance score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def get_random_clips(self, count: int, emotion: Optional[str] = None) -> List[ClipMetadata]:
        """
        Get random clips, optionally filtered by emotion
        
        Args:
            count: Number of clips to return
            emotion: Optional emotional filter
            
        Returns:
            List of random clip metadata
        """
        if not self.loaded:
            raise RuntimeError("Clips not loaded. Call load_clips() first.")
        
        if emotion:
            available_clips = self.get_clips_by_emotion(emotion)
        else:
            available_clips = list(self.clips.values())
        
        if len(available_clips) < count:
            logger.warning(f"Requested {count} clips but only {len(available_clips)} available")
            return available_clips
        
        # Use numpy for deterministic random selection (for testing reproducibility)
        np.random.seed(42)  # Fixed seed for consistent results
        indices = np.random.choice(len(available_clips), count, replace=False)
        
        return [available_clips[i] for i in indices]
    
    def get_clip_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded clips"""
        if not self.loaded:
            raise RuntimeError("Clips not loaded. Call load_clips() first.")
        
        total_clips = len(self.clips)
        total_duration = sum(clip.duration for clip in self.clips.values())
        total_size_mb = sum(clip.file_size_mb for clip in self.clips.values())
        
        # Emotional distribution
        emotion_counts = {}
        for clip in self.clips.values():
            for emotion in clip.emotional_tags:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Lighting distribution
        lighting_counts = {}
        for clip in self.clips.values():
            lighting = clip.lighting_type
            lighting_counts[lighting] = lighting_counts.get(lighting, 0) + 1
        
        return {
            'total_clips': total_clips,
            'total_duration_seconds': total_duration,
            'total_size_mb': total_size_mb,
            'average_duration': total_duration / total_clips if total_clips > 0 else 0,
            'emotion_distribution': emotion_counts,
            'lighting_distribution': lighting_counts,
            'resolutions': list(set(clip.resolution for clip in self.clips.values())),
            'fps_values': list(set(clip.fps for clip in self.clips.values()))
        }
    
    def create_content_fingerprint(self) -> str:
        """Create fingerprint of all loaded content for cache validation"""
        if not self.loaded:
            raise RuntimeError("Clips not loaded. Call load_clips() first.")
        
        # Create hash of all clip IDs and metadata modification times
        content_data = []
        for clip in sorted(self.clips.values(), key=lambda x: x.id):
            content_data.append(f"{clip.id}:{clip.created_at}:{clip.file_size_mb}")
        
        content_string = '|'.join(content_data)
        return hashlib.sha256(content_string.encode()).hexdigest()[:16]