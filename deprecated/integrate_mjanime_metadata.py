#!/usr/bin/env python3
"""
MJAnime Metadata Integration

Integrates the persistent MJAnime metadata with your existing unified video system.
This script shows how to use the metadata for intelligent content selection.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add unified video system to path
sys.path.insert(0, str(Path(__file__).parent / "unified-video-system-main"))

try:
    from content.mjanime_loader import MJAnimeLoader, ClipMetadata
except ImportError:
    print("Note: unified-video-system-main not available, using standalone mode")
    MJAnimeLoader = None


class MJAnimeMetadataManager:
    """Manager for MJAnime persistent metadata with integration capabilities."""
    
    def __init__(self, metadata_path: str = "./MJAnime/mjanime_metadata.json"):
        self.metadata_path = Path(metadata_path)
        self.metadata = None
        self.clips_by_id = {}
        self.clips_by_tags = {}
        
    def load_metadata(self) -> bool:
        """Load the persistent metadata file."""
        if not self.metadata_path.exists():
            print(f"❌ Metadata file not found: {self.metadata_path}")
            return False
        
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Index clips for fast lookup
        clips = self.metadata.get('clips', [])
        for clip in clips:
            clip_id = clip.get('id')
            self.clips_by_id[clip_id] = clip
            
            # Index by tags
            for tag in clip.get('tags', []):
                if tag not in self.clips_by_tags:
                    self.clips_by_tags[tag] = []
                self.clips_by_tags[tag].append(clip)
        
        print(f"✅ Loaded {len(clips)} clips from metadata")
        return True
    
    def get_clips_for_script_emotion(self, emotion: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get clips that match a specific emotional tone."""
        emotion_mapping = {
            'anxiety': ['dramatic', 'cliff', 'intense', 'elevated'],
            'peace': ['peaceful', 'meditation', 'calm', 'tranquil'],
            'spiritual': ['spiritual', 'devotional', 'sacred', 'temple'],
            'journey': ['walking', 'barefoot', 'path'],
            'nature': ['nature', 'water', 'garden', 'flower'],
            'community': ['community', 'group', 'crowd']
        }
        
        keywords = emotion_mapping.get(emotion.lower(), [emotion])
        matching_clips = []
        
        for keyword in keywords:
            if keyword in self.clips_by_tags:
                matching_clips.extend(self.clips_by_tags[keyword])
        
        # Remove duplicates and score
        unique_clips = {}
        for clip in matching_clips:
            clip_id = clip['id']
            if clip_id not in unique_clips:
                unique_clips[clip_id] = clip
        
        return list(unique_clips.values())[:limit]
    
    def get_clips_by_visual_style(self, 
                                 lighting: str = None,
                                 movement: str = None,
                                 shot_type: str = None,
                                 limit: int = 10) -> List[Dict[str, Any]]:
        """Get clips matching specific visual characteristics."""
        matching_clips = []
        
        for clip in self.clips_by_id.values():
            shot_analysis = clip.get('shot_analysis', {})
            
            # Check filters
            if lighting and shot_analysis.get('lighting') != lighting:
                continue
            if movement and shot_analysis.get('camera_movement') != movement:
                continue
            if shot_type and shot_analysis.get('shot_type') != shot_type:
                continue
            
            matching_clips.append(clip)
        
        return matching_clips[:limit]
    
    def get_content_variety_selection(self, 
                                    script_themes: List[str],
                                    clips_per_theme: int = 2) -> Dict[str, List[Dict]]:
        """Get a varied selection of clips covering multiple themes."""
        selection = {}
        
        for theme in script_themes:
            theme_clips = self.get_clips_for_script_emotion(theme, clips_per_theme)
            selection[theme] = theme_clips
        
        return selection
    
    def get_temporal_sequence(self, 
                            story_arc: List[str],
                            duration_preference: str = "medium") -> List[Dict[str, Any]]:
        """Get a sequence of clips that follows a story arc."""
        sequence = []
        
        # Duration preferences
        duration_filter = {
            "short": lambda d: d < 4.0,
            "medium": lambda d: 4.0 <= d <= 6.0,
            "long": lambda d: d > 6.0
        }
        
        filter_func = duration_filter.get(duration_preference, lambda d: True)
        
        for story_element in story_arc:
            matching_clips = self.get_clips_for_script_emotion(story_element, 10)
            
            # Filter by duration preference
            suitable_clips = [clip for clip in matching_clips 
                            if filter_func(clip.get('duration', 5.0))]
            
            if suitable_clips:
                sequence.append(suitable_clips[0])  # Take best match
            elif matching_clips:
                sequence.append(matching_clips[0])  # Fallback to any match
        
        return sequence
    
    def create_unified_system_config(self) -> Dict[str, Any]:
        """Create configuration for integration with unified video system."""
        if not self.metadata:
            return {}
        
        # Analyze collection for optimization
        clips = self.metadata.get('clips', [])
        
        # Find most common characteristics
        lighting_counts = {}
        movement_counts = {}
        tag_counts = {}
        
        for clip in clips:
            shot_analysis = clip.get('shot_analysis', {})
            lighting = shot_analysis.get('lighting', 'natural')
            movement = shot_analysis.get('camera_movement', 'static')
            
            lighting_counts[lighting] = lighting_counts.get(lighting, 0) + 1
            movement_counts[movement] = movement_counts.get(movement, 0) + 1
            
            for tag in clip.get('tags', []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Get dominant characteristics
        dominant_lighting = max(lighting_counts, key=lighting_counts.get)
        dominant_movement = max(movement_counts, key=movement_counts.get)
        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'collection_stats': {
                'total_clips': len(clips),
                'total_duration': sum(clip.get('duration', 0) for clip in clips),
                'dominant_lighting': dominant_lighting,
                'dominant_movement': dominant_movement,
                'top_content_tags': [tag for tag, count in top_tags]
            },
            'optimization_suggestions': {
                'preferred_lighting': dominant_lighting,
                'preferred_movement': dominant_movement,
                'content_strengths': [tag for tag, count in top_tags[:5]]
            },
            'integration_settings': {
                'metadata_path': str(self.metadata_path),
                'clips_directory': self.metadata['metadata_info']['clips_directory'],
                'last_updated': self.metadata['metadata_info']['created_at']
            }
        }


def demo_integration():
    """Demonstrate integration with your video system."""
    print("🔧 MJAnime Metadata Integration Demo")
    print("=" * 50)
    
    # Initialize metadata manager
    manager = MJAnimeMetadataManager()
    
    if not manager.load_metadata():
        return
    
    # Demo 1: Get clips for different emotional tones
    print("\n1️⃣ Getting clips by emotional tone:")
    emotions = ['peace', 'spiritual', 'nature']
    
    for emotion in emotions:
        clips = manager.get_clips_for_script_emotion(emotion, 3)
        print(f"\n{emotion.title()} clips ({len(clips)} found):")
        for i, clip in enumerate(clips, 1):
            filename = clip['filename'][:45] + "..." if len(clip['filename']) > 45 else clip['filename']
            print(f"  {i}. {filename}")
    
    # Demo 2: Visual style filtering
    print(f"\n2️⃣ Visual style filtering:")
    
    natural_static = manager.get_clips_by_visual_style(
        lighting="natural", 
        movement="static", 
        limit=3
    )
    print(f"\nNatural lighting + Static movement: {len(natural_static)} clips")
    
    dramatic_dynamic = manager.get_clips_by_visual_style(
        lighting="dramatic",
        movement="dynamic",
        limit=3
    )
    print(f"Dramatic lighting + Dynamic movement: {len(dramatic_dynamic)} clips")
    
    # Demo 3: Story arc sequence
    print(f"\n3️⃣ Story arc sequence:")
    story_arc = ['peace', 'journey', 'spiritual', 'community']
    sequence = manager.get_temporal_sequence(story_arc, "medium")
    
    print(f"\nStory sequence for arc: {' → '.join(story_arc)}")
    for i, clip in enumerate(sequence, 1):
        filename = clip['filename'][:40] + "..." if len(clip['filename']) > 40 else clip['filename']
        duration = clip.get('duration', 0)
        print(f"  {i}. {filename} ({duration:.1f}s)")
    
    # Demo 4: Content variety selection
    print(f"\n4️⃣ Content variety selection:")
    themes = ['spiritual', 'nature', 'peace']
    variety_selection = manager.get_content_variety_selection(themes, 2)
    
    for theme, clips in variety_selection.items():
        print(f"\n{theme.title()} theme ({len(clips)} clips):")
        for clip in clips:
            filename = clip['filename'][:40] + "..." if len(clip['filename']) > 40 else clip['filename']
            print(f"  - {filename}")
    
    # Demo 5: Integration configuration
    print(f"\n5️⃣ Integration configuration:")
    config = manager.create_unified_system_config()
    
    print(f"\nCollection overview:")
    stats = config['collection_stats']
    print(f"  Total clips: {stats['total_clips']}")
    print(f"  Total duration: {stats['total_duration']:.1f} seconds")
    print(f"  Dominant style: {stats['dominant_lighting']} lighting, {stats['dominant_movement']} movement")
    print(f"  Top content: {', '.join(stats['top_content_tags'][:5])}")
    
    print(f"\nOptimization suggestions:")
    opt = config['optimization_suggestions']
    print(f"  Best for: {', '.join(opt['content_strengths'])}")
    print(f"  Preferred style: {opt['preferred_lighting']} lighting with {opt['preferred_movement']} movement")
    
    print(f"\n✅ Integration demo complete!")
    print(f"\nThis persistent metadata enables:")
    print(f"  • Fast emotional tone matching for scripts")
    print(f"  • Visual style consistency across video generation")
    print(f"  • Intelligent content sequencing for story arcs")
    print(f"  • Optimized selection based on collection strengths")


if __name__ == "__main__":
    demo_integration()