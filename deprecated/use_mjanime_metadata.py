#!/usr/bin/env python3
"""
Example Usage of MJAnime Persistent Metadata

This script demonstrates how to use the persistent metadata file 
stored in your MJAnime folder for content analysis and selection.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter


def load_mjanime_metadata(metadata_path: str = "./MJAnime/mjanime_metadata.json") -> Dict[str, Any]:
    """Load the persistent MJAnime metadata file."""
    with open(metadata_path, 'r') as f:
        return json.load(f)


def analyze_collection(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze the MJAnime collection using the metadata."""
    clips = metadata.get('clips', [])
    
    # Basic statistics
    total_clips = len(clips)
    total_duration = sum(clip.get('duration', 0) for clip in clips)
    total_size_mb = sum(clip.get('file_size_mb', 0) for clip in clips)
    
    # Content analysis
    all_tags = []
    lighting_types = []
    movement_types = []
    shot_types = []
    
    for clip in clips:
        all_tags.extend(clip.get('tags', []))
        shot_analysis = clip.get('shot_analysis', {})
        lighting_types.append(shot_analysis.get('lighting', 'unknown'))
        movement_types.append(shot_analysis.get('camera_movement', 'unknown'))
        shot_types.append(shot_analysis.get('shot_type', 'unknown'))
    
    # Count occurrences
    tag_counts = Counter(all_tags)
    lighting_counts = Counter(lighting_types)
    movement_counts = Counter(movement_types)
    shot_counts = Counter(shot_types)
    
    return {
        'basic_stats': {
            'total_clips': total_clips,
            'total_duration_seconds': total_duration,
            'total_duration_minutes': total_duration / 60,
            'total_size_mb': total_size_mb,
            'average_clip_duration': total_duration / total_clips if total_clips > 0 else 0
        },
        'content_analysis': {
            'top_tags': dict(tag_counts.most_common(10)),
            'lighting_distribution': dict(lighting_counts),
            'movement_distribution': dict(movement_counts),
            'shot_distribution': dict(shot_counts)
        }
    }


def find_clips_by_content(metadata: Dict[str, Any], 
                         search_terms: List[str],
                         lighting: str = None,
                         movement: str = None) -> List[Dict[str, Any]]:
    """Find clips matching specific content criteria."""
    clips = metadata.get('clips', [])
    matching_clips = []
    
    for clip in clips:
        # Check if any search terms match tags
        clip_tags = [tag.lower() for tag in clip.get('tags', [])]
        search_terms_lower = [term.lower() for term in search_terms]
        
        tag_matches = sum(1 for term in search_terms_lower 
                         if any(term in tag for tag in clip_tags))
        
        if tag_matches > 0:
            shot_analysis = clip.get('shot_analysis', {})
            
            # Check lighting filter
            if lighting and shot_analysis.get('lighting') != lighting:
                continue
            
            # Check movement filter
            if movement and shot_analysis.get('camera_movement') != movement:
                continue
            
            matching_clips.append({
                'filename': clip.get('filename'),
                'id': clip.get('id'),
                'tags': clip.get('tags', []),
                'duration': clip.get('duration', 0),
                'lighting': shot_analysis.get('lighting', 'unknown'),
                'movement': shot_analysis.get('camera_movement', 'unknown'),
                'shot_type': shot_analysis.get('shot_type', 'unknown'),
                'match_score': tag_matches / len(search_terms)
            })
    
    # Sort by match score
    matching_clips.sort(key=lambda x: x['match_score'], reverse=True)
    return matching_clips


def get_content_recommendations(metadata: Dict[str, Any], 
                              script_theme: str) -> Dict[str, List[Dict]]:
    """Get content recommendations based on script theme."""
    
    theme_keywords = {
        'meditation': ['meditation', 'peaceful', 'calm', 'cross-legged', 'tranquil'],
        'spiritual': ['spiritual', 'devotional', 'sacred', 'temple', 'prayer'],
        'nature': ['nature', 'water', 'waterfall', 'mountain', 'garden', 'flower'],
        'community': ['group', 'crowd', 'community', 'together', 'procession'],
        'journey': ['walking', 'path', 'journey', 'barefoot', 'travel'],
        'contemplation': ['alone', 'quiet', 'contemplation', 'individual', 'roof']
    }
    
    keywords = theme_keywords.get(script_theme.lower(), [script_theme])
    matches = find_clips_by_content(metadata, keywords)
    
    # Categorize by quality
    high_quality = [clip for clip in matches if clip['match_score'] >= 0.5]
    medium_quality = [clip for clip in matches if 0.2 <= clip['match_score'] < 0.5]
    low_quality = [clip for clip in matches if clip['match_score'] < 0.2]
    
    return {
        'high_confidence': high_quality[:5],
        'medium_confidence': medium_quality[:5],
        'low_confidence': low_quality[:5]
    }


def main():
    """Demonstrate usage of the persistent metadata."""
    
    print("🎬 MJAnime Metadata Analysis")
    print("=" * 50)
    
    # Load metadata
    try:
        metadata = load_mjanime_metadata()
        print(f"✅ Loaded metadata from: ./MJAnime/mjanime_metadata.json")
        
        # Basic analysis
        analysis = analyze_collection(metadata)
        
        print(f"\n📊 Collection Statistics:")
        stats = analysis['basic_stats']
        print(f"  Total clips: {stats['total_clips']}")
        print(f"  Total duration: {stats['total_duration_minutes']:.1f} minutes")
        print(f"  Total size: {stats['total_size_mb']:.1f} MB")
        print(f"  Average clip length: {stats['average_clip_duration']:.1f} seconds")
        
        print(f"\n🏷️  Top Content Tags:")
        for tag, count in analysis['content_analysis']['top_tags'].items():
            print(f"  {tag}: {count}")
        
        print(f"\n💡 Lighting Distribution:")
        for lighting, count in analysis['content_analysis']['lighting_distribution'].items():
            print(f"  {lighting}: {count}")
        
        # Example searches
        print(f"\n🔍 Example Content Searches:")
        
        # Search for meditation content
        meditation_clips = find_clips_by_content(
            metadata, 
            ['meditation', 'peaceful', 'calm'],
            lighting='natural'
        )
        print(f"\nMeditation clips (natural lighting): {len(meditation_clips)}")
        for clip in meditation_clips[:3]:
            print(f"  - {clip['filename'][:50]}... (score: {clip['match_score']:.2f})")
        
        # Search for temple content
        temple_clips = find_clips_by_content(metadata, ['temple', 'sacred'])
        print(f"\nTemple/Sacred clips: {len(temple_clips)}")
        for clip in temple_clips[:3]:
            print(f"  - {clip['filename'][:50]}... (score: {clip['match_score']:.2f})")
        
        # Get recommendations for different themes
        print(f"\n💡 Content Recommendations by Theme:")
        
        themes = ['meditation', 'spiritual', 'nature', 'community']
        for theme in themes:
            recommendations = get_content_recommendations(metadata, theme)
            high_conf = recommendations['high_confidence']
            print(f"\n{theme.title()} theme - High confidence matches: {len(high_conf)}")
            for clip in high_conf[:2]:
                print(f"  - {clip['filename'][:45]}... ({clip['match_score']:.2f})")
        
        print(f"\n✅ Metadata analysis complete!")
        print(f"\nThe persistent metadata file enables:")
        print(f"  • Fast content searching without re-scanning videos")
        print(f"  • Intelligent clip selection based on semantic tags")
        print(f"  • Content filtering by visual characteristics")
        print(f"  • Theme-based recommendations for script matching")
        
    except FileNotFoundError:
        print("❌ Metadata file not found!")
        print("Run: python3 create_mjanime_metadata.py ./MJAnime/fixed --output ./MJAnime/mjanime_metadata.json")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()