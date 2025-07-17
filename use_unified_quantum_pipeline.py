#!/usr/bin/env python3
"""
Unified Quantum Pipeline Usage Example
Shows how to generate videos using clips from both MJAnime and midjourney_composite
"""

import asyncio
import sys
from pathlib import Path

# Add unified-video-system-main to path
sys.path.append('unified-video-system-main')

from core.quantum_pipeline import UnifiedQuantumPipeline
from content.content_database import ContentDatabase

async def generate_unified_video():
    """Generate a video using clips from both sources"""
    
    print("🚀 Initializing Unified Quantum Pipeline...")
    
    # Initialize the quantum pipeline
    pipeline = UnifiedQuantumPipeline()
    
    # Initialize content database with unified metadata
    content_db = ContentDatabase(
        clips_directory="",  # Not used with unified metadata
        metadata_file="unified_clips_metadata.json",
        scripts_directory="unified-video-system-main/scripts",
        music_file="unified-video-system-main/music/Beanie (Slowed).mp3",
        use_unified_metadata=True
    )
    
    # Load all content
    print("📚 Loading unified content database...")
    success = await content_db.load_all_content()
    if not success:
        print("❌ Failed to load content database")
        return False
    
    print("✅ Content database loaded successfully")
    
    # Show available sources
    source_stats = content_db.clips_loader.get_source_stats()
    print(f"📊 Available sources: {source_stats}")
    
    # Get sample clips from both sources
    mjanime_clips = content_db.clips_loader.get_clips_by_source("mjanime")[:5]
    composite_clips = content_db.clips_loader.get_clips_by_source("midjourney_composite")[:5]
    
    print(f"\n🎬 Sample clips available:")
    print(f"   - MJAnime: {len(mjanime_clips)} sample clips")
    print(f"   - Composite: {len(composite_clips)} sample clips")
    
    # Example: Find clips by emotion across both sources
    print(f"\n🔍 Searching for peaceful clips across all sources...")
    peaceful_clips = content_db.clips_loader.get_clips_by_emotion("peace")
    
    print(f"Found {len(peaceful_clips)} peaceful clips:")
    for clip in peaceful_clips[:3]:
        print(f"   - {clip.filename} ({clip.source_type})")
    
    # Example configuration for video generation
    video_config = {
        "content_name": "unified_peaceful_mixed",
        "script_name": "meditation_script",  # You can add this script
        "mixed_audio": True,
        "total_variations": 1,
        "clips_per_video": 8,
        "prefer_source": None,  # Use clips from any source
        "output_format": "1080x1920"  # Vertical format
    }
    
    print(f"\n🎥 Ready to generate video with configuration:")
    print(f"   - Content: {video_config['content_name']}")
    print(f"   - Clips per video: {video_config['clips_per_video']}")
    print(f"   - Sources: Both MJAnime and Midjourney Composite")
    
    print(f"\n💡 To generate the video, you would call:")
    print(f"   pipeline.generate_video_batch(content_db, video_config)")
    
    print(f"\n✅ Integration complete! You now have access to {sum(source_stats.values())} clips from multiple sources.")
    
    return True

async def main():
    """Main function"""
    success = await generate_unified_video()
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())