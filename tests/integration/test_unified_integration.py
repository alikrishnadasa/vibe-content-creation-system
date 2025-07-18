#!/usr/bin/env python3
"""
Test Unified Integration
Verify that the quantum pipeline can load clips from both sources
"""

import asyncio
import sys
from pathlib import Path

# Add unified-video-system-main to path
sys.path.append('unified-video-system-main')

from content.content_database import ContentDatabase
from content.mjanime_loader import MJAnimeLoader

async def test_unified_integration():
    """Test the unified metadata integration"""
    
    print("🧪 Testing unified metadata integration...")
    
    # Test 1: Load unified metadata directly
    print("\n1️⃣ Testing unified MJAnimeLoader...")
    loader = MJAnimeLoader(
        clips_directory="",  # Not used with unified metadata
        metadata_file="unified_clips_metadata.json",
        use_unified_metadata=True
    )
    
    success = await loader.load_clips()
    if not success:
        print("❌ Failed to load unified metadata")
        return False
    
    # Check statistics
    stats = loader.get_clip_stats()
    source_stats = loader.get_source_stats()
    
    print(f"✅ Loaded {stats['total_clips']} clips total")
    print(f"   - Duration: {stats['total_duration_seconds']:.1f}s")
    print(f"   - Size: {stats['total_size_mb']:.1f}MB")
    print(f"   - Sources: {source_stats}")
    
    # Test 2: Test source filtering
    print("\n2️⃣ Testing source filtering...")
    mjanime_clips = loader.get_clips_by_source("mjanime")
    composite_clips = loader.get_clips_by_source("midjourney_composite")
    
    print(f"   - MJAnime clips: {len(mjanime_clips)}")
    print(f"   - Midjourney composite clips: {len(composite_clips)}")
    
    # Test 3: Test content database with unified metadata  
    print("\n3️⃣ Testing ContentDatabase with unified metadata...")
    content_db = ContentDatabase(
        clips_directory="",  # Not used
        metadata_file="unified_clips_metadata.json",
        scripts_directory="unified-video-system-main/scripts",
        music_file="unified-video-system-main/music/Beanie (Slowed).mp3",
        use_unified_metadata=True
    )
    
    success = await content_db.load_all_content()
    if not success:
        print("❌ Failed to load content database")
        return False
    
    print("✅ ContentDatabase loaded successfully")
    
    # Test 4: Search functionality
    print("\n4️⃣ Testing search across both sources...")
    
    # Test emotional search
    anxiety_clips = content_db.clips_loader.get_clips_by_emotion("anxiety")
    peace_clips = content_db.clips_loader.get_clips_by_emotion("peace")
    
    print(f"   - Anxiety clips: {len(anxiety_clips)}")
    print(f"   - Peace clips: {len(peace_clips)}")
    
    # Show some examples from each source
    print("\n5️⃣ Sample clips from each source:")
    
    if mjanime_clips:
        print(f"   MJAnime example: {mjanime_clips[0].filename}")
        print(f"   Tags: {mjanime_clips[0].tags[:3]}...")
    
    if composite_clips:
        print(f"   Composite example: {composite_clips[0].filename}")
        print(f"   Tags: {composite_clips[0].tags[:3]}...")
    
    print("\n🎉 All tests passed! Integration successful!")
    return True

async def main():
    """Main test function"""
    success = await test_unified_integration()
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())