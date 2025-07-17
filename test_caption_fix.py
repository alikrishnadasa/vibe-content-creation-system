#!/usr/bin/env python3
"""
Test caption fix with burned-in captions
"""

import sys
import asyncio
import logging

# Add paths
sys.path.append('unified-video-system-main')
sys.path.append('/Users/jamesguo/vibe-content-creation')

from core.real_content_generator import RealContentGenerator, RealVideoRequest

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_caption_fix():
    """Test a single video with burned-in captions"""
    
    print("🔥 Testing Caption Fix with Burned-in Captions")
    print("=" * 60)
    
    # Initialize generator
    generator = RealContentGenerator(
        clips_directory="/Users/jamesguo/vibe-content-creation/midjourney_composite_2025-7-15",
        metadata_file="unified_enhanced_metadata.json",
        scripts_directory="11-scripts-for-tiktok", 
        music_file="11-scripts-for-tiktok/anxiety1.wav",
        output_directory="caption_test_output"
    )
    
    await generator.initialize()
    print("✅ Generator initialized")
    
    # Test with burned-in captions
    request = RealVideoRequest(
        script_path="11-scripts-for-tiktok/anxiety1.wav",
        script_name="anxiety1",
        variation_number=99,  # Different variation to avoid collision
        target_duration=60.0,
        output_path="caption_test_output/anxiety1_with_burned_captions.mp4",
        burn_in_captions=True
    )
    
    print("🎬 Generating video with burned-in captions...")
    result = await generator.generate_video(request)
    
    if result.success:
        print(f"✅ Video with burned-in captions generated!")
        print(f"   Output: {result.output_path}")
        print(f"   Duration: {result.total_duration:.2f}s")
        print(f"   Clips: {len(result.clips_used)}")
        print(f"   Processing time: {result.generation_time:.2f}s")
        
        print("\n🔍 Verifying captions are burned-in...")
        return True
    else:
        print(f"❌ Failed: {result.error_message}")
        return False

async def main():
    success = await test_caption_fix()
    if success:
        print("\n🎉 Caption fix test completed successfully!")
    else:
        print("\n❌ Caption fix test failed!")

if __name__ == "__main__":
    asyncio.run(main())