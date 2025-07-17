#!/usr/bin/env python3
"""
Create Unified Test Video
Generate one complete video using the unified metadata with clips from both sources
"""

import logging
import asyncio
import sys
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project paths
sys.path.append('unified-video-system-main')

from core.real_content_generator import RealContentGenerator, RealVideoRequest

async def create_unified_test_video():
    """Create a test video using unified metadata"""
    logger.info("🎬 Creating unified test video with clips from both sources")
    
    try:
        # Initialize with unified metadata
        logger.info("Initializing content generator with unified metadata...")
        generator = RealContentGenerator(
            clips_directory="",  # Not used with unified metadata
            metadata_file="unified_clips_metadata.json",
            scripts_directory="11-scripts-for-tiktok",  # Use actual scripts directory
            music_file="unified-video-system-main/music/Beanie (Slowed).mp3"
        )
        
        # Modify the content database to use unified metadata
        generator.content_database.clips_loader.use_unified_metadata = True
        
        success = await generator.initialize()
        if not success:
            logger.error("❌ Failed to initialize generator")
            return False
        
        logger.info("✅ Generator initialized successfully")
        
        # Show available clips
        stats = generator.content_database.clips_loader.get_clip_stats()
        source_stats = generator.content_database.clips_loader.get_source_stats()
        logger.info(f"📊 Available clips: {source_stats}")
        logger.info(f"   Total: {stats['total_clips']} clips")
        
        # Create test video request
        output_path = "output/unified_test_video.mp4"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Use an existing script
        request = RealVideoRequest(
            script_path="11-scripts-for-tiktok/anxiety1.wav",  # Use existing audio script
            script_name="anxiety1",
            variation_number=1,
            caption_style="default",
            music_sync=True,
            output_path=output_path,
            burn_in_captions=True
        )
        
        logger.info(f"🎥 Generating video: {output_path}")
        start_time = time.time()
        
        # Generate the video
        result = await generator.generate_video(request)
        
        generation_time = time.time() - start_time
        
        if result and result.success:
            logger.info(f"✅ Video generated successfully in {generation_time:.2f}s")
            logger.info(f"📁 Output: {result.output_path}")
            logger.info(f"🎬 Clips used: {len(result.clips_used)}")
            logger.info(f"⏱️ Total duration: {result.total_duration:.1f}s")
            logger.info(f"🎯 Relevance score: {result.relevance_score:.2f}")
            
            # Show clip sources used
            if result.clips_used:
                source_breakdown = {}
                for clip_path in result.clips_used:
                    # Determine source from path
                    if 'MJAnime' in clip_path:
                        source_breakdown['mjanime'] = source_breakdown.get('mjanime', 0) + 1
                    elif 'midjourney_composite' in clip_path:
                        source_breakdown['midjourney_composite'] = source_breakdown.get('midjourney_composite', 0) + 1
                
                if source_breakdown:
                    logger.info(f"🎬 Clips used by source: {source_breakdown}")
            
            return True
        else:
            error_msg = result.error_message if result else 'No result returned'
            logger.error(f"❌ Video generation failed: {error_msg}")
            return False
            
    except Exception as e:
        logger.error(f"💥 Error creating test video: {e}")
        return False

async def main():
    """Main function"""
    success = await create_unified_test_video()
    
    if success:
        print("\n🎉 SUCCESS! Unified test video created!")
        print("📁 Check output/unified_test_video.mp4")
        print("💡 This video was created using clips from both:")
        print("   - MJAnime (spiritual anime content)")
        print("   - Midjourney Composite (artistic AI content)")
    else:
        print("\n💥 FAILED! Could not create unified test video")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())