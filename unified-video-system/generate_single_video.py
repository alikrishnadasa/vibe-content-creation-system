#!/usr/bin/env python3
"""
Generate Single Video
Create one test video with Whisper word-by-word captions
"""

import logging
import asyncio
import sys
from pathlib import Path
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.real_content_generator import RealContentGenerator, RealVideoRequest

async def generate_single_video():
    """Generate one test video with Whisper captions"""
    logger.info("🎬 GENERATING SINGLE TEST VIDEO")
    logger.info("📝 Using Whisper word-by-word captions")
    
    start_time = time.time()
    
    try:
        # Initialize the real content generator
        logger.info("🔧 Initializing content generator...")
        generator = RealContentGenerator(
            clips_directory="../MJAnime",
            metadata_file="../MJAnime/metadata_final_clean_shots.json", 
            scripts_directory="../11-scripts-for-tiktok",
            music_file="music/Beanie (Slowed).mp3",
            output_directory="output"
        )
        
        # Initialize all components
        init_success = await generator.initialize()
        if not init_success:
            logger.error("❌ Failed to initialize generator")
            return False
        
        logger.info("✅ Generator initialized successfully")
        
        # Clear uniqueness cache
        generator.uniqueness_engine.clear_cache()
        
        # Generate one video
        logger.info("🎬 Generating test video...")
        
        request = RealVideoRequest(
            script_path="../11-scripts-for-tiktok/anxiety1.wav",
            script_name="anxiety1",
            variation_number=1,
            caption_style="tiktok",  # Word-by-word with uppercase
            music_sync=True
        )
        
        video_start = time.time()
        result = await generator.generate_video(request)
        video_time = time.time() - video_start
        
        total_time = time.time() - start_time
        
        if result.success:
            logger.info("=" * 60)
            logger.info("🎉 SINGLE VIDEO GENERATION SUCCESSFUL!")
            logger.info("=" * 60)
            logger.info(f"✅ Video: {Path(result.output_path).name}")
            logger.info(f"⚡ Generation time: {video_time:.2f}s")
            logger.info(f"🎯 Total time: {total_time:.2f}s")
            
            # Get caption details
            word_count = len(generator.caption_cache.get_captions_for_script("anxiety1"))
            logger.info(f"📝 Whisper words: {word_count}")
            logger.info(f"🎵 Audio duration: {result.total_duration:.1f}s")
            logger.info(f"🎬 Clips used: {len(result.clips_used)}")
            
            # Check file
            if Path(result.output_path).exists():
                file_size_mb = Path(result.output_path).stat().st_size / (1024 * 1024)
                logger.info(f"📁 File size: {file_size_mb:.1f}MB")
            
            logger.info("=" * 60)
            return True
        else:
            logger.error(f"❌ Video generation failed: {result.error_message}")
            return False
            
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"💥 Generation failed: {e}")
        logger.error(f"   Runtime: {total_time:.1f}s")
        return False

async def main():
    """Main execution"""
    success = await generate_single_video()
    
    if success:
        print("\n🎉 SUCCESS! Single video generated with Whisper captions!")
        return True
    else:
        print("\n💥 FAILED! Video generation encountered an error")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        sys.exit(0)
    else:
        sys.exit(1)