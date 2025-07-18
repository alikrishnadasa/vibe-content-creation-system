#!/usr/bin/env python3
"""
Generate One Video Per Script
Create 1 video for each of the 11 scripts using Whisper word-by-word captions
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

async def generate_one_per_script():
    """Generate 1 video for each of the 11 scripts"""
    logger.info("🎬 GENERATING 1 VIDEO PER SCRIPT")
    logger.info("📋 Target: 11 videos (1 per script) with Whisper word-by-word captions")
    
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
        
        # Clear uniqueness cache for fresh generation
        generator.uniqueness_engine.clear_cache()
        logger.info("🧹 Cleared uniqueness cache")
        
        # Get all script names
        script_names = [
            "anxiety1", "safe1", "miserable1", "before", "adhd",
            "deadinside", "diewithphone", "phone1", "4", "6", "500friends"
        ]
        
        logger.info(f"📝 Scripts to process: {len(script_names)}")
        for i, script in enumerate(script_names, 1):
            logger.info(f"   {i}. {script}")
        
        # Generate videos
        results = []
        successful = 0
        failed = 0
        
        for i, script_name in enumerate(script_names, 1):
            logger.info(f"🎬 Generating video {i}/{len(script_names)}: {script_name}")
            
            request = RealVideoRequest(
                script_path=f"../11-scripts-for-tiktok/{script_name}.wav",
                script_name=script_name,
                variation_number=1,  # First variation for each script
                caption_style="tiktok",  # Word-by-word with uppercase
                music_sync=True
            )
            
            try:
                video_start = time.time()
                result = await generator.generate_video(request)
                video_time = time.time() - video_start
                
                if result.success:
                    successful += 1
                    logger.info(f"✅ {script_name}: {video_time:.2f}s - {Path(result.output_path).name}")
                    
                    # Get word count from cache for this script
                    word_count = len(generator.caption_cache.get_captions_for_script(script_name))
                    logger.info(f"   📝 Words: {word_count}, Duration: {result.total_duration:.1f}s")
                else:
                    failed += 1
                    logger.error(f"❌ {script_name}: {result.error_message}")
                
                results.append(result)
                
            except Exception as e:
                failed += 1
                logger.error(f"💥 {script_name}: Exception - {e}")
        
        # Calculate final metrics
        total_time = time.time() - start_time
        
        # Display comprehensive results
        logger.info("=" * 60)
        logger.info("📊 ONE-PER-SCRIPT GENERATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"📈 Generation Metrics:")
        logger.info(f"   Videos generated: {successful}")
        logger.info(f"   Failed videos: {failed}")
        logger.info(f"   Success rate: {successful/len(script_names)*100:.1f}%")
        logger.info(f"   Target: {len(script_names)} videos")
        
        logger.info(f"⚡ Speed Metrics:")
        logger.info(f"   Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        if successful > 0:
            avg_time = sum(r.generation_time for r in results if r.success) / successful
            logger.info(f"   Average per video: {avg_time:.3f}s")
        
        # Show generated files
        successful_results = [r for r in results if r.success]
        if successful_results:
            logger.info(f"🎬 Generated Videos:")
            total_size_mb = 0
            for i, result in enumerate(successful_results, 1):
                file_path = Path(result.output_path)
                if file_path.exists():
                    file_size_mb = file_path.stat().st_size / (1024 * 1024)
                    total_size_mb += file_size_mb
                    script_name = result.clips_used[0] if result.clips_used else "unknown"
                    logger.info(f"   {i}. {file_path.name} ({file_size_mb:.1f}MB)")
            
            logger.info(f"📁 Total size: {total_size_mb:.1f}MB")
            logger.info(f"📈 Average file size: {total_size_mb/len(successful_results):.1f}MB")
        
        # Show caption statistics
        if generator.caption_cache.cache:
            logger.info(f"📝 Caption Statistics:")
            total_words = 0
            for script_name in script_names:
                if script_name in generator.caption_cache.cache:
                    words = len(generator.caption_cache.cache[script_name].word_timings)
                    total_words += words
                    logger.info(f"   {script_name}: {words} words")
            logger.info(f"   Total words across all scripts: {total_words}")
        
        # Determine success
        if successful == len(script_names):
            logger.info("=" * 60)
            logger.info("🎉 ONE-PER-SCRIPT GENERATION SUCCESSFUL!")
            logger.info("✅ All 11 videos generated with Whisper word-by-word captions")
            logger.info("📝 Each word displays exactly when spoken in the audio")
            logger.info("🚀 Ready for production or further testing")
            logger.info("=" * 60)
            return True
        else:
            logger.warning("=" * 60)
            logger.warning("⚠️ GENERATION INCOMPLETE")
            logger.warning(f"   Generated {successful}/{len(script_names)} videos")
            logger.warning(f"   Failed: {failed} videos")
            logger.warning("=" * 60)
            return False
            
    except Exception as e:
        total_time = time.time() - start_time
        logger.error("=" * 60)
        logger.error(f"❌ ONE-PER-SCRIPT GENERATION FAILED: {e}")
        logger.error(f"   Runtime: {total_time:.1f}s")
        logger.error("=" * 60)
        return False

async def main():
    """Main execution"""
    success = await generate_one_per_script()
    
    if success:
        print("\n🎉 SUCCESS! Generated 1 video for each script!")
        print("📝 All videos use precise Whisper word-by-word captions")
        print("🎬 Ready for review and production scaling")
        return True
    else:
        print("\n💥 INCOMPLETE! Some videos failed to generate")
        print("📋 Check logs for detailed error information")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        sys.exit(0)
    else:
        sys.exit(1)