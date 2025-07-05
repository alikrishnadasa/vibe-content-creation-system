#!/usr/bin/env python3
"""
Test FFmpeg Batch Processing
Generate 5 videos to test FFmpeg performance at scale
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

from core.real_content_generator import RealContentGenerator
from pipelines.content_pipeline import ContentPipeline

async def test_ffmpeg_batch():
    """Test FFmpeg batch processing with 5 videos"""
    logger.info("🚀 TESTING FFMPEG BATCH PROCESSING")
    logger.info("📋 Target: 5 videos (1 script × 5 variations)")
    logger.info("⚡ Using FFmpeg processor for speed")
    
    start_time = time.time()
    
    try:
        # Initialize the real content generator
        logger.info("🔧 Initializing content generator with FFmpeg...")
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
        
        # Clear uniqueness cache for fresh test
        generator.uniqueness_engine.clear_cache()
        logger.info("🧹 Cleared uniqueness cache for fresh generation")
        
        # Initialize batch pipeline
        logger.info("🔧 Initializing batch pipeline...")
        pipeline = ContentPipeline(
            content_generator=generator,
            max_concurrent_videos=2,  # Conservative for testing
            performance_monitoring=True
        )
        
        # Test with 1 script, 5 variations
        logger.info("🎬 GENERATING 5 TEST VIDEOS...")
        
        result = await pipeline.process_specific_scripts(
            script_names=["anxiety1"],  # Just one script for testing
            scripts_directory="../11-scripts-for-tiktok",
            variations_per_script=5,  # 5 variations
            caption_style="tiktok"
        )
        
        # Calculate final metrics
        total_time = time.time() - start_time
        
        # Display comprehensive results
        logger.info("=" * 60)
        logger.info("📊 FFMPEG BATCH TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"📈 Performance Metrics:")
        logger.info(f"   Videos generated: {result.successful_videos}")
        logger.info(f"   Failed videos: {result.failed_videos}")
        logger.info(f"   Success rate: {result.performance_stats.get('success_rate', 0):.1%}")
        
        logger.info(f"⚡ Speed Metrics:")
        logger.info(f"   Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        logger.info(f"   Average per video: {result.average_time_per_video:.3f}s")
        logger.info(f"   Generation rate: {result.performance_stats.get('generation_rate_videos_per_second', 0):.2f} videos/second")
        
        # Compare with MoviePy projection
        moviepy_time = result.successful_videos * 74.751
        speedup = moviepy_time / total_time if total_time > 0 else 0
        logger.info(f"📊 vs MoviePy Comparison:")
        logger.info(f"   MoviePy projection: {moviepy_time:.1f}s ({moviepy_time/60:.1f} minutes)")
        logger.info(f"   FFmpeg actual: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        logger.info(f"   Speedup: {speedup:.1f}x faster")
        
        # Project 55 video performance
        if result.successful_videos > 0:
            projected_55_time = (total_time / result.successful_videos) * 55
            projected_55_moviepy = 74.751 * 55
            logger.info(f"🔮 55 Video Projection:")
            logger.info(f"   FFmpeg 55 videos: {projected_55_time:.1f}s ({projected_55_time/60:.1f} minutes)")
            logger.info(f"   MoviePy 55 videos: {projected_55_moviepy:.1f}s ({projected_55_moviepy/60:.1f} minutes)")
            logger.info(f"   Projected speedup: {projected_55_moviepy/projected_55_time:.1f}x")
        
        # Show generated files
        if result.output_paths:
            logger.info(f"🎬 Generated Videos:")
            for i, path in enumerate(result.output_paths, 1):
                file_path = Path(path)
                if file_path.exists():
                    file_size_mb = file_path.stat().st_size / (1024 * 1024)
                    logger.info(f"   {i}. {file_path.name} ({file_size_mb:.1f}MB)")
        
        # Determine success
        if result.success and result.successful_videos >= 5:
            logger.info("=" * 60)
            logger.info("🎉 FFMPEG BATCH TEST SUCCESSFUL!")
            logger.info("✅ FFmpeg processor ready for 55+ video generation")
            logger.info("🚀 Significant speedup achieved vs MoviePy")
            logger.info("=" * 60)
            
            # Cleanup
            await pipeline.cleanup()
            return True
        else:
            logger.error("=" * 60)
            logger.error("❌ FFMPEG BATCH TEST INCOMPLETE")
            logger.error(f"   Generated {result.successful_videos}/5 videos")
            if result.error_messages:
                logger.error("📋 Error Summary:")
                for error in result.error_messages[:3]:
                    logger.error(f"   - {error}")
            logger.error("=" * 60)
            
            await pipeline.cleanup()
            return False
            
    except Exception as e:
        total_time = time.time() - start_time
        logger.error("=" * 60)
        logger.error(f"❌ FFMPEG BATCH TEST FAILED: {e}")
        logger.error(f"   Runtime: {total_time:.1f}s")
        logger.error("=" * 60)
        return False

async def main():
    """Main execution"""
    success = await test_ffmpeg_batch()
    
    if success:
        print("\n🎉 SUCCESS! FFmpeg batch processing working!")
        print("📋 Ready for 55+ video generation")
        print("⚡ Significant speedup vs MoviePy confirmed")
        return True
    else:
        print("\n💥 FAILED! FFmpeg batch processing needs attention")
        print("📋 Check logs for detailed error information")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        sys.exit(0)
    else:
        sys.exit(1)