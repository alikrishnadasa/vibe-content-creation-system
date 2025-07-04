#!/usr/bin/env python3
"""
Full Production: 55 Video Generation
Complete implementation of 11 scripts × 5 variations = 55 videos
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

async def full_production_run():
    """Execute the complete 55 video production"""
    logger.info("🚀 STARTING FULL PRODUCTION: 55 VIDEOS")
    logger.info("📋 Target: 11 scripts × 5 variations = 55 videos")
    logger.info("⚡ Using optimized FFmpeg processor")
    
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
        
        # Clear uniqueness cache for fresh production run
        generator.uniqueness_engine.clear_cache()
        logger.info("🧹 Cleared uniqueness cache for fresh production")
        
        # Initialize batch pipeline
        logger.info("🔧 Initializing production pipeline...")
        pipeline = ContentPipeline(
            content_generator=generator,
            max_concurrent_videos=3,  # Optimized for production
            performance_monitoring=True
        )
        
        # Get all available scripts
        script_names = [
            "anxiety1", "safe1", "miserable1", "before", "adhd",
            "deadinside", "diewithphone", "phone1", "4", "6", "500friends"
        ]
        
        logger.info(f"📝 Script inventory: {len(script_names)} scripts found")
        for i, script in enumerate(script_names, 1):
            logger.info(f"   {i}. {script}")
        
        # Execute full production
        logger.info("🎬 STARTING FULL 55-VIDEO PRODUCTION...")
        logger.info("=" * 60)
        
        result = await pipeline.process_specific_scripts(
            script_names=script_names,
            scripts_directory="../11-scripts-for-tiktok",
            variations_per_script=5,
            caption_style="tiktok"
        )
        
        # Calculate final metrics
        total_time = time.time() - start_time
        
        # Display comprehensive production results
        logger.info("=" * 60)
        logger.info("📊 FULL PRODUCTION RESULTS")
        logger.info("=" * 60)
        logger.info(f"📈 Production Metrics:")
        logger.info(f"   Videos generated: {result.successful_videos}")
        logger.info(f"   Failed videos: {result.failed_videos}")
        logger.info(f"   Success rate: {result.performance_stats.get('success_rate', 0):.1%}")
        logger.info(f"   Target: 55 videos")
        
        logger.info(f"⚡ Speed Metrics:")
        logger.info(f"   Total production time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        logger.info(f"   Average per video: {result.average_time_per_video:.3f}s")
        logger.info(f"   Generation rate: {result.performance_stats.get('generation_rate_videos_per_second', 0):.2f} videos/second")
        
        # Compare with MoviePy projection
        moviepy_time = result.successful_videos * 74.751
        speedup = moviepy_time / total_time if total_time > 0 else 0
        logger.info(f"📊 vs MoviePy Comparison:")
        logger.info(f"   MoviePy projection: {moviepy_time:.1f}s ({moviepy_time/60:.1f} minutes)")
        logger.info(f"   FFmpeg actual: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        logger.info(f"   Total speedup: {speedup:.1f}x faster")
        
        # Production quality assessment
        if result.successful_videos >= 55:
            logger.info(f"🎯 Production Quality:")
            logger.info(f"   Complete: ✅ All 55 videos generated")
            logger.info(f"   Time efficiency: ✅ {total_time/60:.1f} minutes total")
            logger.info(f"   Speed: ✅ {speedup:.1f}x faster than MoviePy")
        else:
            completion_rate = (result.successful_videos / 55) * 100
            logger.info(f"⚠️ Production Status:")
            logger.info(f"   Completion: {completion_rate:.1f}% ({result.successful_videos}/55)")
            logger.info(f"   Missing: {55 - result.successful_videos} videos")
        
        # Show output statistics
        if result.output_paths:
            total_size_mb = 0
            for path in result.output_paths:
                if Path(path).exists():
                    total_size_mb += Path(path).stat().st_size / (1024 * 1024)
            
            logger.info(f"📁 Output Statistics:")
            logger.info(f"   Files generated: {len(result.output_paths)}")
            logger.info(f"   Total size: {total_size_mb:.1f}MB")
            logger.info(f"   Average file size: {total_size_mb/len(result.output_paths):.1f}MB")
            logger.info(f"   Output directory: output/")
        
        # Determine production success
        if result.success and result.successful_videos >= 55:
            logger.info("=" * 60)
            logger.info("🎉 FULL PRODUCTION SUCCESSFUL!")
            logger.info("✅ All 55 videos generated successfully")
            logger.info("🚀 Massive speedup achieved vs MoviePy")
            logger.info("📦 Production ready for deployment")
            logger.info("=" * 60)
            
            await pipeline.cleanup()
            return True
        else:
            logger.warning("=" * 60)
            logger.warning("⚠️ PRODUCTION INCOMPLETE")
            logger.warning(f"   Generated {result.successful_videos}/55 videos")
            
            if result.error_messages:
                logger.warning("📋 Error Summary:")
                for error in result.error_messages[:5]:
                    logger.warning(f"   - {error}")
                if len(result.error_messages) > 5:
                    logger.warning(f"   ... and {len(result.error_messages) - 5} more errors")
            logger.warning("=" * 60)
            
            await pipeline.cleanup()
            return False
            
    except Exception as e:
        total_time = time.time() - start_time
        logger.error("=" * 60)
        logger.error(f"❌ PRODUCTION FAILED: {e}")
        logger.error(f"   Runtime: {total_time:.1f}s")
        logger.error("=" * 60)
        return False

async def main():
    """Main execution"""
    success = await full_production_run()
    
    if success:
        print("\n🎉 SUCCESS! Full 55-video production complete!")
        print("📦 Videos ready for deployment")
        print("⚡ FFmpeg optimization delivered massive speedup")
        return True
    else:
        print("\n💥 PRODUCTION INCOMPLETE! Check logs for details")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        sys.exit(0)
    else:
        sys.exit(1)