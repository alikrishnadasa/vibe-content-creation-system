#!/usr/bin/env python3
"""
Generate 55 Test Videos

Automated execution of the batch processing system to generate
55 test videos (11 scripts × 5 variations) as specified in
REAL_CONTENT_IMPLEMENTATION.md Phase 4.
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

async def generate_55_test_videos():
    """Generate 55 test videos (11 scripts × 5 variations)"""
    logger.info("🚀 STARTING 55 TEST VIDEO GENERATION")
    logger.info("📋 Target: 11 scripts × 5 variations = 55 videos")
    logger.info("🎵 Music: Beanie (Slowed).mp3 universal background")
    logger.info("🎨 Clips: All 84 MJAnime clips (emotion filtering disabled)")
    
    start_time = time.time()
    
    try:
        # Initialize the real content generator
        logger.info("🔧 Initializing content generator with FFmpeg processor...")
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
        logger.info(f"📊 System Status:")
        logger.info(f"   - Clips loaded: {len(generator.content_database.clips_loader.clips)}")
        logger.info(f"   - Music track: {generator.content_database.music_manager.music_file}")
        logger.info(f"   - Scripts directory: {generator.content_database.script_analyzer.scripts_directory}")
        
        # Clear uniqueness cache for fresh test
        generator.uniqueness_engine.clear_cache()
        logger.info("🧹 Cleared uniqueness cache for fresh generation")
        
        # Initialize batch pipeline with optimized settings
        logger.info("🔧 Initializing batch pipeline...")
        pipeline = ContentPipeline(
            content_generator=generator,
            max_concurrent_videos=3,  # Balanced for stability and speed
            performance_monitoring=True
        )
        
        # Generate all 55 test videos
        logger.info("🎬 GENERATING ALL 55 TEST VIDEOS...")
        logger.info("   Phase 4 of REAL_CONTENT_IMPLEMENTATION.md")
        
        result = await pipeline.process_all_scripts(
            scripts_directory="../11-scripts-for-tiktok",
            variations_per_script=5,  # 5 variations per script = 55 total
            caption_style="tiktok"
        )
        
        # Save comprehensive report
        timestamp = int(time.time())
        report_file = f"full_production_report_{timestamp}.json"
        pipeline.save_batch_report(result, report_file)
        
        # Calculate final metrics
        total_time = time.time() - start_time
        
        # Display comprehensive results
        logger.info("=" * 60)
        logger.info("📊 55 TEST VIDEO GENERATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"📈 Production Metrics:")
        logger.info(f"   Total videos targeted: 55")
        logger.info(f"   Total videos generated: {result.total_videos}")
        logger.info(f"   Successful videos: {result.successful_videos}")
        logger.info(f"   Failed videos: {result.failed_videos}")
        logger.info(f"   Success rate: {result.performance_stats.get('success_rate', 0):.1%}")
        
        logger.info(f"⏱️ Performance Metrics:")
        logger.info(f"   Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        logger.info(f"   Average per video: {result.average_time_per_video:.3f}s")
        logger.info(f"   Generation rate: {result.performance_stats.get('generation_rate_videos_per_second', 0):.2f} videos/second")
        logger.info(f"   <0.7s target: {'✅ MET' if result.average_time_per_video < 0.7 else '❌ MISSED'}")
        
        logger.info(f"🎵 Audio Integration:")
        logger.info(f"   Music track: Beanie (Slowed).mp3")
        logger.info(f"   Beat synchronization: Active")
        logger.info(f"   Script + music mixing: Professional levels")
        
        logger.info(f"🎨 Content Details:")
        logger.info(f"   Scripts processed: {result.performance_stats.get('total_scripts_processed', 0)}")
        logger.info(f"   Variations per script: {result.performance_stats.get('variations_per_script', 0)}")
        logger.info(f"   Emotion filtering: Disabled (semantic matching)")
        logger.info(f"   Available clips: 84 MJAnime clips")
        
        # Show sample of generated files
        if result.output_paths:
            logger.info(f"🎬 Generated Video Samples:")
            for i, path in enumerate(result.output_paths[:10], 1):
                file_path = Path(path)
                if file_path.exists():
                    file_size_mb = file_path.stat().st_size / (1024 * 1024)
                    logger.info(f"   {i}. {file_path.name} ({file_size_mb:.1f}MB)")
            
            if len(result.output_paths) > 10:
                logger.info(f"   ... and {len(result.output_paths) - 10} more videos")
        
        logger.info(f"📄 Full report saved: {report_file}")
        
        # Determine success
        if result.success and result.successful_videos >= 55:
            logger.info("=" * 60)
            logger.info("🎉 55 TEST VIDEO GENERATION SUCCESSFUL!")
            logger.info("✅ Phase 4 of REAL_CONTENT_IMPLEMENTATION.md COMPLETE")
            logger.info("🚀 System ready for 1,100+ production video scaling")
            logger.info("=" * 60)
            
            # Cleanup
            await pipeline.cleanup()
            return True
        else:
            logger.error("=" * 60)
            logger.error("❌ 55 TEST VIDEO GENERATION INCOMPLETE")
            logger.error(f"   Generated {result.successful_videos}/55 videos")
            if result.error_messages:
                logger.error("📋 Error Summary:")
                for error in result.error_messages[:5]:  # Show first 5 errors
                    logger.error(f"   - {error}")
            logger.error("=" * 60)
            
            await pipeline.cleanup()
            return False
            
    except Exception as e:
        total_time = time.time() - start_time
        logger.error("=" * 60)
        logger.error(f"❌ 55 TEST VIDEO GENERATION FAILED: {e}")
        logger.error(f"   Runtime: {total_time:.1f}s")
        logger.error("=" * 60)
        return False

async def main():
    """Main execution"""
    success = await generate_55_test_videos()
    
    if success:
        print("\n🎉 SUCCESS! 55 test videos generated successfully!")
        print("📋 Implementation Phase 4 COMPLETE")
        print("🚀 Ready for Phase 5: Integration and Testing")
        return True
    else:
        print("\n💥 FAILED! 55 test video generation needs attention")
        print("📋 Check logs for detailed error information")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        sys.exit(0)
    else:
        sys.exit(1)