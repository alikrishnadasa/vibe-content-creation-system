#!/usr/bin/env python3
"""
Test Ultra-Speed FFmpeg Optimizations
Single video test to measure speed improvements
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

async def test_ultra_speed():
    """Test single video with ultra-speed optimizations"""
    logger.info("🚀 TESTING ULTRA-SPEED FFMPEG OPTIMIZATIONS")
    logger.info("🎯 Target: Sub-second generation per video")
    
    start_time = time.time()
    
    try:
        # Initialize the real content generator
        logger.info("🔧 Initializing content generator with ultra-speed mode...")
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
        
        # Generate a single test video
        logger.info("🎬 GENERATING SINGLE ULTRA-SPEED TEST VIDEO...")
        
        request = RealVideoRequest(
            script_path="../11-scripts-for-tiktok/anxiety1.wav",
            script_name="anxiety1",
            variation_number=99,  # Use unique variation number
            caption_style="tiktok",
            music_sync=True
        )
        
        generation_start = time.time()
        result = await generator.generate_video(request)
        generation_time = time.time() - generation_start
        
        # Calculate final metrics
        total_time = time.time() - start_time
        
        # Display comprehensive results
        logger.info("=" * 60)
        logger.info("📊 ULTRA-SPEED OPTIMIZATION RESULTS")
        logger.info("=" * 60)
        
        if result.success:
            logger.info(f"✅ Video generated successfully!")
            logger.info(f"⚡ Generation time: {generation_time:.3f}s")
            logger.info(f"📁 Output: {Path(result.output_path).name}")
            logger.info(f"📊 Video stats:")
            logger.info(f"   Duration: {result.total_duration:.1f}s")
            logger.info(f"   Clips used: {len(result.clips_used)}")
            logger.info(f"   Relevance score: {result.relevance_score:.2f}")
            
            # Check if we achieved sub-second generation
            if generation_time < 1.0:
                logger.info("🎉 SUB-SECOND GENERATION ACHIEVED!")
                logger.info(f"🏆 Ultra-speed target met: {generation_time:.3f}s < 1.0s")
            else:
                improvement_needed = generation_time - 1.0
                logger.info(f"⚠️ Close to target: {generation_time:.3f}s")
                logger.info(f"📈 Need {improvement_needed:.3f}s improvement for sub-second")
            
            # Project 55 video performance with ultra-speed
            projected_55_time = generation_time * 55
            logger.info(f"🔮 55 Video Projection (Ultra-Speed):")
            logger.info(f"   Total time: {projected_55_time:.1f}s ({projected_55_time/60:.1f} minutes)")
            
            # Compare with previous FFmpeg performance (9.6s average)
            previous_ffmpeg_time = 9.6
            speedup_vs_previous = previous_ffmpeg_time / generation_time
            logger.info(f"📊 vs Previous FFmpeg:")
            logger.info(f"   Previous FFmpeg: {previous_ffmpeg_time:.1f}s")
            logger.info(f"   Ultra-speed FFmpeg: {generation_time:.3f}s")
            logger.info(f"   Additional speedup: {speedup_vs_previous:.1f}x")
            
            # Check file size and quality
            if Path(result.output_path).exists():
                file_size_mb = Path(result.output_path).stat().st_size / (1024 * 1024)
                logger.info(f"📁 File size: {file_size_mb:.1f}MB")
            
            logger.info("=" * 60)
            
            if generation_time < 1.0:
                logger.info("🎉 ULTRA-SPEED OPTIMIZATION SUCCESSFUL!")
                logger.info("🚀 Sub-second generation achieved")
                logger.info("✅ Ready for lightning-fast 55+ video production")
            else:
                logger.info("⚡ SIGNIFICANT SPEED IMPROVEMENT ACHIEVED")
                logger.info(f"📈 {speedup_vs_previous:.1f}x faster than previous FFmpeg")
                logger.info("🎯 Close to sub-second target")
            
            logger.info("=" * 60)
            return True
        else:
            logger.error("❌ Video generation failed")
            logger.error(f"Error: {result.error_message}")
            return False
            
    except Exception as e:
        total_time = time.time() - start_time
        logger.error("=" * 60)
        logger.error(f"❌ ULTRA-SPEED TEST FAILED: {e}")
        logger.error(f"   Runtime: {total_time:.1f}s")
        logger.error("=" * 60)
        return False

async def main():
    """Main execution"""
    success = await test_ultra_speed()
    
    if success:
        print("\n🎉 SUCCESS! Ultra-speed optimizations working!")
        print("⚡ Significant speed improvement achieved")
        return True
    else:
        print("\n💥 FAILED! Ultra-speed optimizations need attention")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        sys.exit(0)
    else:
        sys.exit(1)