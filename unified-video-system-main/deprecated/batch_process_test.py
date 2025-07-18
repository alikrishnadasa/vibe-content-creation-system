#!/usr/bin/env python3
"""
Batch Processing Test

Test the batch production system by generating multiple videos
from available scripts with multiple variations.
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

async def test_batch_processing():
    """Test batch processing with multiple scripts and variations"""
    logger.info("🚀 Starting Batch Processing Test")
    
    try:
        # Initialize the real content generator
        logger.info("🔧 Initializing content generator...")
        generator = RealContentGenerator(
            clips_directory="../MJAnime",
            metadata_file="../MJAnime/metadata_final_clean_shots.json", 
            scripts_directory="../11-scripts-for-tiktok",
            music_file="../Beanie (Slowed).mp3",
            output_directory="output"
        )
        
        # Initialize all components
        init_success = await generator.initialize()
        if not init_success:
            logger.error("❌ Failed to initialize generator")
            return False
        
        logger.info("✅ Generator initialized successfully")
        
        # Clear uniqueness cache for testing
        generator.uniqueness_engine.clear_cache()
        logger.info("🧹 Cleared uniqueness cache for testing")
        
        # Initialize batch pipeline
        logger.info("🔧 Initializing batch pipeline...")
        pipeline = ContentPipeline(
            content_generator=generator,
            max_concurrent_videos=2,  # Start with 2 concurrent for testing
            performance_monitoring=True
        )
        
        # Test 1: Process specific scripts (smaller test)
        logger.info("🧪 Test 1: Processing specific scripts...")
        test_scripts = ["anxiety1", "safe1", "miserable1"]  # 3 scripts for quick test
        
        result = await pipeline.process_specific_scripts(
            script_names=test_scripts,
            scripts_directory="../11-scripts-for-tiktok",
            variations_per_script=2,  # 2 variations each = 6 videos total
            caption_style="tiktok"
        )
        
        # Save test report
        pipeline.save_batch_report(result, "batch_test_report.json")
        
        # Display results
        logger.info("📊 Batch Processing Test Results:")
        logger.info(f"   Total videos: {result.total_videos}")
        logger.info(f"   Successful: {result.successful_videos}")
        logger.info(f"   Failed: {result.failed_videos}")
        logger.info(f"   Total time: {result.total_time:.3f}s")
        logger.info(f"   Average per video: {result.average_time_per_video:.3f}s")
        logger.info(f"   Success rate: {result.performance_stats.get('success_rate', 0):.1%}")
        logger.info(f"   Generation rate: {result.performance_stats.get('generation_rate_videos_per_second', 0):.2f} videos/second")
        
        if result.success:
            logger.info("✅ Batch processing test SUCCESSFUL!")
            
            # Show generated files
            logger.info("🎬 Generated Videos:")
            for i, path in enumerate(result.output_paths, 1):
                file_path = Path(path)
                if file_path.exists():
                    file_size_mb = file_path.stat().st_size / (1024 * 1024)
                    logger.info(f"   {i}. {file_path.name} ({file_size_mb:.1f}MB)")
                else:
                    logger.warning(f"   {i}. {file_path.name} (FILE NOT FOUND)")
            
            # Cleanup
            await pipeline.cleanup()
            return True
        else:
            logger.error("❌ Batch processing test FAILED!")
            logger.error("Error messages:")
            for error in result.error_messages:
                logger.error(f"   - {error}")
            
            await pipeline.cleanup()
            return False
            
    except Exception as e:
        logger.error(f"❌ Batch processing test failed with exception: {e}")
        return False

async def test_full_batch_processing():
    """Test full batch processing with all scripts (if requested)"""
    logger.info("🚀 Starting Full Batch Processing Test")
    
    try:
        # Initialize the real content generator
        logger.info("🔧 Initializing content generator...")
        generator = RealContentGenerator(
            clips_directory="../MJAnime",
            metadata_file="../MJAnime/metadata_final_clean_shots.json", 
            scripts_directory="../11-scripts-for-tiktok",
            music_file="../Beanie (Slowed).mp3",
            output_directory="output"
        )
        
        # Initialize all components
        init_success = await generator.initialize()
        if not init_success:
            logger.error("❌ Failed to initialize generator")
            return False
        
        logger.info("✅ Generator initialized successfully")
        
        # Clear uniqueness cache for testing
        generator.uniqueness_engine.clear_cache()
        logger.info("🧹 Cleared uniqueness cache for testing")
        
        # Initialize batch pipeline
        logger.info("🔧 Initializing batch pipeline...")
        pipeline = ContentPipeline(
            content_generator=generator,
            max_concurrent_videos=3,  # 3 concurrent for full test
            performance_monitoring=True
        )
        
        # Process all scripts
        logger.info("🎬 Processing ALL scripts with 5 variations each...")
        
        result = await pipeline.process_all_scripts(
            scripts_directory="../11-scripts-for-tiktok",
            variations_per_script=5,  # 5 variations per script
            caption_style="tiktok"
        )
        
        # Save full report
        pipeline.save_batch_report(result, "full_batch_report.json")
        
        # Display results
        logger.info("📊 Full Batch Processing Results:")
        logger.info(f"   Total videos: {result.total_videos}")
        logger.info(f"   Successful: {result.successful_videos}")
        logger.info(f"   Failed: {result.failed_videos}")
        logger.info(f"   Total time: {result.total_time:.3f}s")
        logger.info(f"   Average per video: {result.average_time_per_video:.3f}s")
        logger.info(f"   Success rate: {result.performance_stats.get('success_rate', 0):.1%}")
        logger.info(f"   Generation rate: {result.performance_stats.get('generation_rate_videos_per_second', 0):.2f} videos/second")
        
        # Performance analysis
        total_scripts = result.performance_stats.get('total_scripts_processed', 0)
        variations = result.performance_stats.get('variations_per_script', 0)
        
        logger.info(f"📈 Production Capacity Analysis:")
        logger.info(f"   Scripts processed: {total_scripts}")
        logger.info(f"   Variations per script: {variations}")
        logger.info(f"   Expected monthly capacity (1 video/day): ~{result.performance_stats.get('generation_rate_videos_per_second', 0) * 86400:.0f} videos/day")
        
        if result.success:
            logger.info("🎉 FULL BATCH PROCESSING SUCCESSFUL!")
            
            # Show sample of generated files
            logger.info("🎬 Sample Generated Videos:")
            for i, path in enumerate(result.output_paths[:10], 1):  # Show first 10
                file_path = Path(path)
                if file_path.exists():
                    file_size_mb = file_path.stat().st_size / (1024 * 1024)
                    logger.info(f"   {i}. {file_path.name} ({file_size_mb:.1f}MB)")
            
            if len(result.output_paths) > 10:
                logger.info(f"   ... and {len(result.output_paths) - 10} more videos")
            
            # Cleanup
            await pipeline.cleanup()
            return True
        else:
            logger.error("❌ Full batch processing FAILED!")
            logger.error("Error messages:")
            for error in result.error_messages:
                logger.error(f"   - {error}")
            
            await pipeline.cleanup()
            return False
            
    except Exception as e:
        logger.error(f"❌ Full batch processing failed with exception: {e}")
        return False

async def main():
    """Main test execution"""
    logger.info("🧪 Batch Processing Test Suite")
    
    # Ask user which test to run
    print("\nBatch Processing Test Options:")
    print("1. Quick test (3 scripts × 2 variations = 6 videos)")
    print("2. Full test (all scripts × 5 variations = 55+ videos)")
    print("3. Both tests")
    
    try:
        choice = input("\nEnter choice (1, 2, or 3): ").strip()
        
        if choice == "1":
            success = await test_batch_processing()
        elif choice == "2":
            success = await test_full_batch_processing()
        elif choice == "3":
            logger.info("Running both tests...")
            success1 = await test_batch_processing()
            if success1:
                logger.info("Quick test passed, proceeding to full test...")
                success = await test_full_batch_processing()
            else:
                logger.error("Quick test failed, skipping full test")
                success = False
        else:
            logger.error("Invalid choice")
            success = False
    
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        success = False
    
    if success:
        logger.info("\n✅ BATCH PROCESSING TEST SUCCESSFUL!")
        return True
    else:
        logger.error("\n❌ BATCH PROCESSING TEST FAILED!")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print("\n🎉 SUCCESS! Batch processing is working!")
        sys.exit(0)
    else:
        print("\n💥 FAILED! Batch processing needs attention")
        sys.exit(1)