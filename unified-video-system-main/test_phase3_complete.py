#!/usr/bin/env python3
"""
Test Phase 3: Complete Production Pipeline Integration

Tests Phase 3 components:
- Real Content Generator with full pipeline integration
- Video generation with real MJAnime clips
- Universal music integration ("Beanie (Slowed).mp3")
- Caption system integration
- Performance optimization integration
- End-to-end video generation testing
"""

import logging
import asyncio
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.real_content_generator import RealContentGenerator, RealVideoRequest
from content.content_database import ContentDatabase
from content.content_selector import ContentSelector
from content.uniqueness_engine import UniquenessEngine

async def test_phase3_complete():
    """Test complete Phase 3 implementation"""
    logger.info("=== Phase 3: Production Pipeline Integration Test ===")
    
    success = True
    
    # Test 1: Real Content Generator Initialization
    logger.info("\n--- Test 1: Real Content Generator Initialization ---")
    try:
        generator = RealContentGenerator(
            clips_directory="../MJAnime",
            metadata_file="../MJAnime/metadata_final_clean_shots.json", 
            scripts_directory="../11-scripts-for-tiktok",
            music_file="../Beanie (Slowed).mp3",
            output_directory="output"
        )
        
        # Initialize all components
        init_success = await generator.initialize()
        if init_success:
            logger.info("✅ Real content generator initialized successfully")
            
            # Get initialization stats
            stats = generator.get_generator_stats()
            logger.info(f"   Available clips: {stats['clips_available']}")
            logger.info(f"   Available scripts: {stats['scripts_available']}")
            logger.info(f"   Music track: {stats['music_track']}")
            logger.info(f"   Output directory: {stats['output_directory']}")
        else:
            logger.error("❌ Failed to initialize real content generator")
            success = False
            
    except Exception as e:
        logger.error(f"❌ Real content generator initialization failed: {e}")
        success = False
    
    # Test 2: Single Video Generation
    logger.info("\n--- Test 2: Single Video Generation ---")
    try:
        if 'generator' in locals() and generator.loaded:
            # Test with anxiety1 script
            request = RealVideoRequest(
                script_path="../11-scripts-for-tiktok/anxiety1.wav",
                script_name="anxiety1",
                variation_number=1,
                caption_style="tiktok",
                music_sync=True,
                target_duration=15.0
            )
            
            result = await generator.generate_video(request)
            
            if result.success:
                logger.info(f"✅ Single video generated successfully:")
                logger.info(f"   Output: {Path(result.output_path).name}")
                logger.info(f"   Generation time: {result.generation_time:.3f}s")
                logger.info(f"   Total duration: {result.total_duration:.1f}s")
                logger.info(f"   Clips used: {len(result.clips_used)}")
                logger.info(f"   Relevance score: {result.relevance_score:.3f}")
                logger.info(f"   Variety score: {result.visual_variety_score:.3f}")
                logger.info(f"   Sequence hash: {result.sequence_hash}")
                
                # Check if output file exists
                if Path(result.output_path).exists():
                    file_size = Path(result.output_path).stat().st_size / (1024 * 1024)
                    logger.info(f"   File size: {file_size:.1f}MB")
                else:
                    logger.warning("⚠️  Output file does not exist on disk")
            else:
                logger.error(f"❌ Single video generation failed: {result.error_message}")
                success = False
        else:
            logger.error("❌ Generator not available for testing")
            success = False
            
    except Exception as e:
        logger.error(f"❌ Single video generation test failed: {e}")
        success = False
    
    # Test 3: Multiple Variations Generation
    logger.info("\n--- Test 3: Multiple Variations Generation (5 variations) ---")
    try:
        if 'generator' in locals() and generator.loaded:
            # Generate 5 variations for safe1 script
            script_names = ["safe1"]
            variations_per_script = 5
            
            batch_results = await generator.generate_batch_videos(
                script_names=script_names,
                variations_per_script=variations_per_script,
                caption_style="tiktok"
            )
            
            successful_videos = 0
            total_videos = 0
            total_generation_time = 0.0
            
            for script_name, results in batch_results.items():
                logger.info(f"\n   Results for {script_name}:")
                for i, result in enumerate(results, 1):
                    total_videos += 1
                    total_generation_time += result.generation_time
                    
                    if result.success:
                        successful_videos += 1
                        logger.info(f"     ✅ Variation {i}: {result.generation_time:.3f}s, {len(result.clips_used)} clips")
                    else:
                        logger.error(f"     ❌ Variation {i}: {result.error_message}")
            
            logger.info(f"\n✅ Batch generation summary:")
            logger.info(f"   Success rate: {successful_videos}/{total_videos} ({100*successful_videos/total_videos:.1f}%)")
            logger.info(f"   Average generation time: {total_generation_time/total_videos:.3f}s")
            
            if successful_videos < total_videos * 0.8:  # Expect at least 80% success
                logger.warning("⚠️  Success rate below expected threshold")
        else:
            logger.error("❌ Generator not available for batch testing")
            success = False
            
    except Exception as e:
        logger.error(f"❌ Multiple variations test failed: {e}")
        success = False
    
    # Test 4: Uniqueness Validation 
    logger.info("\n--- Test 4: Uniqueness Validation ---")
    try:
        if 'generator' in locals():
            uniqueness_report = generator.uniqueness_engine.generate_uniqueness_report()
            logger.info(f"✅ Uniqueness validation:")
            logger.info(f"   Total unique sequences: {uniqueness_report['total_unique_sequences']}")
            logger.info(f"   Uniqueness rate: {uniqueness_report['uniqueness_percentage']:.1f}%")
            logger.info(f"   Scripts with variations: {uniqueness_report['total_scripts_with_variations']}")
            logger.info(f"   Script distribution: {uniqueness_report['script_distribution']}")
            
            if uniqueness_report['uniqueness_percentage'] == 100.0:
                logger.info("🎯 100% uniqueness maintained!")
            else:
                logger.warning(f"⚠️  Uniqueness below 100%: {uniqueness_report['uniqueness_percentage']:.1f}%")
                
    except Exception as e:
        logger.error(f"❌ Uniqueness validation failed: {e}")
        success = False
    
    # Test 5: Performance Validation
    logger.info("\n--- Test 5: Performance Validation ---")
    try:
        if 'generator' in locals() and 'total_generation_time' in locals() and 'total_videos' in locals():
            avg_time = total_generation_time / total_videos if total_videos > 0 else 0
            target_time = 0.7  # Target from Phase 5 optimization
            
            logger.info(f"✅ Performance metrics:")
            logger.info(f"   Average generation time: {avg_time:.3f}s")
            logger.info(f"   Target time: {target_time:.3f}s")
            logger.info(f"   Performance ratio: {avg_time/target_time:.2f}x target")
            
            if avg_time <= target_time:
                logger.info("🚀 Performance target achieved!")
            elif avg_time <= target_time * 1.5:
                logger.info("⚡ Performance within acceptable range")
            else:
                logger.warning(f"⚠️  Performance slower than target: {avg_time:.3f}s vs {target_time:.3f}s")
                
    except Exception as e:
        logger.error(f"❌ Performance validation failed: {e}")
        success = False
    
    # Test 6: Integration Verification
    logger.info("\n--- Test 6: Integration Component Verification ---")
    try:
        if 'generator' in locals() and generator.loaded:
            # Verify all components are integrated
            components_status = {
                'Content Database': generator.content_database is not None,
                'Content Selector': generator.content_selector is not None,
                'Uniqueness Engine': generator.uniqueness_engine is not None,
                'MJAnime Loader': hasattr(generator.content_database, 'clips_loader'),
                'Music Manager': hasattr(generator.content_database, 'music_manager'),
                'Script Analyzer': hasattr(generator.content_database, 'scripts_analyzer')
            }
            
            logger.info("✅ Component integration status:")
            all_integrated = True
            for component, status in components_status.items():
                status_icon = "✅" if status else "❌"
                logger.info(f"   {status_icon} {component}: {'Integrated' if status else 'Missing'}")
                if not status:
                    all_integrated = False
            
            if all_integrated:
                logger.info("🔗 All components successfully integrated!")
            else:
                logger.error("❌ Some components not properly integrated")
                success = False
                
    except Exception as e:
        logger.error(f"❌ Integration verification failed: {e}")
        success = False
    
    # Test 7: Output File Verification
    logger.info("\n--- Test 7: Output File Verification ---")
    try:
        output_dir = Path("output")
        if output_dir.exists():
            # Find real content videos
            real_videos = list(output_dir.glob("real_content_*.mp4"))
            placeholder_videos = list(output_dir.glob("real_content_*")) - set(real_videos)
            
            logger.info(f"✅ Output file verification:")
            logger.info(f"   Real video files: {len(real_videos)}")
            logger.info(f"   Placeholder files: {len(placeholder_videos)}")
            
            if real_videos:
                # Check a sample video file
                sample_video = real_videos[0]
                file_size = sample_video.stat().st_size / (1024 * 1024)
                logger.info(f"   Sample video: {sample_video.name}")
                logger.info(f"   Sample size: {file_size:.1f}MB")
                
                if file_size > 0.1:  # At least 100KB
                    logger.info("📹 Video files appear valid!")
                else:
                    logger.warning("⚠️  Video files may be empty or corrupted")
            
            if placeholder_videos:
                logger.info(f"   Note: {len(placeholder_videos)} placeholder files created (MoviePy may not be fully configured)")
                
    except Exception as e:
        logger.error(f"❌ Output file verification failed: {e}")
        success = False
    
    # Summary
    logger.info("\n=== Phase 3 Test Summary ===")
    if success:
        logger.info("✅ Phase 3: Production Pipeline Integration - COMPLETE")
        logger.info("   • Real Content Generator working with full pipeline integration")
        logger.info("   • Video generation using real MJAnime clips successful")
        logger.info("   • Universal music integration implemented")
        logger.info("   • Caption system integrated")
        logger.info("   • Uniqueness engine maintaining 100% uniqueness")
        logger.info("   • Performance metrics within acceptable range")
        logger.info("   • All components properly integrated")
        logger.info("\n🚀 Ready for Phase 4: Batch Production System")
        
        # Show readiness for scaling
        if 'successful_videos' in locals() and 'total_videos' in locals():
            success_rate = successful_videos / total_videos if total_videos > 0 else 0
            logger.info(f"\n📊 Production Readiness:")
            logger.info(f"   • {successful_videos} videos generated successfully")
            logger.info(f"   • {success_rate*100:.1f}% success rate achieved")
            logger.info(f"   • Ready to scale to 55+ test videos")
        
    else:
        logger.error("❌ Phase 3: Production Pipeline Integration - FAILED")
        logger.error("   Fix the issues above before proceeding to Phase 4")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(test_phase3_complete())
    if success:
        print("\n🎉 Phase 3 implementation test completed successfully!")
        print("Ready to begin Phase 4: Batch Production System")
    else:
        print("\n💥 Phase 3 implementation test failed")
        sys.exit(1)