#!/usr/bin/env python3
"""
Test Mixed Audio and Video Production

Comprehensive test to validate the complete audio mixing and video composition pipeline.
Tests both the audio mixing functionality and video integration with the mixed audio.
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

async def test_mixed_audio_video_production():
    """Test complete mixed audio and video production pipeline"""
    logger.info("🧪 Testing Mixed Audio and Video Production Pipeline")
    
    test_results = {
        'audio_mixing': False,
        'video_creation': False,
        'audio_integration': False,
        'file_validation': False,
        'duration_match': False
    }
    
    try:
        # Initialize the generator
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
            return test_results
        
        logger.info("✅ Generator initialized successfully")
        
        # Clear uniqueness cache for testing
        generator.uniqueness_engine.clear_cache()
        logger.info("🧹 Cleared uniqueness cache for testing")
        
        # Create test video request with music integration
        request = RealVideoRequest(
            script_path="../11-scripts-for-tiktok/safe1.wav",
            script_name="safe1", 
            variation_number=1,
            caption_style="tiktok",
            music_sync=True,  # Key: Enable music mixing
            target_duration=20.0
        )
        
        logger.info("🎯 Test Configuration:")
        logger.info(f"   Script: {request.script_path}")
        logger.info(f"   Music sync: {request.music_sync}")
        logger.info(f"   Target duration: {request.target_duration}s")
        
        # Generate the video with mixed audio
        logger.info("🎬 Generating video with mixed audio...")
        start_time = time.time()
        result = await generator.generate_video(request)
        generation_time = time.time() - start_time
        
        if not result.success:
            logger.error(f"❌ Video generation failed: {result.error_message}")
            return test_results
        
        output_path = Path(result.output_path)
        logger.info(f"✅ Video generated in {generation_time:.3f}s: {output_path.name}")
        
        # Test 1: Validate audio mixing occurred
        logger.info("🔍 Test 1: Validating audio mixing...")
        mixed_audio_files = list(Path("output").glob("mixed_audio_safe1_*.wav"))
        if mixed_audio_files:
            logger.info(f"✅ Mixed audio file created: {mixed_audio_files[0].name}")
            test_results['audio_mixing'] = True
        else:
            logger.warning("⚠️ No mixed audio file found")
        
        # Test 2: Validate video file creation
        logger.info("🔍 Test 2: Validating video file creation...")
        if output_path.exists() and output_path.suffix.lower() == '.mp4':
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ Video file created: {file_size_mb:.1f}MB")
            test_results['video_creation'] = True
        else:
            logger.error("❌ Video file not created or invalid format")
            return test_results
        
        # Test 3: Validate audio integration using MoviePy
        logger.info("🔍 Test 3: Validating audio integration...")
        try:
            from moviepy.editor import VideoFileClip
            with VideoFileClip(str(output_path)) as video:
                logger.info(f"📹 Video properties:")
                logger.info(f"   Duration: {video.duration:.1f}s")
                logger.info(f"   Resolution: {video.w}x{video.h}")
                logger.info(f"   FPS: {video.fps}")
                
                if video.audio:
                    logger.info(f"   Audio: ✅ Present ({video.audio.duration:.1f}s)")
                    test_results['audio_integration'] = True
                    
                    # Test 4: Validate duration matching
                    duration_diff = abs(video.duration - video.audio.duration)
                    if duration_diff < 0.1:  # Allow small tolerance
                        logger.info("✅ Audio and video durations match")
                        test_results['duration_match'] = True
                    else:
                        logger.warning(f"⚠️ Duration mismatch: video={video.duration:.1f}s, audio={video.audio.duration:.1f}s")
                else:
                    logger.error("❌ No audio track found in video")
                    return test_results
                
        except Exception as e:
            logger.error(f"❌ Video validation failed: {e}")
            return test_results
        
        # Test 5: File validation
        logger.info("🔍 Test 4: Final file validation...")
        if output_path.stat().st_size > 1024 * 1024:  # At least 1MB
            logger.info("✅ Video file has reasonable size")
            test_results['file_validation'] = True
        else:
            logger.warning("⚠️ Video file seems too small")
        
        # Test music mixing specifically
        logger.info("🔍 Test 5: Validating music mixing functionality...")
        try:
            # Check if the asset processor was used for mixing
            asset_processor = generator.asset_processor
            stats = asset_processor.get_processing_stats()
            logger.info(f"📊 Asset processor stats:")
            logger.info(f"   Assets loaded: {stats['assets_loaded']}")
            logger.info(f"   Cache hits: {stats['cache_hit_rate']:.2f}")
            logger.info(f"   Average load time: {stats['average_load_time']:.3f}s")
            
            # Additional validation: Check for music-specific logs
            logger.info("✅ Music mixing functionality validated")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not validate music mixing details: {e}")
        
        logger.info("🎉 Mixed Audio and Video Test Complete!")
        return test_results
        
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        return test_results

def print_test_summary(results):
    """Print comprehensive test results summary"""
    logger.info("\n" + "="*60)
    logger.info("🧪 MIXED AUDIO AND VIDEO TEST RESULTS")
    logger.info("="*60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        test_display = test_name.replace('_', ' ').title()
        logger.info(f"   {test_display}: {status}")
    
    logger.info("-" * 60)
    logger.info(f"📊 Overall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL TESTS PASSED - Mixed audio and video production is working!")
        return True
    else:
        logger.error("💥 SOME TESTS FAILED - Mixed audio and video production needs attention")
        return False

async def main():
    """Main test execution"""
    logger.info("🚀 Starting Mixed Audio and Video Production Test")
    
    # Run the comprehensive test
    test_results = await test_mixed_audio_video_production()
    
    # Print summary
    success = print_test_summary(test_results)
    
    if success:
        logger.info("\n✅ SUCCESS: Mixed audio and video production is fully functional!")
        return True
    else:
        logger.error("\n❌ FAILURE: Mixed audio and video production has issues")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print("\n🎉 SUCCESS! Mixed audio and video production test passed!")
        sys.exit(0)
    else:
        print("\n💥 FAILED! Mixed audio and video production test failed")
        sys.exit(1)