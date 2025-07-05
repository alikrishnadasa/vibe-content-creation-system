#!/usr/bin/env python3
"""
Test Real Video Generation

Test the actual video generation with MoviePy to create viewable MP4 files.
"""

import logging
import sys
import time
import asyncio
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import real video generation components
from core.real_content_generator import RealContentGenerator, RealVideoRequest
from core.quantum_pipeline import UnifiedQuantumPipeline

async def test_single_real_video():
    """Test generation of a single real video"""
    logger.info("=== Testing Single Real Video Generation ===")
    
    # Initialize quantum pipeline with real content mode
    pipeline = UnifiedQuantumPipeline()
    
    # Initialize real content mode
    success = await pipeline.initialize_real_content_mode(
        clips_directory="../MJAnime",
        metadata_file="../MJAnime/metadata_final_clean_shots.json",
        scripts_directory="../11-scripts-for-tiktok",
        music_file="../Beanie (Slowed).mp3"
    )
    
    if not success:
        logger.error("Failed to initialize real content mode")
        return False
    
    # Generate a real video
    result = await pipeline.generate_real_content_video(
        script_name="anxiety1",
        variation_number=99,  # Use high number to avoid conflicts
        caption_style="tiktok"
    )
    
    if result['success']:
        output_file = Path(result['output_path'])
        
        logger.info(f"✅ Real video generated successfully!")
        logger.info(f"   Output: {output_file.name}")
        logger.info(f"   Path: {result['output_path']}")
        logger.info(f"   Generation time: {result['processing_time']:.3f}s")
        logger.info(f"   Target achieved: {result['target_achieved']}")
        logger.info(f"   File exists: {output_file.exists()}")
        
        if output_file.exists():
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            logger.info(f"   File size: {file_size_mb:.2f}MB")
            
            # Check if it's a real video file or placeholder
            if output_file.suffix.lower() == '.mp4':
                logger.info(f"   ✅ Real MP4 file generated!")
                
                # Try to get video info using moviepy
                try:
                    from moviepy.editor import VideoFileClip
                    with VideoFileClip(str(output_file)) as video:
                        logger.info(f"   Video duration: {video.duration:.1f}s")
                        logger.info(f"   Video resolution: {video.size}")
                        logger.info(f"   Video FPS: {video.fps}")
                        logger.info(f"   Has audio: {video.audio is not None}")
                        
                        logger.info(f"🎉 REAL VIDEO SUCCESSFULLY CREATED AND VERIFIED!")
                        return True
                        
                except Exception as e:
                    logger.error(f"   ❌ Failed to verify video file: {e}")
                    return False
            else:
                logger.warning(f"   ⚠️  Placeholder file created instead of real video")
                return False
        else:
            logger.error(f"   ❌ Output file does not exist")
            return False
    else:
        logger.error(f"❌ Video generation failed: {result.get('error')}")
        return False

async def test_batch_real_videos():
    """Test generation of batch real videos"""
    logger.info("\\n=== Testing Batch Real Video Generation ===")
    
    # Initialize quantum pipeline
    pipeline = UnifiedQuantumPipeline()
    
    # Initialize real content mode
    await pipeline.initialize_real_content_mode(
        clips_directory="../MJAnime",
        metadata_file="../MJAnime/metadata_final_clean_shots.json",
        scripts_directory="../11-scripts-for-tiktok",
        music_file="../Beanie (Slowed).mp3"
    )
    
    # Generate a small batch of real videos
    batch_result = await pipeline.generate_real_content_batch(
        script_names=["safe1", "adhd"],
        variations_per_script=2,
        caption_style="tiktok"
    )
    
    if batch_result['success']:
        logger.info(f"✅ Batch generation completed!")
        logger.info(f"   Total videos: {batch_result['total_videos']}")
        logger.info(f"   Successful: {batch_result['successful_videos']}")
        logger.info(f"   Success rate: {batch_result['success_rate']:.1%}")
        logger.info(f"   Average time per video: {batch_result['average_time_per_video']:.3f}s")
        
        # Check individual results
        real_videos_created = 0
        for i, result in enumerate(batch_result['individual_results']):
            if result['success']:
                output_file = Path(result['output_path'])
                if output_file.exists() and output_file.suffix.lower() == '.mp4':
                    file_size_mb = output_file.stat().st_size / (1024 * 1024)
                    logger.info(f"   Video {i+1}: {output_file.name} ({file_size_mb:.1f}MB)")
                    real_videos_created += 1
        
        logger.info(f"   Real MP4 files created: {real_videos_created}/{batch_result['successful_videos']}")
        
        return real_videos_created > 0
    else:
        logger.error("❌ Batch generation failed")
        return False

async def test_direct_asset_processor():
    """Test the asset processor directly for video composition"""
    logger.info("\\n=== Testing Direct Asset Processor ===")
    
    from core.real_asset_processor import RealAssetProcessor
    
    # Initialize processor
    processor = RealAssetProcessor()
    processor.initialize_gpu_processing()
    
    # Test with real clips
    clip_paths = [
        "../MJAnime/social_u1875146414_A_serene_devotee_offers_banana-leaf_plates_of_pra_fcbb5dd9-0d59-45fc-8b91-46b7d721d4bf_3.mp4",
        "../MJAnime/social_u1875146414_A_Hare_Krishna_devotee_offers_Bhagavad-gita_books_669348a5-48cb-4d40-8544-88071e14b9a8_1.mp4"
    ]
    
    # Filter to existing clips only
    existing_clips = [path for path in clip_paths if Path(path).exists()]
    
    if not existing_clips:
        logger.error("No test clips found")
        return False
    
    logger.info(f"Testing with {len(existing_clips)} clips")
    
    # Test audio mixing first
    mixed_audio_path = "output/test_mixed_audio_direct.wav"
    
    mix_result = await processor.mix_script_with_music(
        script_path="../11-scripts-for-tiktok/anxiety1.wav",
        music_path="../Beanie (Slowed).mp3",
        music_start_time=0.0,
        music_duration=10.0,
        output_path=mixed_audio_path,
        script_volume=1.0,
        music_volume=0.25
    )
    
    logger.info(f"Audio mixing result: {mix_result.success}")
    
    # Test video composition
    output_path = "output/test_direct_video.mp4"
    
    composition_result = await processor.composite_video_with_captions(
        clip_paths=existing_clips,
        audio_path=mixed_audio_path if mix_result.success else "../11-scripts-for-tiktok/anxiety1.wav",
        caption_data={
            'style': 'tiktok',
            'captions': [
                {'text': 'Finding inner peace', 'start_time': 0.0, 'end_time': 3.0},
                {'text': 'Through spiritual practice', 'start_time': 3.0, 'end_time': 6.0},
                {'text': 'Consciousness awakens', 'start_time': 6.0, 'end_time': 9.0}
            ],
            'total_duration': 9.0
        },
        output_path=output_path,
        target_resolution=(1080, 1936)
    )
    
    if composition_result['success']:
        output_file = Path(output_path)
        if output_file.exists():
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            logger.info(f"✅ Direct video composition successful!")
            logger.info(f"   Output: {output_file.name}")
            logger.info(f"   File size: {file_size_mb:.2f}MB")
            logger.info(f"   Duration: {composition_result['duration']:.1f}s")
            logger.info(f"   Clips processed: {composition_result['clips_processed']}")
            logger.info(f"   Captions applied: {composition_result['captions_applied']}")
            
            # Verify it's a real video
            try:
                from moviepy.editor import VideoFileClip
                with VideoFileClip(str(output_file)) as video:
                    logger.info(f"   ✅ Video verified: {video.duration:.1f}s, {video.size}, {video.fps}fps")
                    return True
            except Exception as e:
                logger.error(f"   ❌ Video verification failed: {e}")
                return False
        else:
            logger.error("Output file does not exist")
            return False
    else:
        logger.error(f"❌ Direct composition failed: {composition_result.get('error_message')}")
        return False

async def main():
    """Run all real video generation tests"""
    logger.info("🎬 Starting Real Video Generation Tests")
    logger.info("Testing actual MP4 video creation with MoviePy...")
    
    start_time = time.time()
    tests = [
        ("Direct Asset Processor", test_direct_asset_processor),
        ("Single Real Video", test_single_real_video),
        ("Batch Real Videos", test_batch_real_videos)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\\n--- {test_name} ---")
            if await test_func():
                logger.info(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
    
    elapsed = time.time() - start_time
    
    logger.info(f"\\n=== Real Video Generation Test Results ===")
    logger.info(f"Passed: {passed}/{total}")
    logger.info(f"Success Rate: {passed/total*100:.1f}%")
    logger.info(f"Total Time: {elapsed:.2f}s")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED - REAL VIDEOS ARE BEING GENERATED!")
        logger.info("✅ Phase 3 Production Pipeline Integration - FULLY COMPLETE")
        return True
    else:
        logger.error("❌ Some real video generation tests failed")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)