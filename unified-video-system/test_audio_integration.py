#!/usr/bin/env python3
"""
Test Audio Integration

Test that audio is properly integrated into generated videos.
"""

import logging
import asyncio
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from core.real_asset_processor import RealAssetProcessor

async def test_audio_integration():
    """Test audio integration in video generation"""
    logger.info("=== Testing Audio Integration ===")
    
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
    
    # Test 1: Video with original script audio (no mixing)
    logger.info("\\n--- Test 1: Direct script audio ---")
    
    output_path_1 = "output/test_audio_direct.mp4"
    
    composition_result_1 = await processor.composite_video_with_captions(
        clip_paths=existing_clips,
        audio_path="../11-scripts-for-tiktok/anxiety1.wav",  # Direct script audio
        caption_data={
            'style': 'tiktok',
            'captions': [
                {'text': 'Testing direct audio', 'start_time': 0.0, 'end_time': 3.0},
                {'text': 'Script audio only', 'start_time': 3.0, 'end_time': 6.0},
                {'text': 'Should have sound', 'start_time': 6.0, 'end_time': 9.0}
            ],
            'total_duration': 9.0
        },
        output_path=output_path_1,
        target_resolution=(1080, 1936)
    )
    
    if composition_result_1['success']:
        output_file_1 = Path(output_path_1)
        if output_file_1.exists():
            logger.info(f"✅ Test 1 video created: {output_file_1.name}")
            
            # Verify audio
            try:
                from moviepy.editor import VideoFileClip
                with VideoFileClip(str(output_file_1)) as video:
                    has_audio = video.audio is not None
                    logger.info(f"   Has audio: {has_audio}")
                    if has_audio:
                        logger.info(f"   Audio duration: {video.audio.duration:.1f}s")
                        logger.info(f"   ✅ SUCCESS: Audio integrated!")
                    else:
                        logger.error(f"   ❌ FAILED: No audio in video")
                        return False
            except Exception as e:
                logger.error(f"   ❌ Failed to verify video: {e}")
                return False
        else:
            logger.error("Test 1 output file does not exist")
            return False
    else:
        logger.error(f"❌ Test 1 failed: {composition_result_1.get('error_message')}")
        return False
    
    # Test 2: Audio mixing (fixed version)
    logger.info("\\n--- Test 2: Audio mixing ---")
    
    # First create mixed audio
    mixed_audio_path = "output/test_mixed_audio_fixed.wav"
    
    mix_result = await processor.mix_script_with_music(
        script_path="../11-scripts-for-tiktok/anxiety1.wav",
        music_path="../Beanie (Slowed).mp3",
        music_start_time=0.0,
        music_duration=15.0,
        output_path=mixed_audio_path,
        script_volume=1.0,
        music_volume=0.3
    )
    
    logger.info(f"Audio mixing result: {mix_result.success}")
    
    if mix_result.success:
        # Test the mixed audio in video
        output_path_2 = "output/test_audio_mixed.mp4"
        
        composition_result_2 = await processor.composite_video_with_captions(
            clip_paths=existing_clips,
            audio_path=mixed_audio_path,
            caption_data={
                'style': 'tiktok',
                'captions': [
                    {'text': 'Testing mixed audio', 'start_time': 0.0, 'end_time': 3.0},
                    {'text': 'Script + music', 'start_time': 3.0, 'end_time': 6.0},
                    {'text': 'Should hear both', 'start_time': 6.0, 'end_time': 9.0}
                ],
                'total_duration': 9.0
            },
            output_path=output_path_2,
            target_resolution=(1080, 1936)
        )
        
        if composition_result_2['success']:
            output_file_2 = Path(output_path_2)
            if output_file_2.exists():
                logger.info(f"✅ Test 2 video created: {output_file_2.name}")
                
                # Verify mixed audio
                try:
                    from moviepy.editor import VideoFileClip
                    with VideoFileClip(str(output_file_2)) as video:
                        has_audio = video.audio is not None
                        logger.info(f"   Has audio: {has_audio}")
                        if has_audio:
                            logger.info(f"   Audio duration: {video.audio.duration:.1f}s")
                            logger.info(f"   ✅ SUCCESS: Mixed audio integrated!")
                        else:
                            logger.error(f"   ❌ FAILED: No mixed audio in video")
                except Exception as e:
                    logger.error(f"   ❌ Failed to verify mixed video: {e}")
            else:
                logger.error("Test 2 output file does not exist")
        else:
            logger.error(f"❌ Test 2 composition failed: {composition_result_2.get('error_message')}")
    else:
        logger.warning(f"⚠️  Audio mixing failed: {mix_result.error_message}")
    
    logger.info("\\n=== Audio Integration Test Complete ===")
    logger.info("Check the output files:")
    logger.info(f"  1. {output_path_1} - Direct script audio")
    logger.info(f"  2. {output_path_2} - Mixed script + music audio")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_audio_integration())
    if success:
        print("\\n🎉 Audio integration test completed!")
        print("Please check the generated video files to verify audio is working.")
    else:
        print("\\n❌ Audio integration test failed")