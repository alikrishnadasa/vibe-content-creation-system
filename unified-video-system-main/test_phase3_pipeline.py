#!/usr/bin/env python3
"""
Test Phase 3: Production Pipeline Integration

Test real content video generator, real asset processor, and quantum pipeline integration.
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

# Import Phase 3 components
from core.real_content_generator import RealContentGenerator, RealVideoRequest
from core.real_asset_processor import RealAssetProcessor
from core.quantum_pipeline import UnifiedQuantumPipeline

async def test_real_content_generator():
    """Test real content video generator"""
    logger.info("=== Testing Real Content Generator ===")
    
    # Initialize generator
    generator = RealContentGenerator(
        clips_directory="../MJAnime",
        metadata_file="../MJAnime/metadata_final_clean_shots.json",
        scripts_directory="../11-scripts-for-tiktok",
        music_file="../Beanie (Slowed).mp3"
    )
    
    # Initialize
    success = await generator.initialize()
    if not success:
        logger.error("Failed to initialize real content generator")
        return False
    
    # Test single video generation
    test_scripts = ["anxiety1", "safe1"]
    
    for script_name in test_scripts:
        request = RealVideoRequest(
            script_path=f"../11-scripts-for-tiktok/{script_name}.wav",
            script_name=script_name,
            variation_number=1,
            caption_style="tiktok",
            music_sync=True
        )
        
        result = await generator.generate_video(request)
        
        if result.success:
            logger.info(f"✅ {script_name}: Generated in {result.generation_time:.3f}s")
            logger.info(f"   Output: {Path(result.output_path).name}")
            logger.info(f"   Clips: {len(result.clips_used)}")
            logger.info(f"   Duration: {result.total_duration:.1f}s")
            logger.info(f"   Relevance: {result.relevance_score:.2f}")
            logger.info(f"   Variety: {result.visual_variety_score:.2f}")
        else:
            logger.error(f"❌ {script_name}: Failed - {result.error_message}")
            return False
    
    # Test batch generation (small batch)
    batch_results = await generator.generate_batch_videos(
        script_names=["anxiety1", "adhd"],
        variations_per_script=2,
        caption_style="tiktok"
    )
    
    successful_batch = sum(1 for r in batch_results if r.success)
    logger.info(f"✅ Batch test: {successful_batch}/{len(batch_results)} successful")
    
    # Test generator stats
    stats = generator.get_generator_stats()
    logger.info(f"✅ Generator stats: {stats['clips_available']} clips, {stats['scripts_available']} scripts")
    
    return successful_batch > 0

async def test_real_asset_processor():
    """Test real asset processor"""
    logger.info("\\n=== Testing Real Asset Processor ===")
    
    # Initialize processor
    processor = RealAssetProcessor(gpu_memory_pool_mb=1024)
    
    # Initialize GPU processing
    gpu_success = processor.initialize_gpu_processing()
    logger.info(f"✅ GPU processing initialized: {gpu_success}")
    
    # Test clip loading
    test_clips = [
        "../MJAnime/social_u1875146414_A_Hare_Krishna_devotee_offers_Bhagavad-gita_books_669348a5-48cb-4d40-8544-88071e14b9a8_1.mp4",
        "../MJAnime/social_u1875146414_A_serene_devotee_offers_banana-leaf_plates_of_pra_fcbb5dd9-0d59-45fc-8b91-46b7d721d4bf_3.mp4"
    ]
    
    load_results = []
    for clip_path in test_clips:
        result = await processor.load_clip_efficiently(clip_path, preload_to_gpu=True)
        load_results.append(result)
        
        if result.success:
            logger.info(f"✅ Loaded clip: {Path(clip_path).name}")
            logger.info(f"   Duration: {result.duration:.1f}s")
            logger.info(f"   Resolution: {result.resolution}")
            logger.info(f"   GPU Memory: {result.memory_usage_mb:.1f}MB")
            logger.info(f"   Load time: {result.load_time:.3f}s")
        else:
            logger.warning(f"⚠️  Failed to load: {Path(clip_path).name} - {result.error_message}")
    
    successful_loads = sum(1 for r in load_results if r.success)
    
    # Test music loading
    music_result = await processor.load_music_track(
        "../Beanie (Slowed).mp3",
        segment_start=0.0,
        segment_duration=30.0
    )
    
    if music_result.success:
        logger.info(f"✅ Music loaded: {music_result.duration:.1f}s segment")
        logger.info(f"   Memory usage: {music_result.memory_usage_mb:.1f}MB")
    else:
        logger.warning(f"⚠️  Music loading failed: {music_result.error_message}")
    
    # Test audio mixing
    mix_result = await processor.mix_script_with_music(
        script_path="../11-scripts-for-tiktok/anxiety1.wav",
        music_path="../Beanie (Slowed).mp3",
        music_start_time=0.0,
        music_duration=25.0,
        output_path="temp/test_mixed_audio.wav",
        script_volume=1.0,
        music_volume=0.25
    )
    
    if mix_result.success:
        logger.info(f"✅ Audio mixed: {mix_result.mixed_duration:.1f}s")
        logger.info(f"   Script: {mix_result.script_duration:.1f}s")
        logger.info(f"   Music: {mix_result.music_duration:.1f}s")
        logger.info(f"   Mix time: {mix_result.mix_time:.3f}s")
    else:
        logger.error(f"❌ Audio mixing failed: {mix_result.error_message}")
        return False
    
    # Test video composition
    clip_paths = [r.asset_path for r in load_results if r.success]
    if clip_paths and mix_result.success:
        composition_result = await processor.composite_video_with_captions(
            clip_paths=clip_paths,
            audio_path=mix_result.output_path,
            caption_data={
                'style': 'tiktok',
                'captions': [
                    {'text': 'Test caption 1', 'start_time': 0.0, 'end_time': 5.0},
                    {'text': 'Test caption 2', 'start_time': 5.0, 'end_time': 10.0}
                ],
                'total_duration': 10.0
            },
            output_path="temp/test_final_video.mp4"
        )
        
        if composition_result['success']:
            logger.info(f"✅ Video composition: {composition_result['clips_processed']} clips")
            logger.info(f"   Duration: {composition_result['duration']:.1f}s")
            logger.info(f"   Captions: {composition_result['captions_applied']}")
            logger.info(f"   Composition time: {composition_result['composition_time']:.3f}s")
        else:
            logger.error(f"❌ Video composition failed: {composition_result.get('error_message')}")
            return False
    
    # Test processor stats
    stats = processor.get_processing_stats()
    logger.info(f"✅ Processor stats:")
    logger.info(f"   Assets loaded: {stats['assets_loaded']}")
    logger.info(f"   Cache hit rate: {stats['cache_hit_rate']:.1%}")
    logger.info(f"   GPU memory used: {stats['gpu_memory_used_mb']:.1f}MB")
    logger.info(f"   Average load time: {stats['average_load_time']:.3f}s")
    
    return successful_loads > 0 and music_result.success and mix_result.success

async def test_quantum_pipeline_integration():
    """Test quantum pipeline integration with real content"""
    logger.info("\\n=== Testing Quantum Pipeline Integration ===")
    
    # Initialize quantum pipeline
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
    
    logger.info("✅ Real content mode initialized")
    
    # Test single real content video generation
    test_scripts = ["anxiety1", "safe1"]
    
    for script_name in test_scripts:
        result = await pipeline.generate_real_content_video(
            script_name=script_name,
            variation_number=1,
            caption_style="tiktok"
        )
        
        if result['success']:
            logger.info(f"✅ {script_name}: Generated via pipeline")
            logger.info(f"   Time: {result['processing_time']:.3f}s")
            logger.info(f"   Target achieved: {result['target_achieved']}")
            logger.info(f"   Clips: {len(result['real_content_data']['clips_used'])}")
            logger.info(f"   Relevance: {result['real_content_data']['relevance_score']:.2f}")
        else:
            logger.error(f"❌ {script_name}: Pipeline generation failed - {result.get('error')}")
            return False
    
    # Test batch generation through pipeline (testing phase: 2 scripts × 2 variations)
    batch_result = await pipeline.generate_real_content_batch(
        script_names=["anxiety1", "adhd"],
        variations_per_script=2,
        caption_style="tiktok"
    )
    
    if batch_result['success']:
        logger.info(f"✅ Batch via pipeline: {batch_result['successful_videos']}/{batch_result['total_videos']}")
        logger.info(f"   Batch time: {batch_result['batch_time']:.2f}s")
        logger.info(f"   Average per video: {batch_result['average_time_per_video']:.3f}s")
        logger.info(f"   Success rate: {batch_result['success_rate']:.1%}")
        logger.info(f"   Target achievement: {batch_result['target_achievement_rate']:.1%}")
    else:
        logger.error("❌ Batch generation through pipeline failed")
        return False
    
    # Test pipeline statistics
    stats = pipeline.get_real_content_stats()
    logger.info(f"✅ Pipeline real content stats:")
    logger.info(f"   Mode: {stats['real_content_mode']}")
    logger.info(f"   Clips available: {stats['generator']['clips_available']}")
    logger.info(f"   Scripts available: {stats['generator']['scripts_available']}")
    logger.info(f"   Performance: {stats['performance']['average_time']:.3f}s avg")
    
    return True

async def test_performance_targets():
    """Test if real content generation meets performance targets"""
    logger.info("\\n=== Testing Performance Targets ===")
    
    pipeline = UnifiedQuantumPipeline()
    
    # Initialize real content mode
    await pipeline.initialize_real_content_mode(
        clips_directory="../MJAnime",
        metadata_file="../MJAnime/metadata_final_clean_shots.json",
        scripts_directory="../11-scripts-for-tiktok",
        music_file="../Beanie (Slowed).mp3"
    )
    
    # Test multiple generations to verify consistent performance
    test_count = 5
    results = []
    target_time = 0.7  # 0.7s target
    
    for i in range(test_count):
        result = await pipeline.generate_real_content_video(
            script_name="anxiety1",
            variation_number=i + 10,  # Avoid conflicts with previous tests
            caption_style="tiktok"
        )
        
        if result['success']:
            results.append(result['processing_time'])
            logger.info(f"  Test {i+1}: {result['processing_time']:.3f}s {'✅' if result['target_achieved'] else '⚠️'}")
        else:
            logger.error(f"  Test {i+1}: Failed")
            return False
    
    if results:
        avg_time = sum(results) / len(results)
        min_time = min(results)
        max_time = max(results)
        under_target = sum(1 for t in results if t <= target_time)
        
        logger.info(f"✅ Performance summary:")
        logger.info(f"   Average: {avg_time:.3f}s")
        logger.info(f"   Best: {min_time:.3f}s")
        logger.info(f"   Worst: {max_time:.3f}s")
        logger.info(f"   Under target: {under_target}/{len(results)} ({under_target/len(results)*100:.1f}%)")
        
        # Success criteria: average under target, at least 60% of runs under target
        avg_success = avg_time <= target_time
        consistency_success = (under_target / len(results)) >= 0.6
        
        logger.info(f"   Average target achieved: {avg_success}")
        logger.info(f"   Consistency target achieved: {consistency_success}")
        
        return avg_success and consistency_success
    
    return False

async def main():
    """Run all Phase 3 tests"""
    logger.info("Starting Phase 3: Production Pipeline Integration Tests")
    
    start_time = time.time()
    tests = [
        ("Real Content Generator", test_real_content_generator),
        ("Real Asset Processor", test_real_asset_processor),
        ("Quantum Pipeline Integration", test_quantum_pipeline_integration),
        ("Performance Targets", test_performance_targets)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if await test_func():
                logger.info(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
    
    elapsed = time.time() - start_time
    
    logger.info(f"\\n=== Phase 3 Test Results ===")
    logger.info(f"Passed: {passed}/{total}")
    logger.info(f"Success Rate: {passed/total*100:.1f}%")
    logger.info(f"Total Time: {elapsed:.2f}s")
    
    if passed == total:
        logger.info("🎉 Phase 3: Production Pipeline Integration - COMPLETE")
        return True
    else:
        logger.error("❌ Phase 3: Some tests failed")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)