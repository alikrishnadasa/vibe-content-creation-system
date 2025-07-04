#!/usr/bin/env python3
"""
Performance Analyzer

Analyze bottlenecks in the video generation pipeline to identify
optimization opportunities for reaching <0.7s target.
"""

import logging
import asyncio
import sys
import time
from pathlib import Path
import cProfile
import pstats
from contextlib import contextmanager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.real_content_generator import RealContentGenerator, RealVideoRequest

@contextmanager
def profile_context(description: str):
    """Context manager for profiling code sections"""
    logger.info(f"🔍 Profiling: {description}")
    start_time = time.time()
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        yield profiler
    finally:
        profiler.disable()
        elapsed = time.time() - start_time
        logger.info(f"⏱️ {description}: {elapsed:.3f}s")
        
        # Save profile stats
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        
        # Print top functions
        logger.info(f"📊 Top functions for {description}:")
        stats.print_stats(10)

async def analyze_initialization_performance():
    """Analyze initialization bottlenecks"""
    logger.info("🔧 ANALYZING INITIALIZATION PERFORMANCE")
    logger.info("=" * 60)
    
    with profile_context("Complete Initialization"):
        generator = RealContentGenerator(
            clips_directory="../MJAnime",
            metadata_file="../MJAnime/metadata_final_clean_shots.json", 
            scripts_directory="../11-scripts-for-tiktok",
            music_file="../Beanie (Slowed).mp3",
            output_directory="output"
        )
        
        # Time individual components
        with profile_context("Generator Initialize"):
            init_success = await generator.initialize()
        
        if not init_success:
            logger.error("❌ Initialization failed")
            return None
    
    return generator

async def analyze_video_generation_performance(generator):
    """Analyze video generation bottlenecks"""
    logger.info("\n🎬 ANALYZING VIDEO GENERATION PERFORMANCE")
    logger.info("=" * 60)
    
    # Test with a simple request
    request = RealVideoRequest(
        script_path="../11-scripts-for-tiktok/safe1.wav",
        script_name="perf_test",
        variation_number=1,
        caption_style="tiktok",
        music_sync=True,
        target_duration=15.0
    )
    
    # Profile each major stage
    with profile_context("Complete Video Generation"):
        with profile_context("Content Selection"):
            # This would profile the content selection phase
            pass
        
        with profile_context("Audio Processing"):
            # This would profile audio mixing
            pass
        
        with profile_context("Video Composition"):
            result = await generator.generate_video(request)
    
    return result

async def identify_bottlenecks():
    """Identify specific performance bottlenecks"""
    logger.info("\n🎯 IDENTIFYING PERFORMANCE BOTTLENECKS")
    logger.info("=" * 60)
    
    # Analyze different components
    bottlenecks = []
    
    # 1. File I/O bottlenecks
    logger.info("📁 Analyzing File I/O Performance...")
    start_time = time.time()
    
    # Test video file loading
    video_files = list(Path("../MJAnime").glob("*.mp4"))[:5]
    for video_file in video_files:
        file_size = video_file.stat().st_size / (1024 * 1024)
        load_start = time.time()
        # Simulate reading file metadata
        with open(video_file, 'rb') as f:
            f.read(1024)  # Read first 1KB for metadata
        load_time = time.time() - load_start
        logger.info(f"   {video_file.name}: {file_size:.1f}MB, {load_time*1000:.1f}ms")
    
    io_time = time.time() - start_time
    bottlenecks.append(("File I/O", io_time))
    
    # 2. Audio processing bottlenecks
    logger.info("🎵 Analyzing Audio Processing Performance...")
    start_time = time.time()
    
    # Test audio file analysis
    try:
        import librosa
        audio_file = "../11-scripts-for-tiktok/safe1.wav"
        if Path(audio_file).exists():
            y, sr = librosa.load(audio_file, duration=5.0)  # Load 5 seconds
            audio_time = time.time() - start_time
            bottlenecks.append(("Audio Processing", audio_time))
            logger.info(f"   Audio loaded: {len(y)} samples, {audio_time:.3f}s")
    except ImportError:
        logger.warning("   Librosa not available for audio analysis")
    
    # 3. Video processing bottlenecks
    logger.info("🎥 Analyzing Video Processing Performance...")
    start_time = time.time()
    
    try:
        from moviepy.editor import VideoFileClip
        if video_files:
            with VideoFileClip(str(video_files[0])) as clip:
                duration = clip.duration
                resolution = (clip.w, clip.h)
            video_time = time.time() - start_time
            bottlenecks.append(("Video Loading", video_time))
            logger.info(f"   Video loaded: {duration:.1f}s, {resolution}, {video_time:.3f}s")
    except Exception as e:
        logger.warning(f"   Video loading test failed: {e}")
    
    return bottlenecks

def suggest_optimizations(bottlenecks):
    """Suggest specific optimizations based on bottlenecks"""
    logger.info("\n💡 OPTIMIZATION RECOMMENDATIONS")
    logger.info("=" * 60)
    
    total_bottleneck_time = sum(time for _, time in bottlenecks)
    
    logger.info("🎯 Priority Optimizations (to reach <0.7s target):")
    
    # Sort bottlenecks by impact
    sorted_bottlenecks = sorted(bottlenecks, key=lambda x: x[1], reverse=True)
    
    for i, (component, time_taken) in enumerate(sorted_bottlenecks, 1):
        percentage = (time_taken / total_bottleneck_time) * 100 if total_bottleneck_time > 0 else 0
        logger.info(f"\n{i}. {component}: {time_taken:.3f}s ({percentage:.1f}% of bottlenecks)")
        
        if "File I/O" in component:
            logger.info("   💾 Optimizations:")
            logger.info("   • Pre-load frequently used clips into memory")
            logger.info("   • Implement clip caching with LRU eviction")
            logger.info("   • Use memory-mapped files for large video files")
            logger.info("   • Store clips in optimized format (e.g., compressed)")
        
        elif "Audio" in component:
            logger.info("   🎵 Optimizations:")
            logger.info("   • Pre-process and cache audio analysis results")
            logger.info("   • Use faster audio libraries (e.g., PyAudio, soundfile)")
            logger.info("   • Implement audio segment caching")
            logger.info("   • Optimize audio mixing algorithms")
        
        elif "Video" in component:
            logger.info("   🎥 Optimizations:")
            logger.info("   • Pre-load video metadata into database")
            logger.info("   • Use GPU acceleration for video processing")
            logger.info("   • Implement frame-level caching")
            logger.info("   • Optimize video encoding settings")
    
    logger.info("\n🚀 Implementation Strategy:")
    logger.info("1. IMMEDIATE (Target: 50% improvement):")
    logger.info("   • Implement clip metadata caching")
    logger.info("   • Pre-load most common clips")
    logger.info("   • Optimize audio processing pipeline")
    
    logger.info("\n2. SHORT-TERM (Target: 80% improvement):")
    logger.info("   • GPU acceleration for video operations")
    logger.info("   • Memory pooling for large assets")
    logger.info("   • Parallel processing optimizations")
    
    logger.info("\n3. ADVANCED (Target: 90%+ improvement):")
    logger.info("   • Custom video codec optimizations")
    logger.info("   • Hardware-specific acceleration")
    logger.info("   • Real-time streaming pipeline")

async def benchmark_current_performance():
    """Benchmark current performance for comparison"""
    logger.info("\n📊 CURRENT PERFORMANCE BENCHMARK")
    logger.info("=" * 60)
    
    try:
        generator = RealContentGenerator(
            clips_directory="../MJAnime",
            metadata_file="../MJAnime/metadata_final_clean_shots.json", 
            scripts_directory="../11-scripts-for-tiktok",
            music_file="../Beanie (Slowed).mp3",
            output_directory="output"
        )
        
        # Quick initialization
        init_start = time.time()
        await generator.initialize()
        init_time = time.time() - init_start
        
        # Generate single video for benchmarking
        request = RealVideoRequest(
            script_path="../11-scripts-for-tiktok/safe1.wav",
            script_name="benchmark_test",
            variation_number=1,
            caption_style="tiktok",
            music_sync=True,
            target_duration=10.0  # Shorter for faster benchmark
        )
        
        generation_start = time.time()
        result = await generator.generate_video(request)
        generation_time = time.time() - generation_start
        
        logger.info(f"📈 Performance Metrics:")
        logger.info(f"   Initialization: {init_time:.3f}s")
        logger.info(f"   Video Generation: {generation_time:.3f}s")
        logger.info(f"   Total Time: {init_time + generation_time:.3f}s")
        logger.info(f"   Success: {result.success}")
        
        # Calculate improvement needed
        current_avg = 8.0  # From production run
        target = 0.7
        improvement_needed = ((current_avg - target) / current_avg) * 100
        
        logger.info(f"\n🎯 Performance Gap Analysis:")
        logger.info(f"   Current Average: {current_avg:.1f}s")
        logger.info(f"   Target: {target:.1f}s")
        logger.info(f"   Improvement Needed: {improvement_needed:.1f}%")
        logger.info(f"   Speed Multiplier Required: {current_avg/target:.1f}x")
        
        return {
            'init_time': init_time,
            'generation_time': generation_time,
            'total_time': init_time + generation_time,
            'success': result.success
        }
        
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        return None

async def main():
    """Main performance analysis"""
    logger.info("🚀 PERFORMANCE ANALYSIS - VIDEO GENERATION OPTIMIZATION")
    logger.info("Target: Reduce generation time from 8.0s to <0.7s (90%+ improvement)")
    logger.info("=" * 80)
    
    # 1. Benchmark current performance
    benchmark = await benchmark_current_performance()
    
    # 2. Analyze initialization
    generator = await analyze_initialization_performance()
    
    if generator:
        # 3. Analyze video generation
        await analyze_video_generation_performance(generator)
    
    # 4. Identify bottlenecks
    bottlenecks = await identify_bottlenecks()
    
    # 5. Suggest optimizations
    suggest_optimizations(bottlenecks)
    
    logger.info("\n🎯 NEXT STEPS:")
    logger.info("1. Implement clip metadata caching system")
    logger.info("2. Create GPU-accelerated video processor")
    logger.info("3. Build memory pooling for asset management")
    logger.info("4. Optimize audio processing pipeline")
    logger.info("5. Test performance improvements incrementally")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print("\n✅ Performance analysis complete - ready for optimization!")
    else:
        print("\n❌ Performance analysis failed")
        sys.exit(1)